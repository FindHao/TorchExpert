#!/usr/bin/env python3
import argparse
from typing import List
from torch import profiler
from torch._C._autograd import DeviceType
from torch._C._profiler import (_ProfilerEvent, _ExtraFields_TorchOp,
                                _ExtraFields_PyCCall, _ExtraFields_PyCall,
                                _EventType)
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
from common_func import get_latest_file, merge_interval, check_event_mem_related, print_all_event_time, get_all_events_in_path, check_event_in_gpu_trace
from profile_event import ProfileEventSlim, IdleEvent, TraceEvent
import torch
import json
from analysis_result import AnalysisResult
import numpy as np
from occupancy_calculator import CudaOccupancyCalculator
import os
import re
from collections import deque
from stream_assign import AllGraphs


INPUT_MODE_JSON = 0
INPUT_MODE_PROF = 1
# the default stream id in pytorch profiler is 7, not 0.
DEFAULT_STREAM_ID = 7
# the default stream id in stream assignment json is 0.
DEFAULT_STREAM_ID_JSON = 0

# add logger
import logging
logger = logging.getLogger(__name__)
# set logger level to info
logger.setLevel(logging.INFO)

# @FindHao TODO: how about other new events?
KINETO_EVENT_ON_CPU = [
    "cudaDeviceGetStreamPriorityRange",
    "cudaStreamGetPriority",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaStreamIsCapturing",
    "cudaFuncGetAttributes",
    "cudaStreamWaitEvent",
    "cudaLaunchKernel",
    "cudaFuncSetAttribute",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "cudaMemcpyAsync",
]

USE_KINDETO_API = False

buf_reg = re.compile(r'buf(\d+)')

class TorchExpert:
    def __init__(self, input_mode=INPUT_MODE_PROF, model_name='', output_csv_file=None, profiler_folder='./logs/', log_file=None, json_trace_path=None):
        self.prof = None
        # events_raw: the raw events from the profiler.
        self.events_raw = []
        # there could be multiple root events in the trace view
        self.event_tree_roots = []
        # save all CUDA kernels and memory related functions
        self.cuda_events = []
        # @deprecated
        self.events_kineto = []
        # this config is for pytorch profiler
        self.profiler_config = {
            "activities": [profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU],
            "profile_detailed": True,
            "profile_folder": profiler_folder,
            "nwarmup": 3
        }
        # json_trace: dict, the json trace file
        self.json_trace = None
        # json_trace_path: str, the path of the json trace file
        self.json_trace_path = json_trace_path
        # It is an instance of AnalysisResult
        self.analysis_result = None
        # if it is true, will do the offline analysis
        self.input_mode = input_mode
        self.model_name = model_name
        # store the gpu active time analysis
        self.output_csv_file = output_csv_file
        # @deprecated
        self.occup_calc = CudaOccupancyCalculator("8.0")
        # I create virtual idle events to represent the idleness between raw events
        self.idle_events = []
        # store the idleness analysis for now
        self.log_file = log_file
        self.start_time_ns = None
        # saved all graphs collected from stream_assignment.json.
        self.all_graphs = None
        # it saves the events in aside streams but don't have overlapping with other events
        self.non_overlapping_kernels_ordered_list = []
        # TODO: for now, it only supports models with one forward graph and one backward graph.
        self.stream_assignment = {}
        # saved the picked cuda events in the multi-stream mode. In profiler trace generated by torch/benchmarks/dynamo, there are multiple iterations. Each iteration includes one expected and actual part.We need to pick an actual one to analyze.
        self.picked_cuda_events = []
        # it only saves the stream assignment of nodes in aside streams appearing in trace. Some nop node in the graph may not appear in the trace.
        self.stream_assigned_in_trace = {}

    def set_profile_config(self, activity_groups, profile_detailed, profile_folder, nwarmup):
        self.profiler_config = {
            "activities": activity_groups,
            "profile_detailed": profile_detailed,
            "profile_folder": profile_folder,
            "nwarmup": nwarmup
        }

    def set_profile(self, prof):
        """
        If the profiling happens outside this class, you can set the profile reference here.
        """
        self.prof = prof
        if USE_KINDETO_API:
            self.events_raw = prof.profiler.kineto_results.events()
        else:
            self.event_tree_roots = prof.profiler.kineto_results.experimental_event_tree()
            self.events_raw = list(traverse_bfs(self.event_tree_roots))

    def profile(self, func, *args, **kwargs):
        """
        This function is used to profile the model. It is not necessary to call this function. 
        You can directly use the profiler outside this class.
        """
        nwarmup = int(self.profiler_config["nwarmup"])
        with profiler.profile(
            schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1),
            activities=self.profiler_config["activities"],
            record_shapes=self.profiler_config["profile_detailed"],
            profile_memory=self.profiler_config["profile_detailed"],
            with_stack=self.profiler_config["profile_detailed"],
            with_flops=self.profiler_config["profile_detailed"],
            on_trace_ready=profiler.tensorboard_trace_handler(
                self.profiler_config["profile_folder"]),
        ) as prof:
            iter_range = nwarmup+1
            for _i in range(iter_range):
                func(*args, **kwargs)
                # Need to sync here to match run_one_step()'s timed run.
                torch.cuda.synchronize()
                # The last call of prof.step() will clean the profile,
                # so ignore it in the last iteration.
                if _i != iter_range - 1:
                    prof.step()
        # print(prof.key_averages(group_by_input_shape=True).table(
        #     sort_by="cpu_time_total", row_limit=30))
        self.set_profile(prof)

    def get_all_idleness(self, events: List[ProfileEventSlim]):
        """
        This function is used to get the idleness of the events.
        Args:
            events: a sorted list of events by start time
        Returns:
            a list of idleness event
        """
        if len(events) == 0:
            return []
        idle_events = []
        last_end_event = events[0]
        last_end_time_ns = events[0].end_time_ns
        # print_all_event_time(events)
        for i in range(1, len(events)):
            duration = events[i].start_time_ns - last_end_time_ns
            # ignore the idleness less than 0.01ms
            if duration > 0.01 * 1e6:
                idle_events.append(IdleEvent(start_time_ns=last_end_time_ns,
                                   end_time_ns=events[i].start_time_ns, left_event=last_end_event, right_event=events[i]))
            last_end_time_ns = events[i].end_time_ns
            last_end_event = events[i]
        # print_all_event_time(idle_events)
        return idle_events

    def get_events_from_json(self):
        slimevents = []
        end_time_ns = 0
        start_time_ns = 0
        memcpy_time = 0
        for event in self.json_trace['traceEvents']:
            if event.get('cat', '').lower() == 'kernel' or check_event_mem_related(event):
                dur = event['dur']*1e3
                ts = event['ts']*1e3
                te = ts + dur
                slimevents.append(ProfileEventSlim(
                    event=None, duration_time_ns=dur, start_time_ns=ts, end_time_ns=te))
                end_time_ns = max(end_time_ns, te)
                if start_time_ns == 0:
                    start_time_ns = ts
                else:
                    start_time_ns = min(start_time_ns, ts)
                if check_event_mem_related(event):
                    memcpy_time += dur
        return slimevents, start_time_ns, end_time_ns, memcpy_time

    def get_cuda_events_from_kineto(self):
        slimevents = []
        end_time_ns = 0
        start_time_ns = self.events_raw[0].start_us(
        ) * 1e3 if len(self.events_raw) else 0
        memcpy_time = 0
        for event in self.events_raw:
            if event.device_type() == DeviceType.CUDA:
                slimevents.append(ProfileEventSlim(event))
                # @Future: Update to _ProfilerEvent. The kineto event only has us resolution.
                end_time_ns = max(
                    end_time_ns, (event.start_us() + event.duration_us())*1e3)
                start_time_ns = min(start_time_ns, event.start_us() * 1e3)
                # @todo: check if the name is correct
                if event.name().strip().startswith("Memcpy"):
                    memcpy_time += event.duration_us() * 1e3
        return slimevents, start_time_ns, end_time_ns, memcpy_time

    def get_cuda_events(self):
        """
        In this function, the slim events are shadow of the raw events.
        """
        slimevents = []
        end_time_ns = 0
        start_time_ns = self.events_raw[0].start_time_ns if len(
            self.events_raw) else 0
        memcpy_time = 0
        for event in self.events_raw:
            if event.tag == _EventType.Kineto:
                if event.name.strip() in KINETO_EVENT_ON_CPU:
                    continue
                if event.name.strip().startswith("Memcpy"):
                    memcpy_time += event.duration_time_ns
                if event.parent.name.strip() not in ['cudaLaunchKernel', 'cudaMemcpyAsync']:
                    print("TorcheExpert-> Unexpected kernel ",
                          event.name.strip())
                slimevents.append(ProfileEventSlim(event))
                end_time_ns = max(end_time_ns, event.end_time_ns)
                if start_time_ns == 0:
                    start_time_ns = event.start_time_ns
                else:
                    start_time_ns = min(start_time_ns, event.start_time_ns)
            else:
                # @FindHao TODO: check other events
                pass

        return slimevents, start_time_ns, end_time_ns, memcpy_time

    def get_cuda_events_from_profile(self):
        # TODO: do we still need two different functions?
        if USE_KINDETO_API:
            return self.get_cuda_events_from_kineto()
        else:
            return self.get_cuda_events()

    def analyze_idleness(self):
        """
        This function is used to analyze the root causes of the idleness.
        """
        for idle_event in self.idle_events:
            # the last raw CUDA event before idle
            left_raw_event = idle_event.left_event.include_events[-1]
            # the event near to LCA in the path from left_raw_event to root
            left_raw_event_top = None
            # the first raw CUDA event after idle
            right_raw_event = idle_event.right_event.include_events[0]
            # the event near to LCA in the path from right_raw_event to root
            right_raw_event_top = None
            # LCA problem. Could be optimized.
            all_events_left = get_all_events_in_path(left_raw_event)
            all_events_right = get_all_events_in_path(right_raw_event)
            min_len = min(len(all_events_left), len(all_events_right))
            lca = None
            # @TODO: add a fake root node to avoid this check
            if all_events_left[-1] != all_events_right[-1]:
                raise Exception("They don't share the same root event.")
            for i in range(2, min_len+1):
                if all_events_left[len(all_events_left) - i] != all_events_right[len(all_events_right) - i]:
                    lca = all_events_left[len(all_events_left) - i + 1]
                    left_raw_event_top = all_events_left[len(
                        all_events_left) - i]
                    right_raw_event_top = all_events_right[len(
                        all_events_right) - i]
                    break
            assert lca is not None
            self.analysis_result.idle_event_pairs.append(
                (idle_event, lca, left_raw_event, left_raw_event_top, right_raw_event, right_raw_event_top))

    def build_tree_from_json(self):
        data = self.json_trace

        # Remove "ac2g" events and events with no duration and cat
        events = []
        for event in data["traceEvents"]:
            if event.get("dur", 0) == 0 or event.get("cat", "") == "":
                continue
            if event["cat"] == "ac2g":
                continue
            events.append(event)
        # Sort events based on timestamp
        events.sort(key=lambda x: x["ts"])

        first_node = events[0]
        assert first_node['tid'] == 'PyTorch Profiler'
        root = TraceEvent(parent=None, **first_node)
        stack = [root]
        external_id_map = {}

        for event in events[1:]:
            start_time = event["ts"]
            end_time = start_time + event.get("dur", 0)

            # As the node is being added, the parent will be the last node in the stack
            current_node = TraceEvent(parent=stack[-1], **event)
            if current_node.__dict__.get("External id", None) is not None and not check_event_in_gpu_trace(event):
                external_id_map[current_node.__dict__[
                    "External id"]] = current_node
            # Handle kernel events differently
            if check_event_in_gpu_trace(event):
                stream_id = event["args"]["stream"]
                # add stream attribute to the node
                setattr(current_node, "stream", stream_id)
                self.cuda_events.append(current_node)
                ext_id = event["args"]["External id"]
                caller_node = external_id_map.get(ext_id, None)
                assert caller_node is not None
                if caller_node:
                    caller_node.children.append(current_node)
                    current_node.parent = caller_node
            else:
                while stack:
                    if stack[-1].ts <= start_time and (stack[-1].ts + stack[-1].__dict__.get("dur", 0)) >= end_time:
                        stack[-1].children.append(current_node)
                        current_node.parent = stack[-1]
                        stack.append(current_node)
                        break
                    else:
                        stack.pop()
        self.event_tree_roots.append(root)

    def analyze(self, json_path='./'):
        """
        This function is used to analyze the profiling result. Will be changed to add more features in the future.
        """
        print('\n')
        if self.input_mode == INPUT_MODE_JSON:
            self.load_json(json_path)
            slimevents, start_time_ns, end_time_ns, memcpy_time = self.get_events_from_json()
        else:
            slimevents, start_time_ns, end_time_ns, memcpy_time = self.get_cuda_events_from_profile()
        slimevents, start_time_ns, end_time_ns, memcpy_time
        merged_slimevents = merge_interval(slimevents)
        sum_gpu_busy = 0
        for slimevent in merged_slimevents:
            # print(slimevent.start_us, slimevent.end_us)
            sum_gpu_busy += slimevent.end_time_ns - slimevent.start_time_ns
            # for event in slimevent.include_events:
            #     print(event.name())
        if start_time_ns == 0:
            print("Error: No events found.")
            return
        self.start_time_ns = start_time_ns
        app_duration = end_time_ns - start_time_ns
        self.analysis_result = AnalysisResult(
            app_duration=app_duration, memcpy_time=memcpy_time, gpu_busy_time=sum_gpu_busy, model_name=self.model_name, output_csv_file=self.output_csv_file, log_file=self.log_file, start_time_ns=start_time_ns)

        if self.input_mode == INPUT_MODE_PROF:
            self.idle_events = self.get_all_idleness(merged_slimevents)
            self.analyze_idleness()
            self.analysis_result.print_to_log()

        # self.analysis_result.print_as_str()
        self.analysis_result.print_as_csv()

    def analyze_multi_stream(self):
        assert len(self.event_tree_roots) != 0
        # Yueming: may need update for online analyze
        assert len(self.event_tree_roots) == 1
        # root should be the `PyTorch Profiler (0)`
        root = self.event_tree_roots[0]
        # This is for torch/benchmarks/dynamo profiling results. Need to update for normal profile trace
        num_iter = len(root.children) // 2
        picked_iter = num_iter // 2
        # an actual node in the tree
        picked_node = root.children[picked_iter]
        # get all kernels in aside streams. is it always correct to assume the default stream is 7?
        # tmp_cuda_events = self.cuda_events.copy()
        tmp_cuda_events = []
        queue = deque([picked_node])
        while queue:
            node = queue.popleft()
            if check_event_in_gpu_trace(node.__dict__):
                tmp_cuda_events.append(node)
            for child in node.children:
                queue.append(child)
        tmp_cuda_events.sort(key=lambda x: x.ts)
        self.picked_cuda_events = tmp_cuda_events
        non_overlapping_kernels = []
        for i, current_event in enumerate(tmp_cuda_events[:-1]):
            # Check if it's not in DEFAULT_STREAM_ID.
            if current_event.stream != DEFAULT_STREAM_ID:
                next_event = tmp_cuda_events[i+1]
                # Check if they are not overlapping.
                if (next_event.ts >= (current_event.ts + current_event.dur)):
                    non_overlapping_kernels.append(current_event)
                elif (next_event.stream != DEFAULT_STREAM_ID):
                    non_overlapping_kernels.append(next_event)
        for event in non_overlapping_kernels:
            if event not in self.non_overlapping_kernels_ordered_list:
                self.non_overlapping_kernels_ordered_list.append(event)
        assert self.all_graphs is not None
        
    def get_predecessor_stream(self, ssnode, ssgraph, tid):
        """Get the predecessor's stream id. There are some cases that the predecessors are abnormal nodes. For example, buf12->buf13->buf14 while buf13 is an assert_size node. Then, buf13 will not appear in the trace file and no GPU kernels generated. In this case, we need to find the stream id of buf12. More complicated cases are possible such as multiple predecessors.
        """
        target_stream = -1
        if len(ssnode.predecessors) != 0:
            found = False
            for predecessor_name in ssnode.predecessors:
                predecessor = ssgraph.name_mapping[predecessor_name]
                if predecessor.stream_id == DEFAULT_STREAM_ID_JSON:
                    target_stream = DEFAULT_STREAM_ID_JSON
                    found = True
                    break
                else:
                    if predecessor.name in self.stream_assigned_in_trace[tid]:
                        target_stream = predecessor.stream_id
                        found = True
                        break
                    else:
                        # the predecessor is an empty node such as assert_size
                        continue
            if not found:
                for predecessor_name in ssnode.predecessors:
                    predecessor = ssgraph.name_mapping[predecessor_name]
                    search_result = self.get_predecessor_stream(predecessor, ssgraph, tid)
                    if search_result != -1:
                        found = True
                        target_stream = search_result
        else:
            target_stream = DEFAULT_STREAM_ID_JSON
        return target_stream
    
    def check_one_graph(self, tid, ssgraph):
        """update the stream assignments for kernels in non_overlapping_kernels_ordered_list.
        """
        updated_nodes = set()
        # update stream assignment
        for event in self.non_overlapping_kernels_ordered_list:
            if event.parent.tid != tid:
                continue
            parent = event
            found = False
            while(not parent.name.startswith("CompiledFunction")):
                match = buf_reg.match(parent.name)
                if match:
                    ssnode = ssgraph.name_mapping[parent.name]
                    target_stream = self.get_predecessor_stream(ssnode, ssgraph, tid)
                    if target_stream == -1 or (target_stream == ssnode.stream_id and ssnode.name not in updated_nodes):
                        raise Exception(f"Can't find a better stream assignment for node {ssnode.name}. current result: {ssnode.stream_id} -> {target_stream}")
                    updated_nodes.add(ssnode.name)
                    print(f"Node {ssnode.name}'s stream has been changed from {ssnode.stream_id} to {target_stream}.")
                    # update stream assignment
                    ssnode.stream_id = target_stream
                    found = True
                    break
                parent = parent.parent
            if not found:
                print(f"Can't find the corresponding ssnode for event(ts {event.ts})")
        self.check_empty_nodes(tid, ssgraph)
    
    def check_empty_nodes(self, tid, ssgraph):
        """This function is used to put the empty nodes to the same stream as its predecessor. 
        """
        for node in ssgraph.ssnodes:
            if node.stream_id != DEFAULT_STREAM_ID_JSON and node.name not in self.stream_assigned_in_trace[tid]:
                target_stream = self.get_predecessor_stream(node, ssgraph, tid)
                if target_stream == -1:
                    raise Exception(f"Can't find a better stream assignment for empty node {node.name}. current result: {node.stream_id} -> {target_stream}")
                if target_stream == node.stream_id:
                    logger.info(f"Empty node {node.name}'s better stream is the same as its current stream: {node.stream_id}.")
                else:
                    logger.info(f"Empty node {node.name}'s stream has been changed from {node.stream_id} to {target_stream}.")
                node.stream_id = target_stream

    def update_stream_assignment(self, export_graph_path):
        """Compare the stream assignment collected from the trace file and stream_assignment.json. Update the assigment doesn't bring any benefit in stream_assignment.json.
        """
        # there are only two threads. One for forward and one for backward.
        thread_ids = set()
        for event in self.picked_cuda_events:
            parent_event_tid = event.parent.tid
            thread_ids.add(parent_event_tid)
        assert len(thread_ids) == 2
        forward_tid = min(thread_ids) # forward
        backward_tid = max(thread_ids) # backward
        graph_names = self.all_graphs.graphs.keys()
        reg = re.compile(r'graph_(\d+)')
        graph_ids = []
        for graph_name in graph_names:
            match = reg.match(graph_name)
            if match:
                graph_ids.append(int(match.group(1)))
        assert len(graph_ids) == 2
        forward_graph = self.all_graphs.graphs['graph_{}'.format(min(graph_ids))]
        backward_graph = self.all_graphs.graphs['graph_{}'.format(max(graph_ids))]

        for event in self.picked_cuda_events:
            parent_event_tid = event.parent.tid
            if parent_event_tid == forward_tid:
                target_graph = forward_graph
            else:
                target_graph = backward_graph
            # collect all nodes' stream assignments in aside streams
            if event.stream != DEFAULT_STREAM_ID:
                found = False
                parent = event
                # dead loop possible?
                while(not parent.name.startswith("CompiledFunction")):
                    match = buf_reg.match(parent.name)
                    if match:
                        ssnode = target_graph.name_mapping[parent.name]
                        if parent_event_tid not in self.stream_assigned_in_trace:
                            self.stream_assigned_in_trace[parent_event_tid] = {}
                        # use its stream_id in json rather than trace because we need to update the stream assignment in json. This for loop here just collects which nodes are actually in aside streams.
                        self.stream_assigned_in_trace[parent_event_tid][ssnode.name] = ssnode.stream_id
                        found = True
                        break
                    parent = parent.parent
                if not found:
                    logger.warning(f"Can't find the corresponding ssnode for event(ts {event.ts})")

        self.check_one_graph(forward_tid, forward_graph)
        
        print("forward phase finished.")
        self.check_one_graph(backward_tid, backward_graph)
        print("backward phase finished.")
        if export_graph_path is not None:
            parent_folder_path = os.path.dirname(export_graph_path)
            if not os.path.exists(parent_folder_path):
                os.makedirs(parent_folder_path)
            self.all_graphs.export_graphs(export_graph_path)

    def load_json(self, json_path):
        """
        This function is used to load the profiling result from a json file.
        Args:
            json_file: the path of the json file
        """
        # check if json_path is a file or a folder
        if os.path.isfile(json_path):
            json_file = json_path
        else:
            json_file = get_latest_file(json_path)
        if json_file is None:
            print("Error: No json file found.")
            return
        print("Analyzing json file: {}".format(json_file))
        if self.model_name == '':
            self.model_name = os.path.basename(json_file).split('.')[0]
        with open(json_file, "r") as f:
            self.json_trace = json.load(f)

    def build_all_graphs(self, json_path):
        self.all_graphs = AllGraphs(json_path)

    def get_avg_kernel_occupancy(self):
        """
        This function is used to get the occupancy of all kernels in the trace file.
        Returns:
            a dictionary of kernel occupancy
        """
        sum_duration = 0
        kernel_occupancies = []
        assert self.json_trace is not None
        for event in self.json_trace['traceEvents']:
            if event.get('cat', '').lower() == 'kernel':
                duration = event['dur']*1e3
                block_size = np.prod(event['args']['block'])
                reg_per_thread = event['args']['registers per thread']
                smem = event['args'].get('shared memory', 0)
                self.occup_calc.set_inputs(
                    block_size, reg_per_thread, "8.0", smem)
                occupancy = self.occup_calc.occupancyOfMultiprocessor()
                occupancy_in_trace = event['args'].get(
                    'est. achieved occupancy %', 0)
                # if occupancy*100 !gccgjudnvkdcghjcetthjvkeggdnkggicy in the trace file: ", occupancy_in_trace)
                kernel_occupancies.append(occupancy*duration)
                sum_duration += duration

        # print("kernel_occupancies: ", kernel_occupancies)
        avg_occupancy = sum(kernel_occupancies) / \
            sum_duration * 100 if sum_duration > 0 else 0
        return avg_occupancy

# def analyze_multi_report(te_single, te_multi):
#     te_single.load_json(te_single.json_trace_path)
#     te_single.build_tree_from_json()
#     te_multi.load_json(te_multi.json_trace_path)
#     te_multi.build_tree_from_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default='./',
                        help="The path of the json file or the path containing the json files. If you specify a path, torchexpert will search for the latest json file in that path.")
    parser.add_argument('--multistream', type=str,
                        help="the trace file enabled multi-stream")
    parser.add_argument('-m', "--model_name", type=str,
                        help="the name of the model")
    parser.add_argument('-s', "--stream_assignment", type=str,
                        help="the json file saved stream assignment")
    parser.add_argument('-o', "--output_csv_file", type=str,
                        default='analysis_result.csv', help="the name of the output csv file")
    parser.add_argument('--export_graph', type=str, help="the json path to save the updated stream assignment")
    parser.add_argument("--log_file", type=str, default='torchexpert.log',
                        help="the log file to save the idleness analysis results")
    args = parser.parse_args()
    torchexpert = TorchExpert(model_name=args.model_name, output_csv_file=args.output_csv_file,
                              input_mode=INPUT_MODE_JSON, log_file=args.log_file, json_trace_path=args.input)
    # disable the idleness analysis temporarily for multi-stream analysis
    # torchexpert.analyze(args.input)
    torchexpert.load_json(args.input)
    torchexpert.build_tree_from_json()
    if args.stream_assignment is not None:
        torchexpert.build_all_graphs(args.stream_assignment)
        torchexpert.all_graphs.print_streams()
    torchexpert.analyze_multi_stream()
    torchexpert.update_stream_assignment(args.export_graph)

    # else:
    #     # this mode is used to analyze the difference between the optimized trace and the original trace. So we need another torchexpert instance.
    #     torchexpert_opt = TorchExpert(model_name=args.model_name, output_csv_file=args.output_csv_file,
    #                                   input_mode=INPUT_MODE_JSON, log_file=args.log_file, json_trace_path=args.multistream)
    #     analyze_multi_report(torchexpert, torchexpert_opt)
