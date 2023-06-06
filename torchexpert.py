import argparse
from torch import profiler
from torch._C._autograd import DeviceType
from torch._C._profiler import (_ProfilerEvent, _ExtraFields_TorchOp,
                                _ExtraFields_PyCCall, _ExtraFields_PyCall,
                                _EventType)
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
from common_func import get_latest_file, merge_interval, check_event_mem_related, print_all_event_time, get_all_events_in_path
from profile_event import ProfileEventSlim, IdleEvent
import torch
import json
from analysis_result import AnalysisResult
import numpy as np
from occupancy_calculator import CudaOccupancyCalculator
import os

INPUT_MODE_JSON=0
INPUT_MODE_PROF=1


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

class TorchExpert:
    def __init__(self, input_mode=INPUT_MODE_PROF, model_name='', output_csv_file=None, profiler_folder='./logs/', log_file=None):
        self.prof = None
        # events_raw: the raw events from the profiler. 
        self.events_raw = []
        # there could be multiple root events in the trace view
        self.event_tree_roots = []
        # @deprecated
        self.events_kineto = []
        # this config is for pytorch profiler
        self.profiler_config = {
            "activities": [profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU],
            "profile_detailed": True,
            "profile_folder": profiler_folder,
            "nwarmup": 3
        }
        # json_trace: the json file of the profiling result
        self.json_trace = None
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
            iter_range=nwarmup+1
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

    def get_all_idleness(self, events: list[ProfileEventSlim]):
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
                idle_events.append(IdleEvent(start_time_ns=last_end_time_ns, end_time_ns=events[i].start_time_ns, left_event=last_end_event, right_event=events[i]))
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
        start_time_ns = self.events_raw[0].start_us() * 1e3 if len(self.events_raw) else 0
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

    """
    In this function, the slim events are shadow of the raw events.
    """
    def get_cuda_events(self):
        slimevents = []
        end_time_ns = 0
        start_time_ns = self.events_raw[0].start_time_ns if len(self.events_raw) else 0
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
        if USE_KINDETO_API:
            return self.get_cuda_events_from_kineto()
        else:
            return self.get_cuda_events()

    """
    This function is used to analyze the root causes of the idleness.
    """
    def analyze_idleness(self):
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
                    left_raw_event_top = all_events_left[len(all_events_left) - i]
                    right_raw_event_top = all_events_right[len(all_events_right) - i]
                    break
            assert lca is not None
            self.analysis_result.idle_event_pairs.append((idle_event, lca, left_raw_event, left_raw_event_top, right_raw_event, right_raw_event_top))


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
                self.occup_calc.set_inputs(block_size, reg_per_thread, "8.0", smem)
                occupancy = self.occup_calc.occupancyOfMultiprocessor()
                occupancy_in_trace = event['args'].get('est. achieved occupancy %', 0)
                # if occupancy*100 !gccgjudnvkdcghjcetthjvkeggdnkggicy in the trace file: ", occupancy_in_trace)
                kernel_occupancies.append(occupancy*duration)
                sum_duration += duration
                
        # print("kernel_occupancies: ", kernel_occupancies)
        avg_occupancy = sum(kernel_occupancies)/sum_duration * 100 if sum_duration > 0 else 0
        return avg_occupancy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=str, default='./', help="The path of the json file or the path containing the json files. If you specify a path, torchexpert will search for the latest json file in that path.")
    parser.add_argument('-m', "--model_name", type=str, help="the name of the model")
    parser.add_argument('-o', "--output_csv_file", type=str, default='analysis_result.csv', help="the name of the output csv file")
    parser.add_argument("--log_file", type=str, default='torchexpert.log', help="the log file to save the idleness analysis results")
    args = parser.parse_args()
    torchexpert = TorchExpert(model_name=args.model_name, output_csv_file=args.output_csv_file, input_mode=INPUT_MODE_JSON, log_file=args.log_file)
    torchexpert.analyze(args.input)
    