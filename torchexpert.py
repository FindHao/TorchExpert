import argparse
from torch import profiler
from torch._C._autograd import DeviceType
from torch._C._profiler import (_ProfilerEvent, _ExtraFields_TorchOp,
                                _ExtraFields_PyCCall, _ExtraFields_PyCall,
                                _EventType)
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
from common_func import get_latest_file, merge_interval, check_event_mem_related, print_all_event_time
from profile_event import ProfileEventSlim, IdleEvent
import torch
import json
from analysis_result import AnalysisResult
import numpy as np
from occupancy_calculator import CudaOccupancyCalculator


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
    """
    This class is used to profile the model and do the analysis.
    Attribute:
        prof: the profiler reference
        events_raw: the raw events from the profiler.
        profiler_config: the config for profiling
        json_trace: the json file of the profiling result
        cuda_kernels: cuda kernels, ProfilerEventSlim
    TorchExpert requires PyTorch >= August 8st, 2022
    """

    def __init__(self, analyze_json_only=True, model_name='', output_csv_file=None, profiler_folder='./logs/'):
        self.prof = None
        self.events_raw = []
        self.event_tree_roots = []
        self.events_kineto = []
        self.profiler_config = {
            "activities": [profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU],
            "profile_detailed": True,
            "profile_folder": profiler_folder,
            "nwarmup": 3
        }
        self.json_trace = None
        self.analysis_result = None
        self.analyze_json_only = analyze_json_only
        self.model_name = model_name
        self.output_csv_file = output_csv_file
        self.occup_calc = CudaOccupancyCalculator("8.0")
        self.idle_events = []


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

    def analyze_idleness(self):
        pass

    def analyze(self, json_path='./'):
        """
        This function is used to analyze the profiling result. Will be changed to add more features in the future.
        """
        print('\n')
        if self.analyze_json_only:
            slimevents, start_time_ns, end_time_ns, memcpy_time = self.get_events_from_json()
            self.load_json(json_path)
        else:
            slimevents, start_time_ns, end_time_ns, memcpy_time = self.get_cuda_events_from_profile()
        merged_slimevents = merge_interval(slimevents)
        # @Debug: print all the events in merged_slimevents
        # for event in merged_slimevents:
        #     for include_event in event.include_events:
        #         print("%s start from %.2fms, last for %.2fms" % (include_event.name(), (include_event.start_time_ns - start_time_ns)/1e6, (include_event.end_time_ns - include_event.start_time_ns)/1e6))
        # get all idleness
        self.idle_events = self.get_all_idleness(merged_slimevents)
        # @Debug: print all the idleness
        # for event in idle_events:
        #     print("Idle starts from %.2fms, lasts for %.2fms " % ((event.start_time_ns - start_time_ns)/1e6, (event.end_time_ns - start_time_ns)/1e6))
        self.analyze_idleness()
        sum_gpu_busy = 0
        for slimevent in merged_slimevents:
            # print(slimevent.start_us, slimevent.end_us)
            sum_gpu_busy += slimevent.end_time_ns - slimevent.start_time_ns
            # for event in slimevent.include_events:
            #     print(event.name())
        if start_time_ns == 0:
            print("Error: No events found.")
            return
        app_duration = end_time_ns - start_time_ns
        self.analysis_result = AnalysisResult(
            app_duration=app_duration, memcpy_time=memcpy_time, gpu_busy_time=sum_gpu_busy, model_name=self.model_name, output_csv_file=self.output_csv_file)
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
    parser.add_argument("-i","--input", type=str, default='./', help="the path of the json file or the folder containing the json files")
    parser.add_argument('-m', "--model_name", type=str, default='model', help="the name of the model")
    parser.add_argument('-o', "--output_csv_file", type=str, default='analysis_result.csv', help="the name of the output csv file")
    parser.add_argument("--analyze_json_only", type=bool, default=True, help="If True, will only analyze the json file. If False, will do the profiling and analysis of the json trace file.")
    parser.add_argument("--profiler_folder", type=str, default='./logs/', help="the folder to save the PyTorch profiler results")
    args = parser.parse_args()
    torchexpert = TorchExpert(model_name=args.model_name, output_csv_file=args.output_csv_file, analyze_json_only=args.analyze_json_only, profiler_folder=args.profiler_folder)
    torchexpert.analyze(args.json_path)
    