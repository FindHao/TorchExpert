from asyncio import events
from torch import profiler
from torch._C._autograd import _ProfilerEvent, DeviceType, _EventType
from torch.profiler._pattern_matcher import eventTreeBFS
from common_func import *
from profile_event import ProfileEventSlim
import torch


KINETO_EVENT_ON_CPU = [
    "cudaDeviceGetStreamPriorityRange",
    "cudaStreamGetPriority",
    "cudaDeviceSynchronize",
    "cudaStreamIsCapturing",
    "cudaFuncGetAttributes",
    "cudaStreamWaitEvent",
    "cudaLaunchKernel",
    "cudaFuncSetAttribute",
]


class TorchExpert:
    """
    This class is used to profile the model and do the analysis.
    Attribute:
        prof: the profiler reference
        events_raw: the raw events from the profiler.
        profiler_config: the config for profiling
    TorchExpert requires PyTorch >= August 8st, 2022
    """

    def __init__(self):
        self.prof = None
        self.events_raw = []
        self.event_tree_roots = []
        self.events_kineto = []
        self.profiler_config = {
            "activities": [profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU],
            "profile_detailed": True,
            "profile_folder": "./logs/",
            "nwarmup": 3
        }

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
        self.event_tree_roots = prof.profiler.kineto_results.experimental_event_tree()
        self.events_raw = list(eventTreeBFS(self.event_tree_roots))
        self.events_kineto = prof.profiler.kineto_results.events()

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
            for _i in range(nwarmup + 1):
                func(*args, **kwargs)
                # Need to sync here to match run_one_step()'s timed run.
                torch.cuda.synchronize()
                # The last call of prof.step() will clean the profile,
                # so ignore it in the last iteration.
                if _i != nwarmup:
                    prof.step()
        self.prof = prof
        # print(prof.key_averages(group_by_input_shape=True).table(
        #     sort_by="cpu_time_total", row_limit=30))
        self.event_tree_roots = prof.profiler.kineto_results.experimental_event_tree()
        self.events_raw = list(eventTreeBFS(self.event_tree_roots))
        self.events_kineto = prof.profiler.kineto_results.events()

    def analyze(self):
        """
        This function is used to analyze the profiling result. Will be changed to add more features in the future.
        """
        slimevents = []
        end_time_ns = 0
        start_time_ns = self.events_raw[0].start_time_ns if len(
            self.events_raw) else 0
        for event in self.events_raw:
            if event.tag == _EventType.Kineto:
                if event.name().strip() in KINETO_EVENT_ON_CPU:
                    continue
                slimevents.append(ProfileEventSlim(event))
                end_time_ns = max(end_time_ns, event.end_time_ns)
                start_time_ns = min(start_time_ns, event.start_time_ns)
        merged_slimevents = merge_interval(slimevents)
        sum_gpu_active = 0
        for slimevent in merged_slimevents:
            # print(slimevent.start_us, slimevent.end_us)
            sum_gpu_active += slimevent.end_time_ns - slimevent.start_time_ns
            # for event in slimevent.include_events:
            #     print(event.name())
        print("GPU active time:", sum_gpu_active / 1e3, "ms")
        if start_time_ns != 0:
            print("GPU active time ratio: %.2f%%" %
                  (sum_gpu_active * 100 / (end_time_ns - start_time_ns)))
        else:
            print("Missing total time")
