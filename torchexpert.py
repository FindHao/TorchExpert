from torch import profiler
import torch
from torch._C._autograd import _ProfilerEvent, DeviceType
from common_func import *
from profile_event import ProfileEventSlim


class TorchExpert:
    def __init__(self):
        self.prof = None
        self.events_raw = []

    def profile(self, func, *args, **kwargs):
        activity_groups = []
        activity_groups.append(profiler.ProfilerActivity.CUDA)
        activity_groups.append(profiler.ProfilerActivity.CPU)
        profile_detailed = True
        with profiler.profile(
            schedule=profiler.schedule(wait=0, warmup=0, active=1),
            activities=activity_groups,
            record_shapes=profile_detailed,
            profile_memory=profile_detailed,
            with_stack=profile_detailed,
            with_flops=profile_detailed,
            on_trace_ready=profiler.tensorboard_trace_handler("./logs/")
        ) as prof:
            func(*args, **kwargs)
        self.prof = prof
        self.events_raw = prof.profiler.kineto_results.events()

    def analyze(self):
        slimevents = []
        total_time = 0
        for event in self.events_raw:
            if event.device_type() == DeviceType.CUDA:
                slimevents.append(ProfileEventSlim(event))
            # @Yueming Hao: is this compatible with all casses?
            if event.name().startswith("ProfilerStep"):
                total_time = event.duration_us()
        merged_slimevents = merge_interval(slimevents)
        sum_gpu_active = 0
        for slimevent in merged_slimevents:
            print(slimevent.start_us, slimevent.end_us)
            sum_gpu_active += slimevent.end_us - slimevent.start_us
            for event in slimevent.include_events:
                print(event.name())
        print("GPU active time:", sum_gpu_active/1e3, "ms")
        if total_time != 0:
            print("GPU active time ratio: %.2f%%" %
                  (sum_gpu_active*100/total_time))
        else:
            print("Missing total time")
