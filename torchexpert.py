from asyncio import events
from torch import profiler
from torch._C._autograd import _ProfilerEvent, DeviceType
from common_func import *
from profile_event import ProfileEventSlim
import torch


class TorchExpert:
    def __init__(self):
        self.prof = None
        self.events_raw = []
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
        self.prof = prof
        self.events_raw = prof.profiler.kineto_results.events()

    def profile(self, func, *args, **kwargs):
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
                if _i != nwarmup:
                    prof.step()
        self.prof = prof
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=30))
        self.events_raw = prof.profiler.kineto_results.events()

    def analyze(self):
        slimevents = []
        end_us = 0
        start_us = self.events_raw[0].start_us() if len(self.events_raw) else 0
        for event in self.events_raw:
            if event.device_type() == DeviceType.CUDA:
                slimevents.append(ProfileEventSlim(event))
                # @Yueming Hao: may need to change it in the future
                end_us = max(
                    end_us, event.start_us() + event.duration_us())
                start_us = min(start_us, event.start_us())
        merged_slimevents = merge_interval(slimevents)
        sum_gpu_active = 0
        for slimevent in merged_slimevents:
            # print(slimevent.start_us, slimevent.end_us)
            sum_gpu_active += slimevent.end_us - slimevent.start_us
            # for event in slimevent.include_events:
            #     print(event.name())
        print("GPU active time:", sum_gpu_active/1e3, "ms")
        if start_us != 0:
            print("GPU active time ratio: %.2f%%" %
                  (sum_gpu_active*100/(end_us-start_us)))
        else:
            print("Missing total time")
