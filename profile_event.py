from torch._C._autograd import _ProfilerEvent
class ProfileEventSlim:
    """
    A simplified version of event class. 
    Attributes:
        duration_us: duration of the event in microseconds(us)
        start_us: start time of the event in microseconds(us)
        end_us: end time of the event in microseconds(us)
        include_events: a list of raw events that have overlaps
    """
    def __init__(self, event:_ProfilerEvent):
        self.duration_time_ns = event.duration_time_ns
        self.start_time_ns = event.start_time_ns
        self.end_time_ns = event.end_time_ns
        self.include_events = [event]
