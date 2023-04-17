from torch._C._profiler import _ProfilerEvent
class ProfileEventSlim:
    """
    A simplified version of event class. It merges all the raw events that have overlaps.
    Attributes:
        duration_us: duration of the event in microseconds(us)
        start_us: start time of the event in microseconds(us)
        end_us: end time of the event in microseconds(us)
        include_events: a list of raw events that have overlaps
    """
    def __init__(self, event=None, duration_time_ns=None, start_time_ns=None, end_time_ns=None, include_events=None):
        if event is not None:
            if type(event) is _ProfilerEvent:
                self.duration_time_ns = event.duration_time_ns
                self.start_time_ns = event.start_time_ns
                self.end_time_ns = event.end_time_ns
            else:
                self.duration_time_ns = event.duration_us() * 1e3
                self.start_time_ns = event.start_us() * 1e3
                self.end_time_ns = (event.start_us() + event.duration_us()) * 1e3
            self.include_events = [event]
        else:
            self.duration_time_ns = duration_time_ns
            self.start_time_ns = start_time_ns
            self.end_time_ns = end_time_ns
            self.include_events = []


class TraceEvent:
    """
    A class to store the trace event.
    """
    def __init__(self, tmp_dict=None) -> None:
        if tmp_dict is not None:
            self.name = tmp_dict["name"]
            self.ph = tmp_dict["ph"]
            self.pid = tmp_dict["pid"]
            self.tid = tmp_dict["tid"]
            self.start_time = tmp_dict["ts"]
            self.duration = tmp_dict["dur"]
            self.args = tmp_dict["args"]
        else:
            self.name = None
            self.ph = None
            self.pid = None
            self.tid = None
            self.start_time = None
            self.duration = None
            self.args = None


class Shadow_ProfilerEvent():
    """
    A shadow class of _ProfilerEvent to add more attributes.
    """
    def __init__(self, event=None, parent=None):
        if event is not None:
            self.tag = event.tag
            self.id = event.id
            self.correlation_id = event.correlation_id
            self.start_tid = event.start_tid
            self.start_time_ns = event.start_time_ns
            self.end_time_ns = event.end_time_ns
            self.duration_time_ns = event.duration_time_ns
            self.children = []
            self.extra_fields = event.extra_fields
            self.parent = parent
            self.refer_event = event
        else:
            self.tag = None
            self.id = None
            self.correlation_id = None
            self.start_tid = None
            self.start_time_ns = None
            self.end_time_ns = None
            self.duration_time_ns = None
            self.children = []
            self.extra_fields = None
            self.parent = parent
            self.refer_event = None
    def add_child(self, child):
        self.children.append(child)