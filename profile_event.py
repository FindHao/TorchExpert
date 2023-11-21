from torch._C._profiler import _ProfilerEvent

class ProfileEventSlim:
    """
    A simplified version of event class. 

    It is used to either become a shadow of the original event class or merge all the raw events that have overlaps.
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
            elif type(event) is TraceEvent:
                self.duration_time_ns = event.dur
                self.start_time_ns = event.ts
                self.end_time_ns = event.ts + event.dur
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

class IdleEvent:
    """
    A class to store the idle event. An idle event is an event that has no corresponding raw event between the left ProfileEventSlim event and the right ProfileEventSlim event.
    Attributions:
        start_time: start time of the idle event
        end_time: end time of the idle event
        left_event: the ProfileEventSlim event that is on the left of the idle event
        right_event: the ProfileEventSlim event that is on the right of the idle event. 
    """
    def __init__(self, start_time_ns=None, end_time_ns=None, left_event=None, right_event=None):
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.left_event = left_event
        self.right_event = right_event


class TraceEvent:
    """
    A class to store the trace event. A trace event is an event in the original trace file.
    """
    def __init__(self, parent=None, **entries):
        self.parent = parent
        self.children = []
        # ansestor is used to merge the trace events having same ansestor like `aten::add` when we check the stream assignment.
        self.ansestor = None
        # Dynamically set attributes for the instance
        self.__dict__.update(entries)
        if "args" in entries:
            for key, value in entries["args"].items():
                setattr(self, key, value)

class Shadow_ProfilerEvent:
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