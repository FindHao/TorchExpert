class ProfileEventSlim:
    """
    A simplified version of event class. 
    Attributes:
        duration_us: duration of the event in microseconds(us)
        start_us: start time of the event in microseconds(us)
        end_us: end time of the event in microseconds(us)
        include_events: a list of raw events that have overlaps
    """
    def __init__(self, event, duration_time_ns=None, start_time_ns=None, end_time_ns=None):
        if event is not None:
            # self.duration_time_ns = event.duration_time_ns
            # self.start_time_ns = event.start_time_ns
            # self.end_time_ns = event.end_time_ns
            self.duration_time_ns = event.duration_us() * 1e3
            self.start_time_ns = event.start_us() * 1e3
            self.end_time_ns = (event.start_us() + event.duration_us()) * 1e3
            self.include_events = [event]
        else:
            self.duration_time_ns = duration_time_ns
            self.start_time_ns = start_time_ns
            self.end_time_ns = end_time_ns
            self.include_events = []

