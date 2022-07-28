class ProfileEventSlim:
    """
    A simplified version of event class. 
    Attributes:
        duration_us: duration of the event in microseconds(us)
        start_us: start time of the event in microseconds(us)
        end_us: end time of the event in microseconds(us)
        include_events: a list of raw events that have overlaps
    """
    def __init__(self, event):
        self.duration_us = event.duration_us()
        self.start_us = event.start_us()
        self.end_us = event.duration_us() + event.start_us()
        self.include_events = [event]
