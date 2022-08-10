from profile_event import ProfileEventSlim
from typing import List
def merge_interval(intervals) -> 'list[ProfileEventSlim]':
    """
    Merge intervals that are overlapped.
    Args:
        intervals: a list of intervals
    Returns:
        a list of merged intervals
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x.start_us)
    res = []
    res.append(intervals[0])
    for i in range(1, len(intervals)):
        last = res[-1]
        if intervals[i].start_us <= last.end_us:
            last.end_us = max(last.end_us, intervals[i].end_us)
            last.include_events.extend(intervals[i].include_events)
        else:
            res.append(intervals[i])
    return res
