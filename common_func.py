import glob
import os
from profile_event import ProfileEventSlim
from torch._C._profiler import _ProfilerEvent

def print_all_event_time(intervals):
    # print all event start time and end time
    print("=====================================")
    all_start_time = intervals[0].start_time_ns
    for interval in intervals:
        all_start_time = min(all_start_time, interval.start_time_ns)
    for interval in intervals:
        # print the event's start time - all_start_time and convert it to miliseconds
        print((interval.start_time_ns - all_start_time) / 1e6,
              (interval.end_time_ns - all_start_time) / 1e6)


def merge_interval(intervals:list[ProfileEventSlim]):
    """
    Merge events that have overlapps.
    At the begining, all ProfileEventSlim events in intervals are just simple shadow of the raw events. After merging, the ProfileEventSlim events in intervals will be merged. some ProfileEventSlim events are duplicated, and their include_events will be merged.
    Args:
        intervals: a list of intervals/events
    Returns:
        a list of merged intervals
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x.start_time_ns)
    # print_all_event_time(intervals)
    res = []
    res.append(intervals[0])
    for i in range(1, len(intervals)):
        last = res[-1]
        # if there is overlap, merge the two intervals
        if intervals[i].start_time_ns <= last.end_time_ns:
            last.end_time_ns = max(last.end_time_ns, intervals[i].end_time_ns)
            last.include_events.extend(intervals[i].include_events)
        else:
            res.append(intervals[i])
    # print_all_event_time(res)
    return res




def get_latest_file(path):
    """
    Get the latest file in a directory.
    Args:
        path: the directory path
    Returns:
        the latest file path
    """
    list_of_files = glob.glob(path + '/*')
    list_of_files = [ _ for _ in list_of_files if os.path.isfile(_) and _.endswith(".json")]
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

standard_mem_cat = ['gpu_memcpy', 'gpu_memset', 'memcpy', 'memset']
def check_event_mem_related(event):
    category = event.get('cat', '').lower()
    # old trace format use 'memcpy and memset' as category
    return category in standard_mem_cat

standard_cat = ['gpu_memcpy', 'gpu_memset', 'memcpy', 'memset', 'kernel']
def check_event_in_gpu_trace(event):
    category = event.get('cat', '').lower()
    return category in standard_cat

"""
Args:
    event: the target event
Returns:
    a list of events in a path from the root to the event
"""
def get_all_events_in_path(event:_ProfilerEvent):
    all_events = []
    all_events.append(event)
    tmp_event = event
    while tmp_event.parent:
        tmp_event = tmp_event.parent
        all_events.append(tmp_event)
    return all_events