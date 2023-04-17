import glob
import os
from profile_event import ProfileEventSlim


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

def check_event_mem_related(event):
    category = event.get('cat', '').lower()
    # old trace format use 'memcpy and memset' as category
    if category in ['gpu_memcpy', 'gpu_memset', 'memcpy', 'memset']:
        return True
    else:
        return False