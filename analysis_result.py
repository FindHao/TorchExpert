class AnalysisResult:
    """
    A class to store the analysis result.
    """

    def __init__(self, memcpy_time=0,app_duration=0, gpu_busy_time=0):
        self.memcpy_time = memcpy_time
        self.app_duration = app_duration
        self.sum_gpu_active = app_duration - memcpy_time
        self.sum_gpu_busy = gpu_busy_time

    
    def print_as_str(self):
        print("{:<25} {:>10}".format(
                "GPU memcpy time", "%.2fms" % (self.memcpy_time / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU active time", "%.2fms" % (self.sum_gpu_active / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU busy time", "%.2fms" % (self.sum_gpu_busy / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU total time:", "%.2fms" % (self.app_duration / 1e6)))
        print("{:<25} {:>10}".format("GPU memcpy time ratio:", "%.2f%%" %
                                    (self.memcpy_time * 100 / self.app_duration)))
        print("{:<25} {:>10}".format("GPU active time ratio:", "%.2f%%" %
                                    (self.sum_gpu_active * 100 / self.app_duration)))
        print("{:<25} {:>10}".format("GPU busy time ratio:", "%.2f%%" %
                                        (self.sum_gpu_busy * 100 / self.app_duration)))


