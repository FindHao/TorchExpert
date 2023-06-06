import os


def check_first_line(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        return first_line == "Model, memcpy, active, busy, total, memcpy ratio, active ratio, busy ratio"


class AnalysisResult:
    """
    A class to store the analysis result.
    app_duration: the duration of the whole application
    gpu_busy_time: the duration when the GPU has at least one active kernel or memcpy
    memcpy_time: the duration for memcpy
    gpu_active_time: the duration when the GPU has at least one active kernel
    """

    def __init__(self, memcpy_time=0, app_duration=0, gpu_busy_time=0, model_name="", output_csv_file=None, log_file=None, start_time_ns=None):
        self.memcpy_time = memcpy_time
        self.app_duration = app_duration
        self.gpu_active_time = gpu_busy_time - memcpy_time
        self.gpu_busy_time = gpu_busy_time
        self.model_name = model_name
        self.output_csv_file = output_csv_file
        # This file comes from the TorchExpert.log_file
        self.log_file = log_file
        # [(idle_event, lca, left_raw_event, left_raw_event_top, right_raw_event, right_raw_event_top),]
        self.idle_event_pairs = []
        self.start_time_ns = start_time_ns

    def print_as_str(self):
        print("\nModel: %s" % self.model_name)
        print("{:<25} {:>10}".format(
            "GPU memcpy time", "%.2fms" % (self.memcpy_time / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU active time", "%.2fms" % (self.gpu_active_time / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU busy time", "%.2fms" % (self.gpu_busy_time / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU total time:", "%.2fms" % (self.app_duration / 1e6)))
        print("{:<25} {:>10}".format("GPU memcpy time ratio:", "%.2f%%" %
                                     (self.memcpy_time * 100 / self.app_duration)))
        print("{:<25} {:>10}".format("GPU active time ratio:", "%.2f%%" %
                                     (self.gpu_active_time * 100 / self.app_duration)))
        print("{:<25} {:>10}".format("GPU busy time ratio:", "%.2f%%" %
                                     (self.gpu_busy_time * 100 / self.app_duration)))

    def print_as_csv(self):
        # print("Model, memcpy, active, busy, total, memcpy ratio, active ratio, busy ratio")
        if self.output_csv_file:
            print("Output csv file: %s" % self.output_csv_file)
            lines = []
            if os.path.isfile(self.output_csv_file):
                with open(self.output_csv_file, 'r') as f:
                    lines = f.readlines()
            table_head = "Model, memcpy, active, busy, total, memcpy ratio, active ratio, busy ratio\n"
            if (len(lines) > 0 and lines[0] != table_head) or len(lines) == 0:
                lines.insert(0, table_head)
            out_str = "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
                self.model_name,
                self.memcpy_time / 1e6,
                self.gpu_active_time / 1e6,
                self.gpu_busy_time / 1e6,
                self.app_duration / 1e6,
                self.memcpy_time * 100 / self.app_duration,
                self.gpu_active_time * 100 / self.app_duration,
                self.gpu_busy_time * 100 / self.app_duration)
            lines.append(out_str)
            with open(self.output_csv_file, 'w') as f:
                f.writelines(lines)
        else:
            print("output_csv_file is not set")

    def print_to_log(self):
        with open(self.log_file, 'w') as f:
            f.write("Model: %s\n" % self.model_name)
            for i, idle_pair in enumerate(self.idle_event_pairs):
                f.write("Idle event pair %d:\n" % i)
                idle_event, lca, left_raw_event, left_raw_event_top, right_raw_event, right_raw_event_top = idle_pair
                f.write("from %.2fms to %.2fms, duration: %.2fms\n" % ((idle_event.start_time_ns - self.start_time_ns) / 1e6,
                        (idle_event.end_time_ns - self.start_time_ns) / 1e6, (idle_event.end_time_ns - idle_event.start_time_ns) / 1e6))
                f.write("LCA: %s\n" % lca.name)
                f.write("Left raw event: %s\n" % left_raw_event.name)
                f.write("Left raw event top: %s\n" % left_raw_event_top.name)
                f.write("Right raw event: %s\n" % right_raw_event.name)
                f.write("Right raw event top: %s\n" % right_raw_event_top.name)
                left_i = lca.children.index(left_raw_event_top)
                right_i = lca.children.index(right_raw_event_top)
                for i in range(left_i+1, right_i):
                    f.write("Idle event %d: %s\n" % (i, lca.children[i].name))

