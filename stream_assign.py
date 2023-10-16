
import json
class SSNode:
    # original_json_snippet is one key-value pair in the original json file. (key: node name, value: node info)
    def __init__(self, json_snippet) -> None:
        assert json_snippet is not None
        tmp_value= json_snippet[1]
        self.successors = tmp_value.get("successors", [])
        self.predecessors = tmp_value.get("predecessors", [])
        self.name = json_snippet[0]
        self.stream_id = tmp_value["stream_id"]

class SSGraph:
    def __init__(self, json_snippet) -> None:
        self.ssnodes = []
        self.name_mapping = {}
        self.stream_assignment = {}
        self.build_graph(json_snippet)
        self.analyze_stream()
    
    def build_graph(self, json_snippet):
        all_nodes = {}
        for node_name, node_info in json_snippet.items():
            if node_name != "order":
                all_nodes[node_name] = SSNode((node_name, node_info))
        order = json_snippet["order"]
        for node_name in order:
            self.ssnodes.append(all_nodes[node_name])
            self.name_mapping[node_name] = all_nodes[node_name]
    
    def analyze_stream(self):
        for node in self.ssnodes:
            if self.stream_assignment.get(node.stream_id) is None:
                self.stream_assignment[node.stream_id] = []
            self.stream_assignment[node.stream_id].append(node.name)
        

class AllGraphs:
    def __init__(self, original_json_path):
        self.stream_assignment_json = None
        self.load_stream_assignment(original_json_path)
        self.graphs = {}
        for graph_name in self.stream_assignment_json.keys():
            self.graphs[graph_name] = SSGraph(self.stream_assignment_json[graph_name])

    def load_stream_assignment(self, stream_assignment_json_path):
        print("Analyzing stream assignment json file: {}".format(stream_assignment_json_path))
        with open(stream_assignment_json_path, "r") as f:
            self.stream_assignment_json = json.load(f)
        graph_keys = [_ for _ in self.stream_assignment_json.keys() if _.startswith("graph")]
        # for now, we can only deal with a model with two graphs(forward and backward)
        assert len(graph_keys) == 2
    
    def print_streams(self):
        for graph_name, graph in self.graphs.items():
            print("Graph: {}".format(graph_name))
            for stream_id, node_names in graph.stream_assignment.items():
                print("Stream {}: {}".format(stream_id, node_names))