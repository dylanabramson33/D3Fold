import networkx as nx
class DAG:
    """
    Class for a directed acyclic graph.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
    
    def add_edge(self, node_1: tuple, node_2: tuple) -> None:
        """
        Add a node to the graph.
        """
        self.graph.add_edge(node_1, node_2)
    
    def compute_dependencies(self, node) -> list:
        """
        Traverse the graph from start_node until node with end label
        """
        return list(nx.descendants(self.graph, node))


        
