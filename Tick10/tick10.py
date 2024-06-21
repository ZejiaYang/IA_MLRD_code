import os
from typing import Dict, Set
from collections import defaultdict, deque

def load_graph(filename: str) -> Dict[int, Set[int]]:
    """
    Load the graph file. Each line in the file corresponds to an edge; the first column is the source node and the
    second column is the target. As the graph is undirected, your program should add the source as a neighbour of the
    target as well as the target a neighbour of the source.

    @param filename: The path to the network specification
    @return: a dictionary mapping each node (represented by an integer ID) to a set containing all the nodes it is
        connected to (also represented by integer IDs)
    """
    nodes = defaultdict(set)
    with open(filename, 'r') as f:
        for edge in f.readlines():
            u = int(edge.split()[0])
            v = int(edge.split()[1])
            nodes[u].add(v)
            nodes[v].add(u)
    return nodes

def get_node_degrees(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Find the number of neighbours of each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to the degree of the node
    """
    return {node:len(size) for node, size in graph.items()}


def get_diameter(graph: Dict[int, Set[int]]) -> int:
    """
    Find the longest shortest path between any two nodes in the network using a breadth-first search.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the length of the longest shortest path between any pair of nodes in the graph
    """
    nodes = list(graph.keys())  
    diameter = 0
    for v in nodes:
        visited = set([v])
        distance = 0
        toexplored = deque([v])
        while toexplored:
            cur = len(toexplored)
            for _ in range(cur):
                w = toexplored.popleft()
                for n in graph[w]:
                    if n in visited:
                        continue
                    visited.add(n)
                    toexplored.append(n)
            if toexplored:
                distance += 1
        diameter = max(diameter, distance)
    return diameter
                    


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    degrees = get_node_degrees(graph)
    print(f"Node degrees: {degrees}")

    diameter = get_diameter(graph)
    print(f"Diameter: {diameter}")


if __name__ == '__main__':
    main()
