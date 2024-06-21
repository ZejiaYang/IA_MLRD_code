import os
from typing import Set, Dict, List, Tuple
from collections import defaultdict, deque
# from tick10 import load_graph

def get_number_of_edges(graph: Dict[int, Set[int]]) -> int:
    """
    Find the number of edges in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the number of edges
    """
    return sum([len(value) for _, value in graph.items()]) / 2
 

def get_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find the number of components in the graph using a DFS.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: list of components for the graph.
    """
    vertices = set()
    components = []

    def dfs(vertex: int):
        component = components[-1]
        component.add(vertex)
        vertices.add(vertex)

        for neighbor in graph[vertex]:
            if neighbor not in vertices:
                dfs(neighbor)
        
    for vertex, _ in graph.items():
        if vertex in vertices:
            continue
        else:
            components.append(set())
            dfs(vertex)

    return components

def get_edge_betweenness(graph: Dict[int, Set[int]]) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: betweenness for each pair of vertices in the graph connected to each other by an edge
    """
    vertices = list(graph.keys())
    d = defaultdict(int)
    for v in vertices:
        queue = deque([v]) #BSF
        stack = deque([]) #record

        dis = {k: float('inf') for k in vertices} # distance 
        dis[v] = 0

        count = defaultdict(int) # number of shortest path
        count[v] = 1

        pre = defaultdict(list)

        while queue:
            c = queue.popleft()
            stack.append(c)

            for w in graph[c]:
                if dis[w] == float('inf'):
                    dis[w] = dis[c] + 1
                    queue.append(w) 
                
                if dis[w] == dis[c] + 1:
                    pre[w].append(c)
                    count[w] += count[c]
        
        depen = defaultdict(int)
        while stack:
            c = stack.pop()
            for w in pre[c]:
                d[(w, c)] += (1 + depen[c]) * count[w] / count[c]
                depen[w] += (1 + depen[c]) * count[w] / count[c]
    return d


def girvan_newman(graph: Dict[int, Set[int]], min_components: int) -> List[Set[int]]:
    """     * Find the number of edges in the graph.
     *
     * @param graph
     *        {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *        loaded graph
     * @return {@link Integer}> Number of edges.
    """
    c = get_components(graph)
    cgraph = graph
    while len(c) < min_components:
        edges_betweenness = get_edge_betweenness(cgraph)
        ans = 0
        remove = []
        visited = set()
        for (v, w), value in list(edges_betweenness.items()):
            if (w, v) in visited:
                continue
            visited.add((v, w))
            if value + edges_betweenness[w, v] >= ans - 10e-6 and value + edges_betweenness[w, v] <= ans + 10e-6:
                remove.append((v, w))
            elif value > ans + 1e-6:
                remove = [(v, w)]
                ans = value
        for (w, v) in remove:
            cgraph[w].remove(v)
            cgraph[v].remove(w)
        c = get_components(cgraph)
    return c


# def main():
#     graph = load_graph(os.path.join('data', 'social_networks', 'facebook_circle.edges'))

#     num_edges = get_number_of_edges(graph)
#     print(f"Number of edges: {num_edges}")

#     components = get_components(graph)
#     print(f"Number of components: {len(components)}")

#     edge_betweenness = get_edge_betweenness(graph)
#     print(f"Edge betweenness: {edge_betweenness}")

#     clusters = girvan_newman(graph, min_components=20)
#     print(f"Girvan-Newman for 20 clusters: {clusters}")


# if __name__ == '__main__':
#     main()