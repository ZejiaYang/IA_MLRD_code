import os
from typing import Dict, Set
from collections import deque, defaultdict
from tick10 import load_graph

def get_node_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    """
    Use Brandes' algorithm to calculate the betweenness centrality for each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to that node's betweenness centrality
    """
    vertices = list(graph.keys())
    d = defaultdict(int)
    for v in vertices:
        queue = deque([v]) #BFS
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
        # print(count)
        # print(pre)
        depen = defaultdict(int)
        while stack:
            c = stack.pop()
            for w in pre[c]:
                depen[w] += (1 + depen[c]) * count[w] / count[c]
            if v != c: 
                d[c] += depen[c]/2
    return d

def get_edge_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    """
    Use Brandes' algorithm to calculate the betweenness centrality for each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to that node's betweenness centrality
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
        print(count)
        print(pre)
        depen = defaultdict(int)
        while stack:
            c = stack.pop()
            for w in pre[c]:
                depen[w] += (1 + depen[c]) * count[w] / count[c]
            if v != c: 
                d[c] += depen[c]/2
        print(d)
    return d


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'supo.edges'))
    betweenness = get_node_betweenness(graph)
    edge_betweenness = get_edge_betweenness(graph)
    print(f"Node betweenness values: {betweenness}")
    print(f"Edge betweenness values: {edge_betweenness}")

if __name__ == '__main__':
    main()
