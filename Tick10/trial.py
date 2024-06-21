import networkx as nx

G = nx.Graph()
G.add_edges_from([(1,2),(2,3),(2,4),(3,4), (1, 4), (1, 5)])

print(nx.edge_betweenness_centrality(G, normalized=False))
print(nx.betweenness_centrality(G, normalized=False))