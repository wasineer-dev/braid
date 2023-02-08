
import igraph as ig
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from time import time as timer

from inputFile import inputFile
from meanfield import simulateLikelihood as smlt

from spoke_model import countMatrixModel as cmm 
import functools

Nk = 500
inputSet = inputFile.CInputSet("gavin2002.csv", cmm.CountMatrixModel)
nProteins = inputSet.observationG.nProteins
cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk) # default

def make_graph(x):
    mle = cmfa.estimate(inputSet.observationG, nProteins, Nk, x)  
    cmfa.find_argmax()
    g.delete_edges()
    for i in range(nProteins):
        for j in inputSet.observationG.lstAdjacency[i]:
            s = inputSet.observationG.mObserved[i][j]
            if (s == 0):
                continue
            if (i < j):
                g.add_edge(i,j)


g = ig.Graph(nProteins)

def update_graph(frame, psi, ax):
    #mst = g.spanning_tree()
    #components = g.components()
    ax.clear()
    x = psi[frame]
    make_graph(x)
    gclust = g.connected_components(mode='weak')
    gd = gclust.giant()
    community_greedy = gd.community_fastgreedy()
    communities = community_greedy.as_clustering()
    """
    ig.plot(
            communities,
            target=ax,
            mark_groups=True,
            vertex_size=2,
            edge_width=0.5   
        )
    """
    # mst = g.spanning_tree()
    num_communities = len(communities)
    palette1 = ig.RainbowPalette(n=num_communities)
    for i, community in enumerate(communities):
        gd.vs[community]["color"] = i
        community_edges = gd.es.select(_within=community)
        community_edges["color"] = i

    layout = gd.layout("kk")
    gd.vs["x"], gd.vs["y"] = list(zip(*layout))
    gd.vs["size"] = 1
    gd.es["size"] = 1

    cluster_graph = communities.cluster_graph(
        combine_vertices={
            "x": "mean",
            "y": "mean",
            "color": "first",
            "size": "sum",
        },
        combine_edges={
            "size": "sum"
        }
    )

    palette2 = ig.GradientPalette("gainsboro", "black")
    gd.es["color"] = [palette2.get(int(i)) for i in ig.rescale(cluster_graph.es["size"], (0, 255), clamp=True)]
    #mst.es["color"] = [palette2.get(10)]

    ig.plot(
        cluster_graph,
        target=ax,
        palette=palette1,
        # set a minimum size on vertex_size, otherwise vertices are too small
        vertex_size=[max(0.2, size / 100) for size in cluster_graph.vs["size"]],
        edge_color=gd.es["color"],
        edge_width=0.8,
    )

    handles = ax.get_children()[:frame]
    return handles
    

vlist = [8.0, 4.0, 2.0]
num_steps = 3
# Creating the Animation object
fig, ax = plt.subplots(figsize=(5,5))
ani = animation.FuncAnimation(
    fig, update_graph, num_steps, fargs=(vlist, ax), interval=100, blit=True)

plt.show()
    