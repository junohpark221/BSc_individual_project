import networkx as nx
from pymnet import *
import random
import matplotlib
import math
import cascade as cas
matplotlib.use('TkAgg')

coords = {}

def draw_network(net):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=True)
    fig.savefig("net4.pdf")


if __name__ == "__main__":
    # net = MultilayerNetwork(aspects=1, fullyInterconnected=False)
    """
    for i in range(5):
        net.add_node(i, 'a')

    for i in range(5):
        net.add_node(i + 5, 'b')

    for i in range(5):
        net.add_node(i + 10, 'c')

    for i in range(14):
        coords[i] = (0.05*i, 0.1 + 0.05*i)

    draw_network(net)
    """

    """
    list = ['a', 'b', 'c', 'd']
    a = list.index('b')
    print(a)
    """
    dicts = {'b': [6, 7, 8], 'd': [15, 16, 17, 18, 19], 'c': [9, 10, 11, 12, 13, 14], 'a': [1, 2, 3, 4, 5], 'e': [20, 21, 22, 23]}
    lists = [['e', 20, 21, 22, 23], ['b', 6, 7, 8], ['a', 1, 2, 3, 4, 5], ['d', 15, 16, 17, 18, 19], ['c', 9, 10, 11, 12, 13, 14]]
    new_list = []

    dicts_key = list(dicts.keys())

    sdicts = dict(sorted(dicts.items()))
    sdicts_key = sorted(dicts_key)

    print(sdicts)
    print(sdicts_key)

    """
    i = 0
    while i < 10:
        for j in range(100):
            if i >= 10:
                break
            for m in range(len(list[j]) - 1):
                if i >= 10:
                    break
                new_list.append(list[j][m+1])
                print(list[j][m+1])
                i += 1
    """
