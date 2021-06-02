import networkx as nx
from pymnet import *
import random
import matplotlib
import numpy as np
import math
import cascade as cas
matplotlib.use('TkAgg')
"""
coords = {}

def draw_network(net):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=True)
    fig.savefig("net4.pdf")


if __name__ == "__main__":
    # net = MultilayerNetwork(aspects=1, fullyInterconnected=False)

    for i in range(5):
        net.add_node(i, 'a')

    for i in range(5):
        net.add_node(i + 5, 'b')

    for i in range(5):
        net.add_node(i + 10, 'c')

    for i in range(14):
        coords[i] = (0.05*i, 0.1 + 0.05*i)

    draw_network(net)



    list = ['a', 'b', 'c', 'd']
    a = list.index('b')
    print(a)

    dicts = {'b': [6, 7, 8], 'd': [15, 16, 17, 18, 19], 'c': [9, 10, 11, 12, 13, 14], 'a': [1, 2, 3, 4, 5], 'e': [20, 21, 22, 23]}
    lists = [['e', 20, 21, 22, 23], ['b', 6, 7, 8], ['a', 1, 2, 3, 4, 5], ['d', 15, 16, 17, 18, 19], ['c', 9, 10, 11, 12, 13, 14]]
    new_list = []

    dicts_key = list(dicts.keys())

    sdicts = dict(sorted(dicts.items()))
    sdicts_key = sorted(dicts_key)

    print(sdicts)
    print(sdicts_key)


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
m = 3
n = 3
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

max = 0
x_ind = 0
y_ind = 0
for i in range(m-1):
    for j in range(n-1):
        cur_max = A[i][j] + A[i][j+1] + A[i+1][j] + A[i+1][j+1]
        if(max <= cur_max):
            max = cur_max
            x_ind = i
            y_ind = j
print("[%d] [%d]" % (A[x_ind][y_ind], A[x_ind][y_ind+1]))
print("[%d] [%d]" % (A[x_ind+1][y_ind], A[x_ind+1][y_ind+1]))


