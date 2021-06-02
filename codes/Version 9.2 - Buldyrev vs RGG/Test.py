import networkx as nx
from pymnet import *
import random
import matplotlib
import numpy as np
import math
import cascade as cas
import pandas as pd
import csv

matplotlib.use('TkAgg')

nodes = 500
layers = 2
inter_thres = 0.05

coords = {}
supporting_nodes = {}


def draw_network(net):
    fig = draw(net, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=True)
    fig.savefig("test.pdf")


def cal_dist(cur_node, target_node):
    x1, y1 = coords[cur_node]
    x2, y2 = coords[target_node]
    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    return d


def find_supp_pair(cur_nodes, target_nodes):
    supp_pairs = []
    for cur_node in cur_nodes:
        cur_node_pair = []

        for target_node in target_nodes:
            cur_dist = cal_dist(cur_node, target_node)
            if cur_dist <= inter_thres:
                cur_node_pair.append(target_node)

        supp_pairs.append([cur_node, cur_node_pair])

    return supp_pairs


def find_supporting_nodes():
    cur_nodes = list(range(nodes))
    target_nodes = list(range(nodes, layers*nodes))

    supp_pairs = find_supp_pair(cur_nodes, target_nodes)
    s_supp_pairs = sorted(supp_pairs, key=lambda l: len(l[1]))

    for _ in cur_nodes:
        if len(s_supp_pairs[0][1]) != 0:
            target = random.choice(s_supp_pairs[0][1])

            supporting_nodes[s_supp_pairs[0][0]] = target
            supporting_nodes[target] = s_supp_pairs[0][0]

            index = 0
            for j in range(len(s_supp_pairs)):
                if target in s_supp_pairs[index][1]:
                    s_supp_pairs[index][1].remove(target)
                index += 1
        else:
            supporting_nodes[s_supp_pairs[0][0]] = -1

        del s_supp_pairs[0]
        s_supp_pairs = sorted(s_supp_pairs, key=lambda l: len(l[1]))

    del supp_pairs
    del s_supp_pairs


def make_nodes():
    for i in range(layers):
        for j in range(nodes):
            coords[(i * nodes) + j] = (random.random(), random.random())


if __name__ == "__main__":
    make_nodes()
    find_supporting_nodes()
