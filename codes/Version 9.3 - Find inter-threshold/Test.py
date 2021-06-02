import networkx as nx
from pymnet import *
import random
import matplotlib
import numpy as np
import math
import cascade as cas
import pandas as pd
import csv
import time

matplotlib.use('TkAgg')

nodes = 500
layers = 2
inter_thres = 0.55

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


def find_target(targets, target_num):
    num_list = []
    for target in targets:
        if target_num[target-nodes] != 0:
            num_list.append([target, target_num[target-nodes]])

    s_num_list = sorted(num_list, key=lambda l: l[1])
    target_num[s_num_list[0][0]-nodes] -= 1

    return s_num_list[0][0], target_num


def find_supp_pair(cur_nodes, target_nodes):
    supp_pairs = []
    target_num = []

    for _ in target_nodes:
        target_num.append(0)

    for cur_node in cur_nodes:
        cur_node_pair = []

        for target_node in target_nodes:
            cur_dist = cal_dist(cur_node, target_node)
            if cur_dist <= inter_thres:
                cur_node_pair.append(target_node)
                target_num[target_node-nodes] += 1

        supp_pairs.append([cur_node, cur_node_pair])

    return supp_pairs, target_num


def init_supp_nodes():
    for i in range(nodes*2):
        supporting_nodes[i] = -1


def find_supporting_nodes():
    init_supp_nodes()

    cur_nodes = list(range(nodes))
    target_nodes = list(range(nodes, layers*nodes))

    supp_pairs, target_num = find_supp_pair(cur_nodes, target_nodes)
    s_supp_pairs = sorted(supp_pairs, key=lambda l: len(l[1]))

    for _ in cur_nodes:
        if len(s_supp_pairs[0][1]) != 0:
            target, target_num = find_target(s_supp_pairs[0][1], target_num)

            supporting_nodes[s_supp_pairs[0][0]] = target
            supporting_nodes[target] = s_supp_pairs[0][0]

            index = 0
            for j in range(len(s_supp_pairs)):
                if target in s_supp_pairs[index][1]:
                    s_supp_pairs[index][1].remove(target)
                index += 1

        del s_supp_pairs[0]
        s_supp_pairs = sorted(s_supp_pairs, key=lambda l: len(l[1]))

    del supp_pairs
    del s_supp_pairs


def make_nodes():
    for i in range(layers):
        for j in range(nodes):
            coords[(i * nodes) + j] = (random.random(), random.random())


if __name__ == "__main__":
    start = time.time()
    print("Start")

    rep = 1

    data = {}
    cur_data = []

    for _ in range(100):
        make_nodes()
        find_supporting_nodes()

        supp_edge_num = 0
        for node in supporting_nodes:
            if supporting_nodes[node] != -1:
                supp_edge_num += 1

        cur_data.append(supp_edge_num)

        if rep % 50 == 0:
            data[inter_thres] = cur_data.copy()

            inter_thres += 0.05
            del cur_data[:]

        rep += 1

        print("time: ", time.time() - start)

    df = pd.DataFrame(data)
    df.to_csv('find inter thres_test.csv')

"""

def coinCount(coins, m):
    memo = [0 for _ in range(m + 1)]

    for k in range(1, m+1):
        steps = memo[k-1]
        for coin in coins:
            if k-coin >= 0:
                steps = min(steps, memo[k-coin])
        memo[k] = steps + 1

    print(memo)
    return memo[m]


def run_coinCount(coins,m):
    from time import perf_counter
    start = perf_counter()
    answer = coinCount(coins,m)
    finish = perf_counter()
    print("coinCount([",coins[0],", ...],",m,") => ",answer,sep="")
    print(round(finish-start, ndigits=6), "seconds")

if __name__ == "__main__":
    my_coins = [50, 40, 20, 10, 5, 4, 2, 1]

    run_coinCount(my_coins, 80)
    run_coinCount(my_coins, 130)
    run_coinCount(my_coins, 180)
    run_coinCount(my_coins, 230)
    run_coinCount(my_coins, 280)
    run_coinCount(my_coins, 330)
    run_coinCount(my_coins, 380)
    """
