import networkx as nx
from pymnet import *
import random
import matplotlib
import cascade as cas
import statistics
import math
import time
import csv

matplotlib.use('TkAgg')

nodes = 10
layers = 2
t = 0.2  # 0.002 -> 1, 0.00472 -> 2.36, 0.005 -> 2.5

attack_size = 100  # p = 0.8

# mean degree = (N-1)t = 499 * t
# p<k> = 0.8 * 499 * t = 399.2 * t
#   p<k>        t
#   2.2         0.005511...
#   2.35        0.00588677...
#   2.5         0.00626252...
#   2.67        0.006711...

supp_nodes = {}

intra_edges_num = []

coords = {}


def make_interlayer_edges(net, cur_layer, layer_names):
    if cur_layer == 0:
        for cur_node in range(nodes):
            net[cur_node, supp_nodes[cur_node], layer_names[cur_layer], layer_names[(cur_layer + 1) % 2]] = 1

    return net


def make_sf_intra_edges(net, cur_layer, cur_layer_name):
    return net


def make_er_intra_edges(net, cur_layer, cur_layer_name):
    temp_intra_edges_num = 0
    if cur_layer == 0:
        cur_nodes = list(range(nodes))
    else:
        cur_nodes = list(range(nodes, nodes * 2))

    G = nx.erdos_renyi_graph(nodes, t)
    edge_list = list(G.edges)
    for edge in edge_list:
        if cur_layer == 0:
            net[edge[0], edge[1], cur_layer_name, cur_layer_name] = 1
        else:
            net[(edge[0] + nodes), (edge[1] + nodes), cur_layer_name, cur_layer_name] = 1
        temp_intra_edges_num += 1

    if len(edge_list) < round(nodes * (nodes - 1) * t / 2):
        diff = round(nodes * (nodes - 1) * t / 2) - len(edge_list)
        while diff > 0:
            cur_node = random.choice(cur_nodes)
            target_node = random.choice(cur_nodes)

            if (cur_node != target_node) and net[cur_node, target_node, cur_layer_name, cur_layer_name] == 0:
                net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1
                temp_intra_edges_num += 1
                diff -= 1
    elif len(edge_list) > round(nodes * (nodes - 1) * t / 2):
        diff = len(edge_list) - round(nodes * (nodes - 1) * t / 2)
        while diff > 0:
            cur_node = random.choice(cur_nodes)
            target_node = random.choice(cur_nodes)

            if (cur_node != target_node) and net[cur_node, target_node, cur_layer_name, cur_layer_name] == 1:
                net[cur_node, target_node, cur_layer_name, cur_layer_name] = 0
                temp_intra_edges_num -= 1
                diff -= 1

    intra_edges_num.append(temp_intra_edges_num)

    return net


def make_intralayer_edges(net, cur_layer, cur_layer_name, net_type):
    if net_type == 'er':
        net = make_er_intra_edges(net, cur_layer, cur_layer_name)
    elif net_type == 'sf_3':
        net = make_sf_intra_edges(net, cur_layer, cur_layer_name)

    return net


def make_edges(net, layer_names, net_type):
    for cur_layer in range(layers):
        net = make_intralayer_edges(net, cur_layer, layer_names[cur_layer], net_type)
        net = make_interlayer_edges(net, cur_layer, layer_names)

    return net


def find_supporting_nodes():
    for cur_node in range(nodes * layers):
        supp_nodes[cur_node] = ((cur_node + nodes) % 1000)


def make_nodes(net, layer_names):
    for i in range(layers):
        for j in range(nodes):
            coords[(i*nodes) + j] = (random.random(), random.random())
            net.add_node((i * nodes) + j, layer_names[i])

    return net


def make_network_layer(net, layer_names):
    for i in range(layers):
        layer_name = chr(97 + i)
        net.add_layer(layer_name, aspect=0)
        layer_names.append(layer_name)

    return net, layer_names


def build_network(net_type):
    layer_names = []
    net = MultilayerNetwork(aspects=1, fullyInterconnected=False, directed=False)

    net, layer_names = make_network_layer(net, layer_names)
    net = make_nodes(net, layer_names)
    net = make_edges(net, layer_names, net_type)

    return net


def draw_network(net, type):
    fig = draw(net, layerLabelRule={}, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule': 'scaled', 'scalecoeff': 0.05},
               defaultEdgeWidth=0.5, show=True)
    fig.savefig("%s Network.pdf" % type)


if __name__ == "__main__":

    start = time.time()
    print("Start")

    find_supporting_nodes()

    # Current number of repeat: 0
    rep = 1306

    # init_intra_edge, init_supp_edge, init_clust, init_mean_deg, init_large_comp
    init_data = []
    # fin_intra_edge, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_clust, fin_mean_deg, fin_larg_comp, deg_assort, step.....
    cas_data = []

    er_net = build_network('er')
    draw_network(er_net, type="er_new")

    print("time: ", time.time() - start)
    print("End")
