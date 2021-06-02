import networkx as nx
import pandas as pd

from pymnet import *
import random
import matplotlib
import cascade as cas
import statistics
import math
import time
import csv

matplotlib.use('TkAgg')

nodes = 75
layers = 2
intra_thres = 0.2
inter_thres = 0.35

attack_size = 50
attack_point = (0.5, 0.5)

attack_type = "spatial_number"  # choose one of the "normal", "spatial_number", "spatial_range"
support_type = "random_nodes"  # choose one of the "random_nodes", "random_layers"
edge_type = "undirected"  # choose one of the "undirected", "directed"

coords = {}

rgg_supp_nodes = {}
rand_supp_nodes = {}

intra_rgg_edges = []
intra_rand_edges = []
inter_rgg_edges = []
inter_rand_edges = []

intra_edges_num = []
inter_edges_num = []  # [for_edge, back_edge, for_supp_edge, back_supp_edge]


def cal_dist(cur_node, target_node):
    x1, y1 = coords[cur_node]
    if target_node == -1:
        x2, y2 = attack_point
    else:
        x2, y2 = coords[target_node]
    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return d


def make_interlayer_edges(net, cur_layer, layer_names, intra_type, inter_type):
    if (intra_type == 'RGG') and (inter_type == 'RGG'):
        for_supp_edges = 0
        back_supp_edges = 0

        for cur_node in range((cur_layer * nodes), (cur_layer + 1) * nodes):
            supp_node = rgg_supp_nodes[cur_node]
            if supp_node != -1:
                net[supp_node, cur_node, layer_names[supp_node // nodes], layer_names[cur_layer]] = 1
                inter_rgg_edges.append((supp_node, cur_node, layer_names[supp_node // nodes], layer_names[cur_layer]))

                if ((supp_node // nodes) > cur_layer) or (
                        (cur_layer == len(layer_names) - 1) and ((supp_node // nodes) == 0)):
                    back_supp_edges += 1
                else:
                    for_supp_edges += 1

        inter_edges_num.append([for_supp_edges, back_supp_edges])

    elif (intra_type == 'RGG') and (inter_type == 'Random'):
        for_supp_edges = 0
        back_supp_edges = 0

        for cur_node in range((cur_layer * nodes), (cur_layer + 1) * nodes):
            supp_node = rand_supp_nodes[cur_node]
            if supp_node != -1:
                if (supp_node // nodes) > cur_layer:
                    if back_supp_edges < inter_edges_num[cur_layer][1]:
                        net[supp_node, cur_node, layer_names[supp_node // nodes], layer_names[cur_layer]] = 1
                        inter_rand_edges.append(
                            (supp_node, cur_node, layer_names[supp_node // nodes], layer_names[cur_layer]))
                        back_supp_edges += 1
                else:
                    if for_supp_edges < inter_edges_num[cur_layer][0]:
                        net[supp_node, cur_node, layer_names[supp_node // nodes], layer_names[cur_layer]] = 1
                        inter_rand_edges.append(
                            (supp_node, cur_node, layer_names[supp_node // nodes], layer_names[cur_layer]))
                        for_supp_edges += 1

    elif (intra_type == 'Random') and (inter_type == 'RGG'):
        for node_from, node_to, layer_from, layer_to in inter_rgg_edges:
            net[node_from, node_to, layer_from, layer_to] = 1

    elif (intra_type == 'Random') and (inter_type == 'Random'):
        for node_from, node_to, layer_from, layer_to in inter_rand_edges:
            net[node_from, node_to, layer_from, layer_to] = 1

    return net


def make_intralayer_edges(net, cur_layer, cur_layer_name, intra_type, inter_type):
    if (intra_type == 'RGG') and (inter_type == 'RGG'):
        edges = 0
        for cur_node in range(cur_layer * nodes, (cur_layer + 1) * nodes):
            for target_node in range(cur_layer * nodes, (cur_layer + 1) * nodes):
                if cur_node != target_node:
                    d = cal_dist(cur_node, target_node)
                    if d <= intra_thres:
                        net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1
                        intra_rgg_edges.append((cur_node, target_node, cur_layer_name))
                        edges += 1
        intra_edges_num.append(edges / 2)

    elif (intra_type == 'RGG') and (inter_type == 'Random'):
        for cur_node, target_node, cur_layer_name in intra_rgg_edges:
            net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1

    elif (intra_type == 'Random') and (inter_type == 'RGG'):
        prob = intra_edges_num[cur_layer] / (nodes * (nodes - 1))

        temp_intra_edges_num = 0
        if cur_layer == 0:
            cur_nodes = list(range(nodes))
        else:
            cur_nodes = list(range(nodes, nodes * 2))

        G = nx.erdos_renyi_graph(nodes, prob)
        edge_list = list(G.edges)
        for edge in edge_list:
            if cur_layer == 0:
                net[edge[0], edge[1], cur_layer_name, cur_layer_name] = 1
                intra_rand_edges.append((edge[0], edge[1], cur_layer_name))
            else:
                net[(edge[0] + nodes), (edge[1] + nodes), cur_layer_name, cur_layer_name] = 1
                intra_rand_edges.append(((edge[0] + nodes), (edge[1] + nodes), cur_layer_name))
            temp_intra_edges_num += 1

        if len(edge_list) < intra_edges_num[cur_layer]:
            diff = intra_edges_num[cur_layer] - len(edge_list)
            while diff > 0:
                cur_node = random.choice(cur_nodes)
                target_node = random.choice(cur_nodes)

                if (cur_node != target_node) and net[cur_node, target_node, cur_layer_name, cur_layer_name] == 0:
                    net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1
                    intra_rand_edges.append((cur_node, target_node, cur_layer_name))
                    temp_intra_edges_num += 1
                    diff -= 1
        elif len(edge_list) > intra_edges_num[cur_layer]:
            diff = len(edge_list) - intra_edges_num[cur_layer]
            while diff > 0:
                cur_node = random.choice(cur_nodes)
                target_node = random.choice(cur_nodes)

                if (cur_node != target_node) and net[cur_node, target_node, cur_layer_name, cur_layer_name] == 1:
                    net[cur_node, target_node, cur_layer_name, cur_layer_name] = 0
                    intra_rand_edges.remove((cur_node, target_node, cur_layer_name))
                    temp_intra_edges_num -= 1
                    diff -= 1

        intra_edges_num.append(temp_intra_edges_num)

        del G

    elif (intra_type == 'Random') and (inter_type == 'Random'):
        for cur_node, target_node, cur_layer_name in intra_rand_edges:
            net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1

    return net


def make_edges(net, layer_names, intra_type, inter_type):
    for cur_layer in range(layers):
        net = make_intralayer_edges(net, cur_layer, layer_names[cur_layer], intra_type, inter_type)
        net = make_interlayer_edges(net, cur_layer, layer_names, intra_type, inter_type)

    return net


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
                target_num[target_node - nodes] += 1

        supp_pairs.append([cur_node, cur_node_pair])

    return supp_pairs, target_num


def init_supp_nodes(intra_type, inter_type):
    if (intra_type == 'RGG') and (inter_type == 'RGG'):
        for i in range(nodes*layers):
            rgg_supp_nodes[i] = -1
    elif (intra_type == 'RGG') and (inter_type == 'Random'):
        for i in range(nodes*layers):
            rand_supp_nodes[i] = -1


def find_supporting_nodes(layer_names, intra_type, inter_type):
    init_supp_nodes(intra_type, inter_type)
    if (intra_type == 'RGG') and (inter_type == 'RGG'):
        cur_nodes = list(range(nodes))
        target_nodes = list(range(nodes, layers*nodes))

        supp_pairs, target_num = find_supp_pair(cur_nodes, target_nodes)
        s_supp_pairs = sorted(supp_pairs, key=lambda l: len(l[1]))

        for _ in cur_nodes:
            if len(s_supp_pairs[0][1]) != 0:
                target, target_num = find_target(s_supp_pairs[0][1], target_num)

                rgg_supp_nodes[s_supp_pairs[0][0]] = target
                rgg_supp_nodes[target] = s_supp_pairs[0][0]

                index = 0
                for j in range(len(s_supp_pairs)):
                    if target in s_supp_pairs[index][1]:
                        s_supp_pairs[index][1].remove(target)
                    index += 1
            else:
                rgg_supp_nodes[s_supp_pairs[0][0]] = -1

            del s_supp_pairs[0]
            s_supp_pairs = sorted(s_supp_pairs, key=lambda l: len(l[1]))

        del supp_pairs
        del s_supp_pairs

    elif (intra_type == 'RGG') and (inter_type == 'Random'):
        cur_nodes = list(range(nodes))
        target_nodes = list(range(nodes, layers * nodes))

        random.shuffle(target_nodes)

        index = 0
        for cur_node in cur_nodes:
            rand_supp_nodes[cur_node] = target_nodes[index]
            rand_supp_nodes[target_nodes[index]] = cur_node

            index += 1


def make_nodes(net, layer_names, intra_type, inter_type):
    for i in range(layers):
        for j in range(nodes):
            if (intra_type == 'RGG') and (inter_type == 'RGG'):
                coords[(i * nodes) + j] = (random.random(), random.random())
            net.add_node((i * nodes) + j, layer_names[i])

    return net


def make_network_layer(net, layer_names):
    for i in range(layers):
        layer_name = chr(97 + i)
        net.add_layer(layer_name, aspect=0)
        layer_names.append(layer_name)

    return net, layer_names


def build_network(rep, intra_type, inter_type):
    layer_names = []
    net = MultilayerNetwork(aspects=1, fullyInterconnected=False, directed=False)

    net, layer_names = make_network_layer(net, layer_names)
    net = make_nodes(net, layer_names, intra_type, inter_type)
    find_supporting_nodes(layer_names, intra_type, inter_type)
    net = make_edges(net, layer_names, intra_type, inter_type)

    return net


def draw_network(net, type):
    fig = draw(net, layerLabelRule={}, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule': 'scaled', 'scalecoeff': 0.01},
               defaultEdgeWidth=0.5, show=True)
    fig.savefig("%s.pdf" % type)


if __name__ == "__main__":
    rep = 1
    rgg_rgg_net = build_network(rep, intra_type='RGG', inter_type='RGG')
    #draw_network(rgg_rgg_net, type="intra_RGG, inter_RGG")

    rgg_rand_net = build_network(rep, intra_type='RGG', inter_type='Random')
    #draw_network(rgg_rand_net, type="intra_RGG, inter_Rand")

    rand_rgg_net = build_network(rep, intra_type='Random', inter_type='RGG')
    #draw_network(rand_rgg_net, type="intra_Rand, inter_RGG")

    rand_rand_net = build_network(rep, intra_type='Random', inter_type='Random')
    draw_network(rand_rand_net, type="intra_Rand, inter_Rand")

