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

nodes = 500
layers = 2
t = 0.005823                                # 0.002 -> 1, 0.00472 -> 2.36, 0.005 -> 2.5

attack_size = 100                           # p = 0.8

# mean degree = (N-1)t = 499 * t
# p<k> = 0.8 * 499 * t = 399.2 * t
#   p<k>        t
#   2.2         0.005511...
#   2.35        0.00588677...
#   2.5         0.00626252...
#   2.67        0.006711...
    
supp_nodes = {}

intra_edges_num = []


def make_interlayer_edges(net, cur_layer, layer_names):
    if cur_layer == 0:
        for cur_node in range(nodes):
                net[cur_node, supp_nodes[cur_node], layer_names[cur_layer], layer_names[(cur_layer + 1) % 2]] = 1

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


def make_intralayer_edges(net, cur_layer, cur_layer_name):
    net = make_er_intra_edges(net, cur_layer, cur_layer_name)

    return net


def make_edges(net, layer_names):
    for cur_layer in range(layers):
        net = make_intralayer_edges(net, cur_layer, layer_names[cur_layer])
        net = make_interlayer_edges(net, cur_layer, layer_names)

    return net


def find_supporting_nodes():
    for cur_node in range(nodes * layers):
        supp_nodes[cur_node] = ((cur_node + nodes) % 1000)


def make_nodes(net, layer_names):
    for i in range(layers):
        for j in range(nodes):
            net.add_node((i * nodes) + j, layer_names[i])

    return net


def make_network_layer(net, layer_names):
    for i in range(layers):
        layer_name = chr(97 + i)
        net.add_layer(layer_name, aspect=0)
        layer_names.append(layer_name)

    return net, layer_names


def build_network():
    layer_names = []
    net = MultilayerNetwork(aspects=1, fullyInterconnected=False, directed=False)

    net, layer_names = make_network_layer(net, layer_names)
    net = make_nodes(net, layer_names)
    net = make_edges(net, layer_names)

    return net


def analyse_initial_network(net, init_data):
    layer_names = net.get_layers()                          # return dictionary
    layer_names = sorted(list(layer_names))

    stats = { "mean degree":[],                             # Mean degree
              "components":[],                              # Components of the graph in each layers
              "largest component":[],                       # The largest component of the graphs
              "size of largest component":[],               # The size of the largest component
            }

    # init_intra_edge, init_inter_edge, init_supp_edge, init_far_node, init_clust, init_mean_deg, init_large_comp
    cur_layer = 0
    for layer in layer_names:
        edges = []
        for edge in net.edges:
            if edge[2] == edge[3] == layer:
                edges.append(edge[:2])

        G = nx.Graph()
        G.add_edges_from(edges)

        components = list(nx.connected_components(G))

        stats["mean degree"].append(len(edges) * 2 / nodes)
        stats["components"].append(components)
        stats["largest component"].append(max(components, key=len))
        stats["size of largest component"].append(len(max(components, key=len)))

        cur_layer += 1

    init_data.append(intra_edges_num)
    init_data.append(len(supp_nodes))
    init_data.append(stats["mean degree"][0])
    init_data.append(stats["mean degree"][1])
    init_data.append(stats["size of largest component"][0])
    init_data.append(stats["size of largest component"][1])

    return init_data


def draw_network(net, type):
    fig = draw(net, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=True)
    fig.savefig("%s Network.pdf" % type)


def make_data_frame(init_data, cas_data, rep):
    f = open('er_100_50_0012.csv', 'a', newline='')
    wr = csv.writer(f)

    # init_intra_edge, init_supp_edge, init_mean_deg_a, init_mean_deg_b, init_large_comp_a, init_large_comp_b
    # fin_intra_edge_a, fin_intra_edge_b, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_larg_comp_a, fin_larg_comp_b, step..
    data = [rep, attack_size, t, init_data[0][0], init_data[0][1], init_data[1], cas_data[0], cas_data[1], cas_data[2], init_data[2], init_data[3], cas_data[3], cas_data[4], cas_data[5], cas_data[6], init_data[4], init_data[5], cas_data[7], cas_data[8]]
    for index in range(len(cas_data[9])):
        data.append(cas_data[9][index])
        data.append(cas_data[10][index])

    wr.writerow(data)
    f.close()


if __name__ == "__main__":

    start = time.time()
    print("Start")

    find_supporting_nodes()

    # Current number of repeat: 0
    rep = 1

    for i in range(500):
        # init_intra_edge, init_supp_edge, init_clust, init_mean_deg, init_large_comp
        init_data = []
        # fin_intra_edge, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_clust, fin_mean_deg, fin_larg_comp, deg_assort, step.....
        cas_data = []

        er_net = build_network()
        init_data = analyse_initial_network(er_net, init_data)
        att_er_net, cas_data = cas.attack_network(er_net, nodes*layers, supp_nodes, cas_data, attack_size)
        make_data_frame(init_data, cas_data, rep)

        print("Repeat %d is done" % rep)

        """
        if rep % 50 == 0:
            t += 0.000012
        """

        rep += 1

        print("time: ", time.time() - start)

    # draw_network(att_rgg_rgg_net, type="Attacked intra_RGG, inter_RGG")

    print("time: ", time.time() - start)
    print("End")