import networkx as nx
from pymnet import *
import random
import matplotlib
import cascade_er as cas
import statistics
import math
import time
import csv
matplotlib.use('TkAgg')

nodes = 500
layers = 2
k = 0.01

attack_size = 250

attack_type = "spatial_number"          # choose one of the "normal", "spatial_number", "spatial_range"
support_type = "random_layers"          # choose one of the "random_nodes", "random_layers"
edge_type = "undirected"                # choose one of the "undirected", "directed"

coords = {}

supp_nodes = {}

intra_edges_num = 0


def make_interlayer_edges(net, cur_layer, layer_names):
    for cur_node in range(nodes):
            net[cur_node, supp_nodes[cur_node], layer_names[0], layer_names[1]] = 1

    return net


def make_intralayer_edges(net, cur_layer, cur_layer_name):
    global intra_edges_num
    if cur_layer == 0:
        G = nx.erdos_renyi_graph(nodes, k)
        edge_list = list(G.edges)
        for edge in edge_list:
            net[edge[0], edge[1], cur_layer_name, cur_layer_name] = 1

        intra_edges_num = len(edge_list)
    else:
        edges = 0
        while edges < intra_edges_num:
            cur_node = random.choice(range(cur_layer*nodes, (cur_layer+1)*nodes))
            target_node = random.choice(range(cur_layer*nodes, (cur_layer+1)*nodes))
            if net[cur_node, target_node, cur_layer_name, cur_layer_name] == 0:
                net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1
                edges += 1

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
            coords[(i * nodes) + j] = (random.random(), random.random())
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
    find_supporting_nodes()
    net = make_edges(net, layer_names)

    return net


def analyse_initial_network(net, init_data):
    layer_names = net.get_layers()                          # return dictionary
    layer_names = sorted(list(layer_names))

    stats = { "clustering":[],                              # Average clustering coefficient
              "mean degree":[],                             # Mean degree
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

        stats["clustering"].append(nx.average_clustering(G))
        stats["mean degree"].append(len(edges) * 2 / nodes)
        stats["components"].append(components)
        stats["largest component"].append(max(components, key=len))
        stats["size of largest component"].append(len(max(components, key=len)))

        cur_layer +=1

    init_data.append(intra_edges_num)
    init_data.append(len(supp_nodes))
    init_data.append(statistics.mean(stats["clustering"]))
    init_data.append(statistics.mean(stats["mean degree"]))
    init_data.append(statistics.mean(stats["size of largest component"]))

    """
    global attack_size
    attack_size = round((1 - (2.5 / init_data[3])) * 500)
    """

    return init_data


def draw_network(net, type):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=True)
    fig.savefig("%s Network.pdf" % type)


def make_data_frame(init_data, cas_data, rep):
    f = open('er_cas_raw.csv', 'a', newline='')
    wr = csv.writer(f)

    # init_intra_edge, init_supp_edge, init_clust, init_mean_deg, init_large_comp
    # fin_intra_edge, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_clust, fin_mean_deg, fin_larg_comp, deg_assort, step.....
    data = [rep, init_data[0], init_data[1], cas_data[0], cas_data[1], cas_data[2], cas_data[3], cas_data[4], cas_data[5], init_data[2], cas_data[6], init_data[3], cas_data[7], init_data[4], cas_data[8], cas_data[9]]
    for index in range(len(cas_data[10])):
        data.append(cas_data[10][index])
        data.append(cas_data[11][index])

    wr.writerow(data)
    f.close()


if __name__ == "__main__":

    start = time.time()
    print("Start")
    # Current number of repeat: 0
    rep = 1

    for i in range(100):
        # init_intra_edge, init_supp_edge, init_clust, init_mean_deg, init_large_comp
        init_data = []
        # fin_intra_edge, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_clust, fin_mean_deg, fin_larg_comp, deg_assort, step.....
        cas_data = []

        er_net = build_network()
        init_data = analyse_initial_network(er_net, init_data)
        # draw_network(er_net, type="ER")
        att_er_net, cas_data = cas.attack_network(er_net, coords, supp_nodes, cas_data, attack_size)
        # draw_network(att_er_net, type="ER")
        make_data_frame(init_data, cas_data, rep)

        print("Repeat %d is done" % rep)

        intra_edges_num = 0

        rep += 1

        print("time: ", time.time() - start)

    # draw_network(att_rgg_rgg_net, type="Attacked intra_RGG, inter_RGG")

    print("time: ", time.time() - start)
    print("End")
