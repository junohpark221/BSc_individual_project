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
layers = 3
intra_thres = 0.2
inter_thres = 0.2

attack_size = 10
attack_point = (0.5, 0.5)

attack_type = "spatial_number"          # choose one of the "normal", "spatial_number", "spatial_range"
support_type = "random_layers"          # choose one of the "random_nodes", "random_layers"
edge_type = "undirected"                # choose one of the "undirected", "directed"

coords = {}

dist_array = []
visited = [False for _ in range(nodes)]
in_thres = [False for _ in range(nodes)]

rgg_supp_nodes = {}
rand_supp_nodes = {}

intra_rgg_edges = []
intra_rand_edges = []
inter_rgg_edges = []
inter_rand_edges = []

intra_edges_num = []
inter_edges_num = []                    # [for_edge, back_edge, for_supp_edge, back_supp_edge]


def cal_dist(cur_node, target_node):
    x1, y1 = coords[cur_node]
    if target_node == -1:
        x2, y2 = attack_point
    else:
        x2, y2 = coords[target_node]
    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    return d


def find_nearest_node(cur_node, supporting_node, neighbours, target_layers):
    candidates = []

    for target_node in neighbours:
        if target_node[1] in target_layers:
            dist = cal_dist(cur_node, target_node[0])
            candidates.append((target_node[0], dist))

    if len(candidates) != 0:
        s_candidates = sorted(candidates, key=lambda dist: dist[1])
        supporting_node = s_candidates[0][0]

    return supporting_node


def make_interlayer_edges(net, cur_layer, layer_names, intra_type, inter_type):
    if (intra_type == 'RGG') and (inter_type == 'RGG'):
        if cur_layer != (len(layer_names) - 1):
            for_edges = 0
            back_edges = 0
            for_supp_edges = 0
            back_supp_edges = 0

            for cur_node in range((cur_layer * nodes), (cur_layer + 1) * nodes):
                for target_node in range((cur_layer + 1) * nodes, (cur_layer + 2) * nodes):
                    d = cal_dist(cur_node, target_node)

                    if d <= inter_thres:
                        net[cur_node, target_node, layer_names[cur_layer], layer_names[cur_layer + 1]] = 1
                        net[target_node, cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]] = 1

                        inter_rgg_edges.append((cur_node, target_node, layer_names[cur_layer], layer_names[cur_layer + 1]))
                        inter_rgg_edges.append((target_node, cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]))

                        if cur_node == rgg_supp_nodes[target_node]:
                            for_supp_edges += 1
                        else:
                            for_edges += 1

                        if target_node == rgg_supp_nodes[cur_node]:
                            back_supp_edges += 1
                        else:
                            back_edges += 1

            inter_edges_num.append([for_edges, back_edges, for_supp_edges, back_supp_edges])

    elif (intra_type == 'RGG') and (inter_type == 'Random'):
        if cur_layer != (len(layer_names) - 1):
            for_edges = 0
            back_edges = 0
            for_supp_edges = 0
            back_supp_edges = 0

            cur_nodes = list(range((cur_layer * nodes), (cur_layer + 1) * nodes))
            target_nodes = list(range(((cur_layer + 1) * nodes), ((cur_layer + 2) * nodes)))

            random.shuffle(target_nodes)
            for target_node in target_nodes:
                if rand_supp_nodes[target_node] in cur_nodes:
                    net[rand_supp_nodes[target_node], target_node, layer_names[cur_layer], layer_names[cur_layer + 1]] = 1
                    inter_rand_edges.append((rand_supp_nodes[target_node], target_node, layer_names[cur_layer], layer_names[cur_layer + 1]))
                    for_supp_edges += 1
                if for_supp_edges >= inter_edges_num[cur_layer][2]:
                    break

            random.shuffle(cur_nodes)
            for cur_node in cur_nodes:
                if rand_supp_nodes[cur_node] in target_nodes:
                    net[rand_supp_nodes[cur_node], cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]] = 1
                    inter_rand_edges.append((rand_supp_nodes[cur_node], cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]))
                    back_supp_edges += 1
                if back_supp_edges >= inter_edges_num[cur_layer][3]:
                    break

            sorted(cur_nodes)
            sorted(target_nodes)
            while for_edges < inter_edges_num[cur_layer][0]:
                cur_node = random.choice(cur_nodes)
                target_node = random.choice(target_nodes)
                if net[cur_node, target_node, layer_names[cur_layer], layer_names[cur_layer + 1]] == 0:
                    net[cur_node, target_node, layer_names[cur_layer], layer_names[cur_layer + 1]] = 1
                    inter_rand_edges.append((cur_node, target_node, layer_names[cur_layer], layer_names[cur_layer + 1]))
                    for_edges += 1

            while back_edges < inter_edges_num[cur_layer][1]:
                cur_node = random.choice(cur_nodes)
                target_node = random.choice(target_nodes)
                if net[target_node, cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]] == 0:
                    net[target_node, cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]] = 1
                    inter_rand_edges.append((target_node, cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]))
                    back_edges += 1

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
        intra_edges_num.append(edges)

    elif (intra_type == 'RGG') and (inter_type == 'Random'):
        for cur_node, target_node, cur_layer_name in intra_rgg_edges:
            net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1

    elif (intra_type == 'Random') and (inter_type == 'RGG'):
        cur_nodes = list(range((cur_layer * nodes), ((cur_layer + 1) * nodes)))
        target_nodes = list(range((cur_layer * nodes), ((cur_layer + 1) * nodes)))

        edges = 0
        while edges < intra_edges_num[cur_layer]:
            cur_node = random.choice(cur_nodes)
            target_node = random.choice(target_nodes)
            if net[cur_node, target_node, cur_layer_name, cur_layer_name] == 0:
                net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1
                intra_rand_edges.append((cur_node, target_node, cur_layer_name))
                edges += 1

    elif (intra_type == 'Random') and (inter_type == 'Random'):
        for cur_node, target_node, cur_layer_name in intra_rand_edges:
            net[cur_node, target_node, cur_layer_name, cur_layer_name] = 1

    return net


def make_edges(net, layer_names, intra_type, inter_type):
    for cur_layer in range(layers):
        net = make_intralayer_edges(net, cur_layer, layer_names[cur_layer], intra_type, inter_type)
        net = make_interlayer_edges(net, cur_layer, layer_names, intra_type, inter_type)

    return net


def find_mean(index, min_val, sum_val, temp_supp_nodes):
    if sum_val > min_val:
        return min_val, temp_supp_nodes

    if index == nodes:
        if sum_val < min_val:
            min_val = sum_val
            return min_val, temp_supp_nodes

    for col in range(nodes):
        if not visited[col]:
            visited[col] = True
            temp_supp_nodes.append(col)
            sum_val += dist_array[index][col]
            find_mean(index+1, min_val, sum_val, temp_supp_nodes)
            sum_val -= dist_array[index][col]
            temp_supp_nodes.remove(col)
            visited[col] = False


def find_supporting_pair(cur_layer, target_nodes):
    min_val = nodes * math.sqrt(2)
    supp_nodes = []

    for cur_node in range(cur_layer * nodes, (cur_layer + 1) * nodes):
        dist_this_layer = []
        for target_node in target_nodes:
            cur_dist = cal_dist(cur_node, target_node)
            dist_this_layer.append(cur_dist)
        dist_array.append(dist_this_layer)

    sum_val = 0
    temp_supp_nodes = []

    min_val = find_mean(0, min_val, sum_val, temp_supp_nodes)


    return target_nodes


def find_supporting_nodes(layer_names, intra_type, inter_type):
    if (intra_type == 'RGG') and (inter_type == 'RGG'):
        for cur_layer in range(len(layer_names)):
            target_nodes = []
            if cur_layer == 0:
                target_nodes = list(range(((cur_layer + 1) * nodes), ((cur_layer + 2) * nodes)))
            elif cur_layer == len(layer_names) - 1:
                target_nodes = list(range(((cur_layer - 1) * nodes), cur_layer * nodes))
            else:
                if support_type == "random_nodes":
                    target_nodes = list(range(((cur_layer + 1) * nodes), ((cur_layer + 2) * nodes)))
                elif support_type == "random_layers":
                    choice = random.choice([(cur_layer - 1), (cur_layer + 1)])
                    target_nodes = list(range((choice * nodes), ((choice + 1) * nodes)))

            supp_nodes = find_supporting_pair(cur_layer, target_nodes)

            for cur_node in range(cur_layer * nodes, (cur_layer + 1) * nodes):
                short_dist = 1
                short_node = -1
                for target_node in target_nodes:
                    cur_dist = cal_dist(cur_node, target_node)
                    if (cur_dist <= inter_thres) and (cur_dist <= short_dist):
                        short_dist = cur_dist
                        short_node = target_node
                rgg_supp_nodes[cur_node] = short_node
                if short_node != -1:
                    target_nodes.remove(short_node)

    elif (intra_type == 'RGG') and (inter_type == 'Random'):
        for cur_layer in range(len(layer_names)):
            target_nodes = []
            if cur_layer == 0:
                target_nodes = list(range(((cur_layer + 1) * nodes), ((cur_layer + 2) * nodes)))
            elif cur_layer == len(layer_names) - 1:
                target_nodes = list(range(((cur_layer - 1) * nodes), cur_layer * nodes))
            else:
                if support_type == "random_nodes":
                    target_nodes = list(range(((cur_layer + 1) * nodes), ((cur_layer + 2) * nodes)))
                elif support_type == "random_layers":
                    if inter_edges_num[cur_layer][3] == 0:
                        choice = cur_layer - 1
                    else:
                        choice = cur_layer + 1
                    target_nodes = list(range((choice * nodes), ((choice + 1) * nodes)))

            random.shuffle(target_nodes)
            cur_layer_nodes = list(range((cur_layer * nodes), ((cur_layer + 1) * nodes)))

            index = 0
            for cur_node in cur_layer_nodes:
                rand_supp_nodes[cur_node] = target_nodes[index]
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


def analyse_initial_network(net, init_data):
    layer_names = net.get_layers()                          # return dictionary
    layer_names = sorted(list(layer_names))

    stats = { "clustering":[],                              # Average clustering coefficient
              "mean degree":[],                             # Mean degree
              "the most far node":[],                       # The most far node from the attack centre
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

        far_dist = 0
        for cur_node in range(cur_layer * nodes, (cur_layer + 1) * nodes):
            d = cal_dist(cur_node, -1)
            if d > far_dist:
                far_dist = d

        stats["clustering"].append(nx.average_clustering(G))
        stats["mean degree"].append(len(edges) * 2 / nodes)
        stats["the most far node"].append(far_dist)
        stats["components"].append(components)
        stats["largest component"].append(max(components, key=len))
        stats["size of largest component"].append(len(max(components, key=len)))

        cur_layer +=1

    inter_edge = []
    supp_edge = []
    for inter_edges in inter_edges_num:
        inter_edge.append(sum(inter_edges))
        supp_edge.append(inter_edges[2] + inter_edges[3])

    init_data.append(statistics.mean(intra_edges_num))
    init_data.append(statistics.mean(inter_edge))
    init_data.append(statistics.mean(supp_edge))
    init_data.append(statistics.mean(stats["the most far node"]))
    init_data.append(statistics.mean(stats["clustering"]))
    init_data.append(statistics.mean(stats["mean degree"]))
    init_data.append(statistics.mean(stats["size of largest component"]))

    return init_data


def draw_network(net, type):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=False)
    fig.savefig("%s Network.pdf" % type)


def make_data_frame(init_data, cas_data, rep, graph_type):
    """
    if graph_type == 'RGG_RGG':
        f = open('rgg_rgg_cas_raw_50.csv', 'a', newline='')
        wr = csv.writer(f)
    elif graph_type == 'RGG_Random':
        f = open('rgg_rand_cas_raw_50.csv', 'a', newline='')
        wr = csv.writer(f)
    elif graph_type == 'Random_RGG':
        f = open('rand_rgg_cas_raw_50.csv', 'a', newline='')
        wr = csv.writer(f)
    else:
        f = open('rand_rand_cas_raw_50.csv', 'a', newline='')
        wr = csv.writer(f)

    """
    f = open('find attack size_10.csv', 'a', newline='')
    wr = csv.writer(f)


    # init_intra_edge, init_inter_edge, init_supp_edge, init_far_node, init_clust, init_mean_deg, init_large_comp
    # fin_intra_edge, fin_inter_edge, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_far_node, fin_clust, fin_mean_deg, fin_larg_comp, deg_assort, dist_deg_cent, dist_bet_cent, step.....
    data = [rep, init_data[0], init_data[1], init_data[2], cas_data[0], cas_data[1], cas_data[2], cas_data[3], cas_data[4], cas_data[5], cas_data[6], init_data[3], cas_data[7], init_data[4], cas_data[8], init_data[5], cas_data[9], init_data[6], cas_data[10], cas_data[11], cas_data[12], cas_data[13]]
    for index in range(len(cas_data[14])):
        data.append(cas_data[14][index])
        data.append(cas_data[15][index])

    wr.writerow(data)
    f.close()


if __name__ == "__main__":

    """
        Types of attacks/cascades:
            1. normal attack: select nodes that will be initially attacked randomly.
            2. spatial_number attack: select the nearest (attack_number) nodes from the attack_point, and they will be initially attacked.
            3. spatial_range attack: nodes in the circle (centre: attack_point, radius: attack_radius) will be attacked initially.

        For "normal" attack,                cas.attack_network(network, coords, supporting_nodes, attack_type, attack_size=20)
        For "spatial_number" attack,        cas.attack_network(network, coords, supporting_nodes, attack_type, attack_size=20, attack_layer='a', attack_point=(0.5, 0.5))
        For "spatial_range" attack,         cas.attack_network(network, coords, supporting_nodes, attack_type, attack_layer='a', attack_point=(0.5, 0.5), attack_radius=0.1)

        attack_size = 20                        # number of nodes that will be initially killed
        attack_layer = 'a'                      # the target layer of the attack.
                                                  'a', 'b', 'c'... means the specific layer. 0 means that suppose every nodes are in the same layer.
        attack_point = (0.5, 0.5)               # attack point for spatial_number and spatial_range attacks
        attack_radius = 0.1                     # the radius of attack in spatial_range attacks
    """

    start = time.time()
    print("Start")
    # Current number of repeat: 0
    rep = 1

    for i in range(20):
        # init_intra_edge, init_inter_edge, init_supp_edge, init_far_node, init_clust, init_mean_deg, init_large_comp,
        init_data = []
        # fin_intra_edge, fin_inter_edge, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_far_node, fin_clust, fin_mean_deg, fin_larg_comp, deg_assort, dist_deg_cent, dist_bet_cent, step.....
        cas_data = []

        rgg_rgg_net = build_network(rep, intra_type='RGG', inter_type='RGG')
        init_data = analyse_initial_network(rgg_rgg_net, init_data)
        # draw_network(rgg_rgg_net, type="intra_RGG, inter_RGG")
        # att_rgg_rgg_net = cas.attack_network(rgg_rgg_net, coords, rgg_supp_nodes, attack_type, graph_type="RGG_RGG", attack_size=20)
        att_rgg_rgg_net, cas_data = cas.attack_network(rgg_rgg_net, coords, rgg_supp_nodes, cas_data, attack_type, graph_type="RGG_RGG", attack_size=attack_size, attack_point=attack_point)
        # att_rgg_rgg_net = cas.attack_network(rgg_rgg_net, coords, rgg_supp_nodes, attack_type, graph_type="RGG_RGG", attack_point=(0.5, 0.5), attack_radius=0.1)
        make_data_frame(init_data, cas_data, rep, graph_type='RGG_RGG')

        init_data = []
        cas_data = []
        rgg_rand_net = build_network(rep, intra_type='RGG', inter_type='Random')
        init_data = analyse_initial_network(rgg_rand_net, init_data)
        # draw_network(rgg_rand_net, type="intra_RGG, inter_Random")
        # att_rgg_rand_net = cas.attack_network(rgg_rand_net, coords, rand_supp_nodes, attack_type, graph_type="RGG_Rand", attack_size=20)
        att_rgg_rand_net, cas_data = cas.attack_network(rgg_rand_net, coords, rand_supp_nodes, cas_data, attack_type, graph_type="RGG_Rand", attack_size=attack_size, attack_point=attack_point)
        # att_rgg_rand_net = cas.attack_network(rgg_rand_net, coords, rand_supp_nodes, attack_type, graph_type="RGG_Rand", attack_point=(0.5, 0.5), attack_radius=0.1)
        make_data_frame(init_data, cas_data, rep, graph_type='RGG_Random')

        init_data = []
        cas_data = []
        rand_rgg_net = build_network(rep, intra_type='Random', inter_type='RGG')
        init_data = analyse_initial_network(rand_rgg_net, init_data)
        # draw_network(rand_rgg_net, type="intra_Random, inter_RGG")
        # att_rand_rgg_net = cas.attack_network(rand_rgg_net, coords, rgg_supp_nodes, attack_type, graph_type="Rand_RGG", attack_size=20)
        att_rand_rgg_net, cas_data = cas.attack_network(rand_rgg_net, coords, rgg_supp_nodes, cas_data, attack_type, graph_type="Rand_RGG", attack_size=attack_size, attack_point=attack_point)
        # att_rand_rgg_net = cas.attack_network(rand_rgg_net, coords, rgg_supp_nodes, attack_type, graph_type="Rand_RGG", attack_point=(0.5, 0.5), attack_radius=0.1)
        make_data_frame(init_data, cas_data, rep, graph_type='Random_RGG')

        init_data = []
        cas_data = []
        rand_rand_net = build_network(rep, intra_type='Random', inter_type='Random')
        init_data = analyse_initial_network(rand_rand_net, init_data)
        # draw_network(rand_rand_net, type="intra_Random, inter_Random")
        # att_rand_rand_net = cas.attack_network(rand_rand_net, coords, rand_supp_nodes, attack_type, graph_type="Rand_Rand", attack_size=20)
        att_rand_rand_net, cas_data = cas.attack_network(rand_rand_net, coords, rand_supp_nodes, cas_data, attack_type, graph_type="Rand_Rand", attack_size=attack_size, attack_point=attack_point)
        # att_rand_rand_net = cas.attack_network(rand_rand_net, coords, rand_supp_nodes, attack_type, graph_type="Rand_Rand", attack_point=(0.5, 0.5), attack_radius=0.1)
        make_data_frame(init_data, cas_data, rep, graph_type='Random_Random')

        print("Repeat %d is done" % rep)

        del intra_rgg_edges[:]
        del intra_rand_edges[:]
        del inter_rgg_edges[:]
        del inter_rand_edges[:]
        del intra_edges_num[:]
        del inter_edges_num[:]

        rep += 1

        print("time: ", time.time() - start)

    # draw_network(att_rgg_rgg_net, type="Attacked intra_RGG, inter_RGG")
    # draw_network(att_rgg_rand_net, type="Attacked intra_RGG, inter_Rand")
    # draw_network(att_rand_rgg_net, type="Attacked intra_Rand, inter_RGG")
    # draw_network(att_rand_rand_net, type="Attacked intra_Rand, inter_Rand")

    print("time: ", time.time() - start)
    print("End")
