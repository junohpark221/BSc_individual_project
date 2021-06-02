import networkx as nx
from pymnet import *
import random
import matplotlib
import cascade as cas
import math
import time
matplotlib.use('TkAgg')

nodes = 300
layers = 3
intra_thres = 0.1
inter_thres = 0.05
attack_size = 30

attack_type = "spatial_number"          # choose one of the "normal", "spatial_number", "spatial_range"
support_type = "random_layers"          # choose one of the "random_nodes", "random_layers"
edge_type = "undirected"                # choose one of the "undirected", "directed"

coords = {}

rgg_rgg_supp_nodes = {}
rgg_rand_supp_nodes = {}
rand_rgg_supp_nodes = {}
rand_rand_supp_nodes = {}

intra_rgg_edges = []
intra_rand_edges = []
inter_rgg_edges = []
inter_rand_edges = []

intra_edges_num = []
inter_edges_num = []                    # [for_edge, back_edge, for_supp_edge, back_supp_edge]

data_set = []


def cal_dist(cur_node, target_node):
    x1, y1 = coords[cur_node]
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

                        if cur_node in rgg_rgg_supp_nodes[target_node]:
                            for_supp_edges += 1
                        else:
                            for_edges += 1

                        if target_node in rgg_rgg_supp_nodes[cur_node]:
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
                for cur_target in rgg_rand_supp_nodes[target_node]:
                    if cur_target in cur_nodes:
                        net[cur_target, target_node, layer_names[cur_layer], layer_names[cur_layer + 1]] = 1
                        inter_rand_edges.append((cur_target, target_node, layer_names[cur_layer], layer_names[cur_layer + 1]))
                        for_supp_edges += 1
                    if for_supp_edges >= inter_edges_num[cur_layer][2]:
                        break
                if for_supp_edges >= inter_edges_num[cur_layer][2]:
                    break

            random.shuffle(cur_nodes)
            for cur_node in cur_nodes:
                for cur_target in rgg_rand_supp_nodes[cur_node]:
                    if cur_target in target_nodes:
                        net[cur_target, cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]] = 1
                        inter_rand_edges.append((cur_target, cur_node, layer_names[cur_layer + 1], layer_names[cur_layer]))
                        back_supp_edges += 1
                    if back_supp_edges >= inter_edges_num[cur_layer][3]:
                        break
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

            for cur_node in range(cur_layer * nodes, (cur_layer + 1) * nodes):
                temp_supp = []
                for target_node in target_nodes:
                    cur_dist = cal_dist(cur_node, target_node)
                    if cur_dist <= inter_thres:
                        temp_supp.append(target_node)
                rgg_rgg_supp_nodes[cur_node] = temp_supp

    elif (intra_type == 'Random') and (inter_type == 'RGG'):
        for cur_node in rgg_rgg_supp_nodes:
            rand_rgg_supp_nodes[cur_node] = rgg_rgg_supp_nodes[cur_node]

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
            random.shuffle(cur_layer_nodes)

            index = 0
            for cur_node in cur_layer_nodes:
                temp_supp = []
                cur_supp = len(rgg_rgg_supp_nodes[index])
                for i in range(cur_supp):
                    temp_supp.append(target_nodes[i])
                random.shuffle(target_nodes)

                rgg_rand_supp_nodes[cur_node] = temp_supp
                index += 1

    elif (intra_type == 'Random') and (inter_type == 'Random'):
        for cur_node in rgg_rand_supp_nodes:
            rand_rand_supp_nodes[cur_node] = rgg_rand_supp_nodes[cur_node]


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


def build_network(intra_type, inter_type):
    layer_names = []
    net = MultilayerNetwork(aspects=1, fullyInterconnected=False, directed=False)

    net, layer_names = make_network_layer(net, layer_names)
    net = make_nodes(net, layer_names, intra_type, inter_type)
    find_supporting_nodes(layer_names, intra_type, inter_type)
    net = make_edges(net, layer_names, intra_type, inter_type)

    return net


def analyse_initial_network(net):
    layer_names = net.get_layers()      # return dictionary
    layer_names = sorted(list(layer_names))

    stats = { "clustering":[],                              # Average clustering coefficient
              "mean degree":[],                             # Mean degree
              "components":[],                              # Components of the graph in each layers
              "largest component":[],                       # The largest component of the graphs
              "size of largest component":[],               # The size of the largest component
            }

    cur_layer = 0
    for layer in layer_names:
        edges = []
        for i in range((cur_layer * nodes), ((cur_layer + 1) * nodes)):
            for j in range((cur_layer * nodes), ((cur_layer + 1) * nodes)):
                if (i != j) & (net[i, j, layer, layer] == 1):
                    edges.append((i, j))

        """
                for edge in net.edges:
                    if edge[2] == edge[3] == layer:
                        edges.append(edge[:1])
        """

        G = nx.Graph()
        G.add_edges_from(edges)

        components = []
        for sub_G in nx.connected_components(G):
            components.append(sub_G)

        stats["clustering"].append(nx.average_clustering(G))
        stats["mean degree"].append(len(edges) * 2 / nodes)
        stats["components"].append(components)
        stats["largest component"].append(max(components, key=len))
        stats["size of largest component"].append(len(max(components, key=len)))

        cur_layer +=1

    keyList = stats.keys()
    for key in keyList:
        print("Key:%s\t"%key)
        print(stats[key])


def analyse_attacked_network(net):
    layer_names = net.get_layers()
    layer_names = sorted(list(layer_names))

    stats = { "clustering":[],                              # Average clustering coefficient
              "mean degree":[],                             # Mean degree
              "components":[],                              # Components of the graph in each layers
              "largest component":[],                       # The largest component of the graphs
              "size of largest component":[],               # The size of the largest component
            }


def draw_network(net, type):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=False)
    fig.savefig("%s Network.pdf" % type)


if __name__ == "__main__":
    start = time.time()

    rgg_rgg_net = build_network(intra_type='RGG', inter_type='RGG')
    # analyse_initial_network(rgg_rgg_net)
    # draw_network(rgg_rgg_net, type="intra_RGG, inter_RGG")
    print("1st: Done")

    rgg_rand_net = build_network(intra_type='RGG', inter_type='Random')
    # analyse_initial_network(rgg_rand_net)
    # draw_network(rgg_rand_net, type="intra_RGG, inter_Random")
    print("2nd: Done")

    rand_rgg_net= build_network(intra_type='Random', inter_type='RGG')
    # analyse_initial_network(rand_rgg_net)
    # draw_network(rand_rgg_net, type="intra_Random, inter_RGG")
    print("3rd: Done")

    rand_rand_net = build_network(intra_type='Random', inter_type='Random')
    # analyse_initial_network(rand_rand_net)
    # draw_network(rand_rand_net, type="intra_Random, inter_Random")
    print("4th: Done")

    print("time: ", time.time() - start)

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

    # att_rgg_rgg_net = cas.attack_network(rgg_rgg_net, coords, rgg_rgg_supp_nodes, attack_type, graph_type="RGG_RGG", attack_size=20)
    att_rgg_rgg_net = cas.attack_network(rgg_rgg_net, coords, rgg_rgg_supp_nodes, attack_type, graph_type="RGG_RGG", attack_size=attack_size, attack_point=(0.5, 0.5))
    # att_rgg_rgg_net = cas.attack_network(rgg_rgg_net, coords, rgg_rgg_supp_nodes, attack_type, graph_type="RGG_RGG", attack_point=(0.5, 0.5), attack_radius=0.1)

    # analyse_attacked_network(attacked_network)
    # draw_network(attacked_network, type="Attacked RGG")

    # att_rgg_rand_net = cas.attack_network(rgg_rand_net, coords, rgg_rand_supp_nodes, attack_type, graph_type="RGG_Rand", attack_size=20)
    att_rgg_rand_net = cas.attack_network(rgg_rand_net, coords, rgg_rand_supp_nodes, attack_type, graph_type="RGG_Rand", attack_size=attack_size, attack_point=(0.5, 0.5))
    # att_rgg_rand_net = cas.attack_network(rgg_rand_net, coords, rgg_rand_supp_nodes, attack_type, graph_type="RGG_Rand", attack_point=(0.5, 0.5), attack_radius=0.1)

    # analyse_attacked_network(attacked_rand_network)
    # draw_network(attacked_rand_network, type="Attacked Random")

    # att_rand_rgg_net = cas.attack_network(rand_rgg_net, coords, rand_rgg_supp_nodes, attack_type, graph_type="Rand_RGG", attack_size=20)
    att_rand_rgg_net = cas.attack_network(rand_rgg_net, coords, rand_rgg_supp_nodes, attack_type, graph_type="Rand_RGG", attack_size=attack_size, attack_point=(0.5, 0.5))
    # att_rand_rgg_net = cas.attack_network(rand_rgg_net, coords, rand_rgg_supp_nodes, attack_type, graph_type="Rand_RGG", attack_point=(0.5, 0.5), attack_radius=0.1)

    # analyse_attacked_network(attacked_rand_network)
    # draw_network(attacked_rand_network, type="Attacked Random")

    # att_rand_rand_net = cas.attack_network(rand_rand_net, coords, rand_rand_supp_nodes, attack_type, graph_type="Rand_Rand", attack_size=20)
    att_rand_rand_net = cas.attack_network(rand_rand_net, coords, rand_rand_supp_nodes, attack_type, graph_type="Rand_Rand", attack_size=attack_size, attack_point=(0.5, 0.5))
    # att_rand_rand_net = cas.attack_network(rand_rand_net, coords, rand_rand_supp_nodes, attack_type, graph_type="Rand_Rand", attack_point=(0.5, 0.5), attack_radius=0.1)

    # analyse_attacked_network(attacked_rand_network)

    # draw_network(att_rgg_rgg_net, type="Attacked intra_RGG, inter_RGG")
    # draw_network(att_rgg_rand_net, type="Attacked intra_RGG, inter_Rand")
    # draw_network(att_rand_rgg_net, type="Attacked intra_Rand, inter_RGG")
    # draw_network(att_rand_rand_net, type="Attacked intra_Rand, inter_Rand")

    print("time: ", time.time() - start)
