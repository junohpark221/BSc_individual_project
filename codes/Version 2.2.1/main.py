import networkx as nx
from pymnet import *
import random
import matplotlib
import cascade as cas
import math

matplotlib.use('TkAgg')

nodes = 100
layers = 3
intra_thres = 0.08
inter_thres = 0.08

coords = {}
supporting_nodes = []

attack_type = "normal"  # choose one of the "normal", "spatial_number", "spatial_range"


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


def find_supporting_nodes(net):
    layer_list = sorted(list(net.get_layers()))

    for cur_node in range(len(coords)):
        supporting_node = -1
        cur_layer_index = int(cur_node // nodes)
        neighbours = list(net[cur_node, layer_list[cur_layer_index]])
        target_layers = []

        if cur_layer_index == 0:
            target_layers.append(layer_list[cur_layer_index + 1])
        elif cur_layer_index == (len(layer_list) - 1):
            target_layers.append(layer_list[cur_layer_index - 1])
        else:
            target_layers.append(layer_list[cur_layer_index - 1])
            target_layers.append(layer_list[cur_layer_index + 1])

        if len(neighbours) != 0:
            supporting_node = find_nearest_node(cur_node, supporting_node, neighbours, target_layers)

        supporting_nodes.append((cur_node, supporting_node))


def make_interlayer_edges(net, cur_layer, layer_names):
    if cur_layer != 0:
        for i in range(cur_layer * nodes, (cur_layer + 1) * nodes):
            for j in range((cur_layer - 1) * nodes, cur_layer * nodes):
                d = cal_dist(i, j)
                if d <= inter_thres:
                    net[i, j, layer_names[cur_layer], layer_names[cur_layer - 1]] = 1

    if cur_layer != (len(layer_names) - 1):
        for i in range((cur_layer * nodes), (cur_layer + 1) * nodes):
            for j in range((cur_layer + 1) * nodes, (cur_layer + 2) * nodes):
                d = cal_dist(i, j)
                if d <= inter_thres:
                    net[i, j, layer_names[cur_layer], layer_names[cur_layer + 1]] = 1

    return net


def make_intralayer_edges(net, cur_layer, cur_layer_name):
    for i in range(nodes):
        for j in range(nodes):
            if i != j:
                d = cal_dist(i, j)
                if d <= intra_thres:
                    net[(cur_layer * nodes) + i, (cur_layer * nodes) + j, cur_layer_name, cur_layer_name] = 1

    return net


def make_edges(net, layer_names):
    for cur_layer in range(layers):
        net = make_intralayer_edges(net, cur_layer, layer_names[cur_layer])
        net = make_interlayer_edges(net, cur_layer, layer_names)

    return net


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
    net = MultilayerNetwork(aspects=1, fullyInterconnected=False)

    net, layer_names = make_network_layer(net, layer_names)
    net = make_nodes(net, layer_names)
    net = make_edges(net, layer_names)

    find_supporting_nodes(net)

    return net


def analyse_network(net):
    layer_names = net.get_layers()  # return dictionary

    layer_names = sorted(list(layer_names))

    stats = {"clustering": [],  # Average clustering coefficient
             "mean degree": [],  # Mean degree
             "components": [],  # Components of the graph in each layers
             "largest component": [],  # The largest component of the graphs
             "size of largest component": [],  # The size of the largest component
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

        cur_layer += 1

    keyList = stats.keys()
    for key in keyList:
        print("Key:%s\t" % key)
        print(stats[key])


def draw_network(net, attacked=False):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule': 'scaled', 'scalecoeff': 0.01},
               defaultEdgeWidth=0.5, show=True)
    if attacked:
        fig.savefig("Attacked Network.pdf")
    else:
        fig.savefig("Initial Network.pdf")


if __name__ == "__main__":
    network = build_network()
    analyse_network(network)
    draw_network(network, attacked=False)

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

    # These are for the "normal", "spatial_number" and "spatial_range" attacks each.

    # attacked_network = cas.attack_network(network, coords, supporting_nodes, attack_type, attack_size=20)
    attacked_network = cas.attack_network(network, coords, supporting_nodes, attack_type, attack_size=20,
                                          attack_layer='a', attack_point=(0.5, 0.5))
    # attacked_network = cas.attack_network(network, coords, supporting_nodes, attack_type, attack_layer='a', attack_point=(0.5, 0.5), attack_radius=0.1)

    analyse_network(attacked_network)
    draw_network(attacked_network, attacked=True)