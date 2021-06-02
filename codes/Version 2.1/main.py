import networkx as nx
from pymnet import *
import random
import matplotlib
import cascade as cas
matplotlib.use('TkAgg')

nodes = 100
layers = 3
intra_thres = 0.1
inter_thres = 0.1

coords = {}

attack_type = "normal"          # choose one of the "normal", "spatial_number", "spatial_range"


def make_interlayer_edges(net, cur_layer, layer_names):
    for i in range(nodes):
        for j in range(nodes):
            if i != j:
                x1, y1 = coords[(cur_layer * nodes) + i]
                x2, y2 = coords[((cur_layer + 1) * nodes) + j]
                d = (x1 - x2)**2 + (y1-y2)**2
                if d <= (inter_thres**2):
                    net[i, j, layer_names[cur_layer], layer_names[cur_layer + 1]] = 1

    return net


def make_intralayer_edges(net, cur_layer, cur_layer_name):
    for i in range(nodes):
        for j in range(nodes):
            if i != j:
                x1, y1 = coords[(cur_layer * nodes) + i]
                x2, y2 = coords[(cur_layer * nodes) + j]
                d = (x1 - x2)**2 + (y1-y2)**2
                if d <= (intra_thres**2):
                    net[(cur_layer * nodes) + i, (cur_layer * nodes) + j, cur_layer_name, cur_layer_name] = 1

    return net


def make_edges(net, layer_names):
    for cur_layer in range(layers):
        net = make_intralayer_edges(net, cur_layer, layer_names[cur_layer])

        if cur_layer != (layers - 1):
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

    return net


def analyse_network(net):
    layer_names = net.get_layers()      # return dictionary

    layer_names = sorted(list(layer_names))

    stats = { "clustering":[],          # Average clustering coefficient
              "mean degree":[],         # Mean degree
              "components":[],          # Components of the graph in each layers
              "largest component":[],   # The largest component of the graphs
              "size of largest component":[],                # The size of the largest component
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


def draw_network(net, attacked=False):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=True)
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
    
        For "normal" attack,                cas.attack_network(network, coords, attack_type, attack_size=20)
        For "spatial_number" attack,        cas.attack_network(network, coords, attack_type, attack_size=20, attack_layer='a', attack_point=(0.5, 0.5))
        For "spatial_range" attack,         cas.attack_network(network, coords, attack_type, attack_layer='a', attack_point=(0.5, 0.5), attack_radius=0.1)
        
        attack_size = 20                        # number of nodes that will be initially killed
        attack_layer = 'a'                      # the target layer of the attack.
                                                  'a', 'b', 'c'... means the specific layer. 0 means that suppose every nodes are in the same layer.
        attack_point = (0.5, 0.5)               # attack point for spatial_number and spatial_range attacks
        attack_radius = 0.1                     # the radius of attack in spatial_range attacks
    """

    # These are for the "normal", "spatial_number" and "spatial_range" attacks each.

    attacked_network = cas.attack_network(network, coords, attack_type, attack_size=20)
    # attacked_network = cas.attack_network(network, coords, attack_type, attack_size=20, attack_layer='a', attack_point=(0.5, 0.5))
    # attacked_network = cas.attack_network(network, coords, attack_type, attack_layer='a', attack_point=(0.5, 0.5), attack_radius=0.1)

    analyse_network(attacked_network)
    draw_network(attacked_network, attacked=True)


