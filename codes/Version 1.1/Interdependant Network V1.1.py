import networkx as nx
from pymnet import *
import random
import matplotlib
matplotlib.use('TkAgg')

def make_interlayer_edges(net, pos_each_layers, thres, layer_names, cur_layer, target_layer):
    height = thres/2
    cur_pos = pos_each_layers[cur_layer]
    target_pos = pos_each_layers[target_layer]

    for i in cur_pos:
        for j in target_pos:
            if i != j:
                x1, y1 = cur_pos[i]
                x2, y2 = target_pos[j]
                d = (x1 - x2)**2 + (y1-y2)**2 + height**2
                if d <= (thres**2):
                    net[i, j, layer_names[cur_layer], layer_names[target_layer]] = 1

    return net

def make_intralayer_edges(net, cur_pos, thres, cur_layer_name):
    for i in cur_pos:
        for j in cur_pos:
            if i != j:
                x1, y1 = cur_pos[i]
                x2, y2 = cur_pos[j]
                d = (x1 - x2)**2 + (y1-y2)**2
                if d <= (thres**2):
                    net[i, j, cur_layer_name, cur_layer_name] = 1

    return net

def make_edges(net, thres, layers, pos_each_layers):
    layer_names = []
    for i in range(0, layers - 1):
        layer_name = chr(97 + i)
        layer_names.append(layer_name)

    for cur_layer in range(0, layers - 1):
        cur_pos = pos_each_layers[cur_layer]
        net = make_intralayer_edges(net, cur_pos, thres, layer_names[cur_layer])

        if cur_layer != (layers - 1):
            for target_layer in range(cur_pos + 1, layers - 1): # range(cur_pos+1,cur_pos+1)
                net = make_interlayer_edges(net, pos_each_layers, thres, layer_names, cur_layer, target_layer)

    return net

def make_network_layer(nodes, thres, cur_layer, net):
    layer_name = chr(97+cur_layer)
    G = nx.random_geometric_graph(nodes, thres)
    pos = nx.get_node_attributes(G, "pos")
    net.add_layer(layer_name)

    i = 0
    for n in pos:
        net.add_node(n, layer_name)
        i += 1

    return net, pos

def build_network(nodes, thres, layers):
    cur_layer = 0
    pos_each_layers = []
    net = MultilayerNetwork()

 #   coords = {}
 #   net.add_layer(layer_name)
 #   for i in range(N):
 #       coords[i] = (random.random(),random.random())
 #       net.add_node(i,layer_name)

 #   1 .. 100 Layer A, 101-200 Layer B

 #   for i in N:
 #      for j in N:
 #           if dist(i,j,coords)<thrsh:
 #               net[i,j,layer_name,layer_name] = 1

 #   for i in range

    for n in range(1, layers):
        new_net, pos = make_network_layer(nodes, thres, cur_layer, net)
        pos_each_layers.append(pos)
        net = new_net
        cur_layer += 1

    net = make_edges(net, thres, layers, pos_each_layers)

    return net

def draw_network(net):
    fig = draw(net, show=True)
    fig.savefig("net4.pdf")

if __name__ == "__main__":
    network = build_network(200, 0.15, 3)
    draw_network(network)

