import networkx as nx
from pymnet import *
import random
import matplotlib
import math
matplotlib.use('TkAgg')


# this is for attacking nodes.
def kill_nodes(net, attack_list):
    new_dead_list = []

    # for every nodes in attack_list, find the neighbours of them and check whether they will be killed or not.
    for i in range(len(attack_list)):
        target = attack_list[i]
        neighbours = list(net[target[0], target[1]])

        for j in range(len(neighbours)):
            target_neighbour = neighbours[j]
            net[target[0], target_neighbour[0], target[1], target_neighbour[1]] = 0

            # after remove the edge, if the neighbour does not have any edge or the neighbour is in different layer, it will be killed next time.
            if (net[target_neighbour[0], target_neighbour[1]].deg() == 0) or (target[1] != target_neighbour[1]):
                new_dead_list.append(target_neighbour)

    return net, new_dead_list


# make a list of tuples, (target node, layer of the node)
def make_node_layer_pair(net, coords, targets):
    node_layer_pair = []

    layer_names = net.get_layers()
    layer_names = sorted(list(layer_names))

    for i in range(len(targets)):
        if targets[i] == 0:
            node_layer_pair.append((targets[i], layer_names[0]))
        else:
            node_layer_pair.append((targets[i], layer_names[targets[i]//(len(coords)//len(layer_names))]))

    return node_layer_pair


# this is for "spatial_number" attack. find the nearest (attack_number) nodes from the attack_point.
def find_near_nodes(net, coords, attack_size, attack_layer, attack_point, attack_layer_list):
    targets = []
    node_dist = {}          # dictionary for nodes. key: distance of attack_point and node / item: list of nodes
                            # set the item as list since some nodes can have same key value(= distance)
                            # ex) node_dist = {0.25: [25, 87], 0.142: [1], 0.162: [63, 89, 74]...}

    cur_layer = attack_layer_list.index(attack_layer)

    # set different bounds for each attack_layers
    if attack_layer == 0:
        bound = range(len(coords))
    else:
        bound = range((cur_layer * len(coords) // len(net.get_layers())), (cur_layer + 1) * len(coords) // len(net.get_layers()))

    for i in bound:
        x1, y1 = attack_point
        x2, y2 = coords[i]
        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # if the distance is already in dictionary, add the node in the node list of the key value.
        if d in node_dist:
            node_dist[d].append(i)
        else:
            node_dist[d] = [i]

    node_dist_key = list(node_dist.keys())

    s_node_dist = dict(sorted(node_dist.items()))
    s_node_dist_key = sorted(node_dist_key)

    breaker = 0
    # find the first (attack_size) number of nodes in the sorted list.
    for m in range(len(coords)):
        if breaker >= attack_size:
            break
        for n in range(len(s_node_dist[s_node_dist_key[m]])):
            if breaker >= attack_size:
                break
            targets.append(s_node_dist[s_node_dist_key[m]][n])
            breaker += 1

    return targets


# this is for "spatial_range" attack. find the nodes in the attack circle
def find_nodes_in_range(net, coords, attack_layer, attack_point, attack_radius, attack_layer_list):
    targets = []
    cur_layer = attack_layer_list.index(attack_layer)

    # set different bounds for each attack_layers
    if attack_layer == 0:
        bound = range(len(coords))
    else:
        bound = range((cur_layer * len(coords) // len(net.get_layers())), (cur_layer + 1) * len(coords) // len(net.get_layers()))

    for i in bound:
        x1, y1 = attack_point
        x2, y2 = coords[i]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2

        if d <= attack_radius:
            targets.append(i)

    return targets


def attack_network(net, coords, attack_type, attack_size=5, attack_layer='a', attack_point=(0.5, 0.5), attack_radius=0.1):
    targets = []        # list of the initial target nodes

    attack_layer_list = sorted(list(net.get_layers()))
    attack_layer_list.append(0)

    # Using try-except method, check the errors when the parameters have wrong value.
    try:
        if attack_type not in ["normal", "spatial_number", "spatial_range"]:
            raise Exception("Wrong attack_type")
        elif attack_layer not in attack_layer_list:
            raise Exception("Wrong attack_layer")
        elif (attack_layer in list(net.get_layers()) and attack_size > (len(coords) // len(net.get_layers()))) or (attack_layer == 0 and attack_size > len(coords)) or (attack_size <= 0):
            raise Exception("Wrong attack_size")
        elif attack_point[0] > 1 or attack_point[0] < 0 or attack_point[1] > 1 or attack_point[1] < 0:
            raise Exception("Wrong attack_point")
        elif attack_radius > 0.5 or attack_radius < 0:
            raise Exception("Wrong attack_radius")

        # With the different attack_type, sample the targets differently.
        if attack_type == "normal":
            targets = random.sample(list(coords), attack_size)
        elif attack_type == "spatial_number":
            targets = find_near_nodes(net, coords, attack_size, attack_layer, attack_point, attack_layer_list)
        elif attack_type == "spatial_range":
            targets = find_nodes_in_range(net, coords, attack_layer, attack_point, attack_radius, attack_layer_list)

        attack_list = make_node_layer_pair(net, coords, targets)

        # repeat the kill_nodes until the updated attack_list is empty.
        while len(attack_list) != 0:
            net, attack_list = kill_nodes(net, attack_list)

        return net

    except Exception as e:
        print("Error is occurred", e)

