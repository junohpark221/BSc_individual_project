import networkx as nx
from pymnet import *
import random
import matplotlib
import math
matplotlib.use('TkAgg')

state = []
largest_components = []

isolated = []
not_supported = []


def initialise_state(coords):
    for i in coords:
        state.append(["Alive", -1])


def supported(net, coords, supporting_nodes, layer_names, cur_node):
    is_supported = False
    supporting_node = supporting_nodes[cur_node][1]

    cur_layer = int(cur_node // (len(coords)//len(layer_names)))
    supporting_layer = int(supporting_node // (len(coords)//len(layer_names)))

    if net[supporting_node, cur_node, layer_names[supporting_layer], layer_names[cur_layer]] == 1:
        is_supported = True

    return is_supported


def update_largest_components(net, coords, layer_names, cur_layer, step):
    edges = []
    for i in range((cur_layer * (len(coords) // len(layer_names))), ((cur_layer + 1) * (len(coords) // len(layer_names)))):
        for j in range((cur_layer * (len(coords) // len(layer_names))), ((cur_layer + 1) * (len(coords) // len(layer_names)))):
            if (i != j) & (net[i, j, layer_names[cur_layer], layer_names[cur_layer]] == 1):
                edges.append((i, j))

    G = nx.Graph()
    G.add_edges_from(edges)

    components = []
    for sub_G in nx.connected_components(G):
        components.append(sub_G)

    if len(components) != 0:
        largest_component = max(components, key=len)
    else:
        largest_component = []

    if step == 1:
        largest_components.append(largest_component)
    else:
        largest_components[cur_layer] = largest_component


def in_largest_component(net, coords, layer_names, cur_node, step):
    is_in_there = False
    cur_layer = int(cur_node // (len(coords) // len(layer_names)))

    if cur_node % (len(coords)//len(layer_names)) == 0:
        update_largest_components(net, coords, layer_names, cur_layer, step)

    if cur_node in largest_components[cur_layer]:
        is_in_there = True

    return is_in_there


# this is for attacking nodes.
def cascade_attack(net, coords, supporting_nodes, step):
    killed = []
    target_list = []
    local_isolated = 0
    local_not_supported = 0

    layer_names = net.get_layers()
    layer_names = sorted(list(layer_names))

    for cur_node in range(len(coords)):
        if state[cur_node][0] == "Alive":
            if not in_largest_component(net, coords, layer_names, cur_node, step):      # divide them
                state[cur_node][0] = "Dead from isolation"                              # different label style
                state[cur_node][1] = step
                killed.append(cur_node)
                local_isolated += 1

                cur_layer = int(cur_node // (len(coords) // len(layer_names)))
                neighbours = list(net[cur_node, layer_names[cur_layer]])

                for neighbour in neighbours:
                    target_list.append((cur_node, neighbour[0], layer_names[cur_layer], neighbour[1]))
            elif not supported(net, coords, supporting_nodes, layer_names, cur_node):
                state[cur_node][0] = "Dead from unsupported"  # different label style
                state[cur_node][1] = step
                killed.append(cur_node)
                local_not_supported += 1

                cur_layer = int(cur_node // (len(coords) // len(layer_names)))
                neighbours = list(net[cur_node, layer_names[cur_layer]])

                for neighbour in neighbours:
                    target_list.append((cur_node, neighbour[0], layer_names[cur_layer], neighbour[1]))

    global isolated, not_supported
    isolated.append(local_isolated)
    not_supported.append(local_not_supported)

    for target in target_list:
        net[target[0], target[1], target[2], target[3]] = 0

    return net, killed


def initial_attack(net, attack_list):
    print(state)
    for target in attack_list:
        state[target[0]][0] = "Initial Dead"
        state[target[0]][1] = 0

        neighbours = list(net[target[0], target[1]])

        for target_neighbour in neighbours:
            net[target[0], target_neighbour[0], target[1], target_neighbour[1]] = 0

    return net


def cal_dist(attack_point, target_node, coords):
    x1, y1 = attack_point
    x2, y2 = coords[target_node]
    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return d


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
    node_dist = []          # dictionary for nodes. key: distance of attack_point and node / item: list of nodes
                            # set the item as list since some nodes can have same key value(= distance)
                            # ex) node_dist = {0.25: [25, 87], 0.142: [1], 0.162: [63, 89, 74]...}

    cur_layer = attack_layer_list.index(attack_layer)

    # set different bounds for each attack_layers
    if attack_layer == 0:
        bound = range(len(coords))
    else:
        bound = range((cur_layer * len(coords) // len(net.get_layers())), (cur_layer + 1) * len(coords) // len(net.get_layers()))

    for i in bound:
        d = cal_dist(attack_point, i, coords)
        node_dist.append((i, d))

    s_node_dist = sorted(node_dist, key=lambda dist: dist[1])

    for m in range(attack_size):
        targets.append(s_node_dist[m][0])

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
        d = cal_dist(attack_point, i, coords)

        if d <= attack_radius:
            targets.append(i)

    return targets


def analyse_attacked_network(net, coords, step, killed_nodes):
    print("\n\n\n---------------------------------Step %d----------------------------------" % step)
    layer_names = net.get_layers()
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

        for i in range((cur_layer * (len(coords) // len(layer_names))), ((cur_layer + 1) * (len(coords) // len(layer_names)))):
            for j in range((cur_layer * (len(coords) // len(layer_names))), ((cur_layer + 1) * (len(coords) // len(layer_names)))):
                if (i != j) & (net[i, j, layer, layer] == 1):
                    edges.append((i, j))

        G = nx.Graph()
        G.add_edges_from(edges)

        components = []
        for sub_G in nx.connected_components(G):
            components.append(sub_G)

        if len(edges) == 0:
            stats["clustering"].append(0)
        else:
            stats["clustering"].append(nx.average_clustering(G))
        stats["mean degree"].append(len(edges) * 2 / (len(coords) // len(layer_names)))
        stats["components"].append(components)
        if len(components) == 0:
            stats["largest component"].append([])
            stats["size of largest component"].append(0)
        else:
            stats["largest component"].append(max(components, key=len))
            stats["size of largest component"].append(len(max(components, key=len)))

        cur_layer += 1

    keyList = stats.keys()
    for key in keyList:
        print("Key:%s\t" % key)
        print(stats[key])

    print("The number of nodes killed in step %d: %d" % (step, killed_nodes))
    if step != 0:
        print("The number of dead nodes since they are not in the largest component: %d" % isolated[step - 1])
        print("The number of dead nodes since they are not supported: %d" % not_supported[step - 1])


def draw_network(net, coords, step):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5)
    fig.savefig("Cascade attack step %d.pdf" % step)


def attack_network(net, coords, supporting_nodes, attack_type, graph_type="RGG", attack_size=5, attack_layer='a', attack_point=(0.5, 0.5), attack_radius=0.1):
    targets = []        # list of the initial target nodes
    killed = []

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

        if graph_type == "Random":
            global state, largest_components, isolated, not_supported
            state = []
            largest_components = []
            isolated = []
            not_supported = []

        initialise_state(coords)

        # With the different attack_type, sample the targets differently.
        if attack_type == "normal":
            targets = random.sample(list(coords), attack_size)
        elif attack_type == "spatial_number":
            targets = find_near_nodes(net, coords, attack_size, attack_layer, attack_point, attack_layer_list)
        elif attack_type == "spatial_range":
            targets = find_nodes_in_range(net, coords, attack_layer, attack_point, attack_radius, attack_layer_list)

        attack_list = make_node_layer_pair(net, coords, targets)

        step = 0

        net = initial_attack(net, attack_list)

        killed = targets
        analyse_attacked_network(net, coords, step, len(killed))
        # draw_network(net, coords, step)

        step += 1

        # repeat the kill_nodes until the updated attack_list is empty.
        while len(killed) != 0:

            net, killed = cascade_attack(net, coords, supporting_nodes, step)

            if len(killed) != 0:
                analyse_attacked_network(net, coords, step, len(killed))
                # draw_network(net, coords, step)
            step += 1

        total_alive = 0
        for cur_state in state:
            if cur_state[0] == 'Alive':
                total_alive += 1
        print("\n\n\nAlive nodes: %d" % total_alive)

        print(supporting_nodes)
        print(state)

        return net

    except Exception as e:
        print("Error is occurred", e)

