import networkx as nx
from pymnet import *
import random
import matplotlib
import math
import statistics
import glob
from PIL import Image
matplotlib.use('TkAgg')

state = []
largest_components = []

isolated = []
not_supported = []


def initialise_state(coords):
    for i in coords:
        state.append(["Alive", -1])


def supported(net, coords, supporting_nodes, layer_names, cur_node, supp_node_num):
    is_supported = False
    supp_nodes = supporting_nodes[cur_node]

    cur_layer = int(cur_node // (len(coords)//len(layer_names)))

    d = 0
    for supp_node in supp_nodes:
        if supp_node != -1:
            supporting_layer = int(supp_node // (len(coords) // len(layer_names)))
            if net[supp_node, cur_node, layer_names[supporting_layer], layer_names[cur_layer]] == 1:
                d += 1

    if d != 0:
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
        largest_component = [-1]

    if step == 1:
        largest_components.append(largest_component)
    else:
        largest_components[cur_layer] = largest_component


def in_largest_component(net, coords, layer_names, prev_node, cur_node, step):
    is_in_there = False
    cur_layer = int(cur_node // (len(coords) // len(layer_names)))

    if (prev_node // (len(coords)//len(layer_names))) != (cur_node // (len(coords)//len(layer_names))):
        update_largest_components(net, coords, layer_names, cur_layer, step)

    if cur_node in largest_components[cur_layer]:
        is_in_there = True

    return is_in_there


# this is for attacking nodes.
def cascade_attack(net, coords, supporting_nodes, step, supp_node_num):
    killed = []
    target_list = []
    local_isolated = 0
    local_not_supported = 0

    layer_names = net.get_layers()
    layer_names = sorted(list(layer_names))

    prev_node = -1
    for cur_node in range(len(coords)):
        if state[cur_node][0] == "Alive":
            if not in_largest_component(net, coords, layer_names, prev_node, cur_node, step):      # divide them
                state[cur_node][0] = "Dead from isolation"                                         # different label style
                state[cur_node][1] = step
                killed.append(cur_node)
                local_isolated += 1

                cur_layer = int(cur_node // (len(coords) // len(layer_names)))
                neighbours = list(net[cur_node, layer_names[cur_layer]])

                for neighbour in neighbours:
                    target_list.append((cur_node, neighbour[0], layer_names[cur_layer], neighbour[1]))

            elif not supported(net, coords, supporting_nodes, layer_names, cur_node, supp_node_num):
                state[cur_node][0] = "Dead from unsupported"  # different label style
                state[cur_node][1] = step
                killed.append(cur_node)
                local_not_supported += 1

                cur_layer = int(cur_node // (len(coords) // len(layer_names)))
                neighbours = list(net[cur_node, layer_names[cur_layer]])

                for neighbour in neighbours:
                    target_list.append((cur_node, neighbour[0], layer_names[cur_layer], neighbour[1]))

            prev_node = cur_node

    global isolated, not_supported
    isolated.append(local_isolated)
    not_supported.append(local_not_supported)

    for target in target_list:
        if target[0] != target[1]:
            net[target[0], target[1], target[2], target[3]] = 0

    return net, killed


def initial_attack(net, attack_list):
    for target in attack_list:
        state[target[0]][0] = "Initial Dead"
        state[target[0]][1] = 0

        neighbours = list(net[target[0], target[1]])

        for target_neighbour in neighbours:
            if target[0] != target_neighbour[0]:
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
def find_near_nodes(net, coords, attack_size, attack_point, attack_layer_list):
    targets = []
    node_dist = []          # dictionary for nodes. key: distance of attack_point and node / item: list of nodes
                            # set the item as list since some nodes can have same key value(= distance)
                            # ex) node_dist = {0.25: [25, 87], 0.142: [1], 0.162: [63, 89, 74]...}

    for cur_layer in range(len(net.get_layers())):
        for cur_node in range((cur_layer*len(coords)//len(net.get_layers())), (cur_layer + 1)*len(coords)//len(net.get_layers())):
            d = cal_dist(attack_point, cur_node, coords)
            node_dist.append((cur_node, d))

        s_node_dist = sorted(node_dist, key=lambda dist: dist[1])

        for m in range(attack_size):
            targets.append(s_node_dist[m][0])

        node_dist = []

    return targets


# this is for "spatial_range" attack. find the nodes in the attack circle
def find_nodes_in_range(net, coords, attack_point, attack_radius, attack_layer_list):
    targets = []

    # set different bounds for each attack_layers
    for cur_layer in range(len(attack_layer_list)):
        for cur_node in range((cur_layer*len(coords)//len(attack_layer_list)), (cur_layer + 1)*len(coords)//len(net.get_layers())):
            d = cal_dist(attack_point, cur_node, coords)

            if d <= attack_radius:
                targets.append(cur_node)

    return targets


def analyse_attacked_network(net, coords, supporting_nodes, cas_data, step, attack_point, killed_nodes):
    # fin_intra_edge, fin_inter_edge, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_far_node, fin_clust, fin_mean_deg, fin_larg_comp, deg_assort, dist_deg_cent, dist_bet_cent, step.....
    # print("\n\n\n---------------------------------Step %d----------------------------------" % step)
    layer_names = net.get_layers()
    layer_names = sorted(list(layer_names))

    stats = { "intra_edge":[],                              # Number of intra edges
              "clustering":[],                              # Average clustering coefficient
              "mean degree":[],                             # Mean degree
              "components":[],                              # Components of the graph in each layers
              "largest component":[],                       # The largest component of the graphs
              "size of largest component":[],               # The size of the largest component
              "degree assortativity":[],                    # Degree assortativity of the alive graph
              "distance to degree centre":[],               # Distance from attack point to degree centre
              "distance to betweennes centre": []           # Distance from attack point to betweennes centre
            }

    index = 0
    cur_layer = 0

    supp_edge_num = 0
    total_alive = 0
    total_unsupp = 0
    total_isol = 0
    far_dist = 0
    for layer in layer_names:
        intra_edges = []
        for edge in net.edges:
            if edge[2] == edge[3] == layer:
                intra_edges.append(edge[:2])
            else:
                supp_edge_num += 1

        G = nx.Graph()
        G.add_edges_from(intra_edges)

        components = list(nx.connected_components(G))

        alive_nodes = []
        for cur_state in state[cur_layer*len(coords)//len(layer_names) : (cur_layer + 1)*len(coords)//len(layer_names)]:
            if cur_state[0] == 'Alive':
                d = cal_dist(attack_point, index, coords)
                if d >= far_dist:
                    far_dist = d

                alive_nodes.append(index)
                total_alive += 1
            elif cur_state[0] == 'Dead from isolation':
                total_isol += 1
            elif cur_state[0] == 'Dead from unsupported':
                total_unsupp += 1
            index += 1

        stats["intra_edge"].append(len(intra_edges))
        if len(intra_edges) == 0:
            stats["clustering"].append(0)
        else:
            stats["clustering"].append(nx.average_clustering(G))
        stats["mean degree"].append(len(intra_edges) * 2 / (len(coords) // len(layer_names)))
        stats["components"].append(components)
        if len(components) == 0:
            stats["largest component"].append([])
            stats["size of largest component"].append(0)
        else:
            stats["largest component"].append(max(components, key=len))
            stats["size of largest component"].append(len(max(components, key=len)))
        if len(alive_nodes) == 0:
            stats["degree assortativity"].append(0)
            stats["distance to degree centre"].append(0)
            stats["distance to betweennes centre"].append(0)
        else:
            try:
                nx.degree_assortativity_coefficient(G, nodes=alive_nodes)
            except Exception as e:
                print("degree_assortativity_coefficient is not exist")
                stats["degree assortativity"].append(0)
            else:
                stats["degree assortativity"].append(nx.degree_assortativity_coefficient(G, nodes=alive_nodes))

            deg_cent = nx.degree_centrality(G)
            bet_cent = nx.betweenness_centrality(G)
            deg_cent = list(sorted(deg_cent.items(), key=lambda x:x[1], reverse=True))
            bet_cent = list(sorted(bet_cent.items(), key=lambda x: x[1], reverse=True))

            d = cal_dist(attack_point, deg_cent[0][0], coords)
            stats["distance to degree centre"].append(d)

            d = cal_dist(attack_point, bet_cent[0][0], coords)
            stats["distance to betweennes centre"].append(d)
        cur_layer += 1

    cas_data.append(statistics.mean(stats["intra_edge"]))
    cas_data.append(supp_edge_num / (len(layer_names)))
    cas_data.append(total_alive)
    cas_data.append(total_isol)
    cas_data.append(total_unsupp)
    cas_data.append(step)
    cas_data.append(far_dist)
    cas_data.append(statistics.mean(stats["clustering"]))
    cas_data.append(statistics.mean(stats["mean degree"]))
    cas_data.append(statistics.mean(stats["size of largest component"]))
    cas_data.append(statistics.mean(stats["degree assortativity"]))
    cas_data.append(statistics.mean(stats["distance to degree centre"]))
    cas_data.append(statistics.mean(stats["distance to betweennes centre"]))
    cas_data.append(isolated)
    cas_data.append(not_supported)

    del G

    return cas_data

"""
    keyList = stats.keys()
    for key in keyList:
        print("Key:%s\t" % key)
        print(stats[key])

    print("The number of nodes killed in step %d: %d" % (step, killed_nodes))
    if step != 0:
        print("The number of dead nodes since they are not in the largest component: %d" % isolated[step - 1])
        print("The number of dead nodes since they are not supported: %d" % not_supported[step - 1])
"""


def draw_network(net, coords, step):
    rep = 1
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, layerLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5)
    fig.savefig("-{:02d}-{:03d}.png".format(rep, step), dpi=300)
    # fig.savefig("Cascade attack step %d.pdf" % step)


def stitch():
    rep = 1

    fp_in = "-{:02d}-*.png".format(rep)
    fp_out = "-{:02d}.gif".format(rep)
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=500, loop=0)


def attack_network(net, coords, supporting_nodes, supp_node_num, cas_data, attack_type, graph_type="RGG_RGG", attack_size=5, attack_point=(0.5, 0.5), attack_radius=0.1):
    targets = []        # list of the initial target nodes
    killed = []

    attack_layer_list = sorted(list(net.get_layers()))
    # Using try-except method, check the errors when the parameters have wrong value.

    del state[:]
    del largest_components[:]
    del isolated[:]
    del not_supported[:]

    initialise_state(coords)

    # With the different attack_type, sample the targets differently.
    if attack_type == "normal":
        targets = random.sample(list(coords), attack_size)
    elif attack_type == "spatial_number":
        targets = find_near_nodes(net, coords, attack_size, attack_point, attack_layer_list)
    elif attack_type == "spatial_range":
        targets = find_nodes_in_range(net, coords, attack_point, attack_radius, attack_layer_list)

    attack_list = make_node_layer_pair(net, coords, targets)

    step = 0
    net = initial_attack(net, attack_list)

    killed = targets

    step += 1

    # repeat the kill_nodes until the updated attack_list is empty.
    while len(killed) != 0:
        net, killed = cascade_attack(net, coords, supporting_nodes, step, supp_node_num)

        # if len(killed) != 0:
        # analyse_attacked_network(net, coords, step, len(killed))
        # draw_network(net, coords, step)
        if len(killed) != 0:
            # draw_network(net, coords, step)
            # print("Step %d is finished." % step)
            step += 1

    cas_data = analyse_attacked_network(net, coords, supporting_nodes, cas_data, step, attack_point, len(killed))

    print("Number of steps: ", step)
    # stitch()

    return net, cas_data


"""
    try:
        del state[:]
        del largest_components[:]
        del isolated[:]
        del not_supported[:]

        if attack_type not in ["normal", "spatial_number", "spatial_range"]:
            raise Exception("Wrong attack_type")
        elif attack_size > (len(coords) // len(net.get_layers())) or (attack_size <= 0):
            raise Exception("Wrong attack_size")
        elif attack_point[0] > 1 or attack_point[0] < 0 or attack_point[1] > 1 or attack_point[1] < 0:
            raise Exception("Wrong attack_point")
        elif attack_radius > 0.5 or attack_radius < 0:
            raise Exception("Wrong attack_radius")

        initialise_state(coords)

        # With the different attack_type, sample the targets differently.
        if attack_type == "normal":
            targets = random.sample(list(coords), attack_size)
        elif attack_type == "spatial_number":
            targets = find_near_nodes(net, coords, attack_size, attack_point, attack_layer_list)
        elif attack_type == "spatial_range":
            targets = find_nodes_in_range(net, coords, attack_point, attack_radius, attack_layer_list)

        attack_list = make_node_layer_pair(net, coords, targets)

        step = 0
        net = initial_attack(net, attack_list)
        draw_network(net, coords, step)

        killed = targets

        step += 1

        # repeat the kill_nodes until the updated attack_list is empty.
        while len(killed) != 0:
            net, killed = cascade_attack(net, coords, supporting_nodes, step)

            # if len(killed) != 0:
            # analyse_attacked_network(net, coords, step, len(killed))
            # draw_network(net, coords, step)
            if len(killed) != 0:
                draw_network(net, coords, step)
                print("Step %d is finished." % step)
                step += 1

        cas_data = analyse_attacked_network(net, coords, supporting_nodes, cas_data, step, attack_point, len(killed))

        print("Number of steps: ", step)
        stitch()

        
        total_alive = 0
        total_unsupp = 0
        total_isol = 0
        total_init = 0
        total_else = 0
        for cur_state in state:
            if cur_state[0] == 'Alive':
                total_alive += 1
            elif cur_state[0] == 'Initial Dead':
                total_init += 1
            elif cur_state[0] == 'Dead from isolation':
                total_isol += 1
            elif cur_state[0] == 'Dead from unsupported':
                total_unsupp += 1
    
        print("\n\n\nAlive nodes: %d" % total_alive)
        print("Initial Dead: %d" % total_init)
        print("Dead from isolation: %d" % total_isol)
        print("Dead from unsupported: %d\n\n\n" % total_unsupp)
    
        print(supporting_nodes)
        print(state)
        

        return net, cas_data
    except Exception as e:
        print("Error is occurred", e)
"""