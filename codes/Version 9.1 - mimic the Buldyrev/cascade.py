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


def initialise_state(tot_nodes):
    for i in range(tot_nodes):
        state.append(["Alive", -1])


def supported(net, tot_nodes, supporting_nodes, layer_names, cur_node):
    is_supported = False
    supporting_node = supporting_nodes[cur_node]

    cur_layer = int(cur_node // (tot_nodes//len(layer_names)))
    supporting_layer = int(supporting_node // (tot_nodes//len(layer_names)))

    if net[supporting_node, cur_node, layer_names[supporting_layer], layer_names[cur_layer]] == 1:
        is_supported = True

    return is_supported


def update_largest_components(net, tot_nodes, layer_names, cur_layer, step):
    edges = []
    for i in range((cur_layer * (tot_nodes // len(layer_names))), ((cur_layer + 1) * (tot_nodes // len(layer_names)))):
        for j in range((cur_layer * (tot_nodes // len(layer_names))), ((cur_layer + 1) * (tot_nodes // len(layer_names)))):
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


def in_largest_component(net, tot_nodes, layer_names, prev_node, cur_node, step):
    is_in_there = False
    cur_layer = int(cur_node // (tot_nodes // len(layer_names)))

    if (prev_node // (tot_nodes//len(layer_names))) != (cur_node // (tot_nodes//len(layer_names))):
        update_largest_components(net, tot_nodes, layer_names, cur_layer, step)

    if cur_node in largest_components[cur_layer]:
        is_in_there = True

    return is_in_there


# this is for attacking nodes.
def cascade_attack(net, tot_nodes, supporting_nodes, step):
    killed = []
    target_list = []
    local_isolated = 0
    local_not_supported = 0

    layer_names = net.get_layers()
    layer_names = sorted(list(layer_names))

    prev_node = -1
    for cur_node in range(tot_nodes):
        if state[cur_node][0] == "Alive":
            if not in_largest_component(net, tot_nodes, layer_names, prev_node, cur_node, step):      # divide them
                state[cur_node][0] = "Dead from isolation"                              # different label style
                state[cur_node][1] = step
                killed.append(cur_node)
                local_isolated += 1

                cur_layer = int(cur_node // (tot_nodes // len(layer_names)))
                neighbours = list(net[cur_node, layer_names[cur_layer]])

                for neighbour in neighbours:
                    target_list.append((cur_node, neighbour[0], layer_names[cur_layer], neighbour[1]))

            elif not supported(net, tot_nodes, supporting_nodes, layer_names, cur_node):
                state[cur_node][0] = "Dead from unsupported"  # different label style
                state[cur_node][1] = step
                killed.append(cur_node)
                local_not_supported += 1

                cur_layer = int(cur_node // (tot_nodes // len(layer_names)))
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


# make a list of tuples, (target node, layer of the node)
def make_node_layer_pair(net, tot_nodes, targets):
    node_layer_pair = []

    layer_names = net.get_layers()
    layer_names = sorted(list(layer_names))

    for i in range(len(targets)):
        if targets[i] == 0:
            node_layer_pair.append((targets[i], layer_names[0]))
        else:
            node_layer_pair.append((targets[i], layer_names[targets[i]//(tot_nodes//len(layer_names))]))

    return node_layer_pair


def analyse_attacked_network(net, tot_nodes, supporting_nodes, cas_data, step):
    # fin_intra_edge, fin_inter_edge, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_far_node, fin_clust, fin_mean_deg, fin_larg_comp, deg_assort, dist_deg_cent, dist_bet_cent, step.....
    layer_names = net.get_layers()
    layer_names = sorted(list(layer_names))

    stats = { "intra_edge":[],                              # Number of intra edges
              "components":[],                              # Components of the graph in each layers
              "largest component":[],                       # The largest component of the graphs
              "size of largest component":[],               # The size of the largest component
            }

    index = 0
    cur_layer = 0
    supp_edge_num = 0
    total_alive = 0
    total_unsupp = 0
    total_isol = 0
    for layer in layer_names:
        intra_edges = []
        for edge in net.edges:
            if edge[2] == edge[3] == layer:
                intra_edges.append(edge[:2])
            else:
                if supporting_nodes[edge[1]] == edge[0]:
                    supp_edge_num += 1

        G = nx.Graph()
        G.add_edges_from(intra_edges)

        components = list(nx.connected_components(G))

        alive_nodes = []
        for cur_state in state[cur_layer*tot_nodes//len(layer_names) : (cur_layer + 1)*tot_nodes//len(layer_names)]:
            if cur_state[0] == 'Alive':
                alive_nodes.append(index)
                total_alive += 1
            elif cur_state[0] == 'Dead from isolation':
                total_isol += 1
            elif cur_state[0] == 'Dead from unsupported':
                total_unsupp += 1
            index += 1

        stats["intra_edge"].append(len(intra_edges))
        stats["components"].append(components)
        if len(components) == 0:
            stats["largest component"].append([])
            stats["size of largest component"].append(0)
        else:
            stats["largest component"].append(max(components, key=len))
            stats["size of largest component"].append(len(max(components, key=len)))

        cur_layer += 1

    # fin_intra_edge_a, fin_intra_edge_b, fin_supp_edge, alive_nodes, tot_isol_node, tot_unsupp_node, cas_steps, fin_larg_comp_a, fin_larg_comp_b, step..
    cas_data.append(stats["intra_edge"][0])
    cas_data.append(stats["intra_edge"][1])
    cas_data.append(supp_edge_num / 2)
    cas_data.append(total_alive)
    cas_data.append(total_isol)
    cas_data.append(total_unsupp)
    cas_data.append(step)
    cas_data.append(stats["size of largest component"][0])
    cas_data.append(stats["size of largest component"][1])
    cas_data.append(isolated)
    cas_data.append(not_supported)

    return cas_data


def draw_network(net, tot_nodes, step):
    rep = 1
    fig = draw(net, nodeLabelRule={}, layerLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5)
    fig.savefig("-{:02d}-{:03d}.png".format(rep, step), dpi=300)
    # fig.savefig("Cascade attack step %d.pdf" % step)


def stitch():
    rep = 1

    fp_in = "-{:02d}-*.png".format(rep)
    fp_out = "-{:02d}.gif".format(rep)
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=500, loop=0)


def attack_network(net, tot_nodes, supporting_nodes, cas_data, attack_size):
    attack_layer_list = sorted(list(net.get_layers()))
    # Using try-except method, check the errors when the parameters have wrong value.
    try:
        del state[:]
        del largest_components[:]
        del isolated[:]
        del not_supported[:]
        del far_dead_node[:]

        if attack_size > (tot_nodes // len(net.get_layers())) or (attack_size <= 0):
            raise Exception("Wrong attack_size")

        initialise_state(tot_nodes)

        # With the different attack_type, sample the targets differently.
        targets = random.sample(list(range(tot_nodes // len(attack_layer_list))), attack_size)

        attack_list = make_node_layer_pair(net, tot_nodes, targets)

        step = 0
        net = initial_attack(net, attack_list)

        killed = targets

        step += 1

        # repeat the kill_nodes until the updated attack_list is empty.
        while len(killed) != 0:
            net, killed = cascade_attack(net, tot_nodes, supporting_nodes, step)

            if len(killed) != 0:
                # analyse_attacked_network(net, tot_nodes, step, len(killed))
                # draw_network(net, tot_nodes, step)
                # print("Step %d is finished." % step)
                step += 1

        cas_data = analyse_attacked_network(net, tot_nodes, supporting_nodes, cas_data, step - 1)

        print("Number of steps: ", step - 1)
        # stitch()

        return net, cas_data
    except Exception as e:
        print("Error is occurred", e)