import os
import networkx as nx
import numpy as np


def read_swc_file(in_file):
    skeleton = []
    with open(os.path.join(in_file), "r") as f:
        #cnt = 0
        for line in f:
            #if cnt > 2000:
            #    break
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            if len(fields) < 7:
                continue
            index = int(fields[0])
            #type_id = int(fields[1])
            x = int(round(float(fields[2])))
            y = int(round(float(fields[3])))
            z = int(round(float(fields[4])))
            radius = float(fields[5])
            parent = int(fields[6])
            skeleton.append((index, x, y, z, radius, parent))
            #cnt += 1
    return skeleton


def create_graph_from_swc(swc):
    # create graph
    graph = nx.DiGraph()
    # create graph node for each swc line
    for point in swc:
        node_id = point[0]
        parent_id = point[5]
        graph.add_node(
            node_id,
            x=point[1],
            y=point[2],
            z=point[3],
            pos=np.array([point[1], point[2], point[3]]),
            radius=point[4],
            parent_id=parent_id
        )
        if parent_id != -1:
            graph.add_edge(node_id, parent_id)
    return graph


def create_graph_from_edge_list(edges):
    # create graph
    graph = nx.DiGraph()
    # create graph node for each swc line
    for u, v in edges:
        graph.add_node(u)
        graph.add_node(v)
        graph.add_edge(u, v)
    return graph


def get_n_degree_nodes(graph, degree):
    nodes = []
    for node_id in graph.nodes():
        if nx.degree(graph, node_id) == degree:
            nodes.append(node_id)
    return nodes


def get_ge_n_degree_nodes(graph, degree):
    nodes = []
    for node_id in graph.nodes():
        if nx.degree(graph, node_id) >= degree:
            nodes.append(node_id)
    return nodes
