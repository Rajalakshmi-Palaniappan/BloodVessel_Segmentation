import os
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def read_swc_file(in_file, directed=True):
    if not directed:
        raise NotImplementedError
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


def read_roots_from_csv(in_file):
    df = pd.read_csv(in_file)
    roots = [np.array((x, y, z, r), dtype=float) for x, y, z, r in zip(
        df['x'], df['y'], df['z'], df['r'])]
    return np.array(roots)


def convert_df_to_coords(points):
    points = points.to_list()
    points = [np.array(p[1:-1].split(","), dtype=float) for p in points]
    return np.array(points)


def read_csv_file(in_file):
    data = pd.read_csv(in_file)
    nodes_a = convert_df_to_coords(data["node1"])
    nodes_b = convert_df_to_coords(data["node2"])
    data = np.stack([nodes_a, nodes_b])
    print(data.shape)
    return data


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


def create_graph_from_point_list(points, roots, min_radius_diff=None,
                                 max_radius_diff=None):
    # create directed graph
    graph = nx.DiGraph()
    virtual_root_index = 0
    pos_to_id = {}
    # create nodes
    unique_points = np.unique(np.concatenate(
        [points[0], points[1]], axis=0), axis=0)
    index = 1
    root_indices = []
    roots_rounded = roots.round(decimals=2)
    for point in unique_points:
        graph.add_node(
            index,
            pos=np.array([point[0], point[1], point[2]]),
            radius=point[3],
        )
        pos_to_id["%f_%f_%f_%f" % (
            point[0], point[1], point[2], point[3])] = index
        if np.any(np.all(point.round(decimals=2) == roots_rounded, axis=1)):
            root_indices.append(index)
        index += 1

    # all edges should be bidirectional, except roots have only outcoming edges
    for node_a, node_b in zip(points[0], points[1]):
        index_a = pos_to_id["%f_%f_%f_%f" % (
            node_a[0], node_a[1], node_a[2], node_a[3])]
        index_b = pos_to_id["%f_%f_%f_%f" % (
            node_b[0], node_b[1], node_b[2], node_b[3])]
        if index_a in root_indices:
            graph.add_edge(index_a, index_b)
        elif index_b in root_indices:
            graph.add_edge(index_b, index_a)
        else:
            radius_a = node_a[3]
            radius_b = node_b[3]
            if min_radius_diff is not None and max_radius_diff is not None:
                if min_radius_diff <= radius_a - radius_b <= max_radius_diff:
                    graph.add_edge(index_a, index_b)
                if min_radius_diff <= radius_b - radius_a <= max_radius_diff:
                    graph.add_edge(index_b, index_a)
            else:
                graph.add_edge(index_a, index_b)
                graph.add_edge(index_b, index_a)

    # only take largest connected component
    largest_cc = list(
        max(nx.connected_components(graph.to_undirected()), key=len))
    graph = graph.subgraph(largest_cc).copy()

    # create virtual root and connect to original roots
    # and check if roots are still contained in largest connected component
    graph.add_node(
        virtual_root_index,
        pos=np.array([0.0, 0.0, 0.0]),
        radius=0.0
    )
    c_root_indices = []
    for root in root_indices:
        if graph.has_node(root):
            graph.add_edge(virtual_root_index, root)
            c_root_indices.append(root)
    root_indices = c_root_indices

    print("created graph with %i nodes and %i edges" % (
        graph.number_of_nodes(), graph.number_of_edges()))
    print("root indices: ", root_indices)

    return graph, root_indices, virtual_root_index


def create_toy_subgraph(graph, roots, vroot, size):
    nodes = [vroot] + roots
    # take subgraph with given size for each root
    for r in roots:
        cnt = 1
        cnode = [r]
        while cnt < size:
            ncnode = []
            for cn in cnode:
                nn = list(graph.successors(cn))
                nn = list(np.array(nn)[np.isin(nn, nodes, invert=True)])
                nodes += nn
                cnt += len(nn)
                ncnode += nn
            cnode = ncnode

    # take path from one root to the other
    graph_without_root = graph.copy()
    graph_without_root.remove_node(vroot)
    for i in range(len(roots)):
        for j in range(i+1, len(roots)):
            path = nx.shortest_path(
                graph_without_root.to_undirected(),
                source=roots[i], target=roots[j])
            nodes += path
    nodes = np.unique(nodes)
    toy_graph = graph.subgraph(nodes)
    return toy_graph, roots, vroot


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


def plot_graph(graph, selected_edges):
    graph = graph.to_undirected()
    print("plot graph: ", graph.number_of_edges(), len(selected_edges))
    # create lookup for selected edges
    lookup = []
    for u, v in selected_edges:
        lookup.append("%i_%i" % (u, v))
        lookup.append("%i_%i" % (v, u))

    # extract edge coordinates
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in graph.to_undirected().edges():
        node1, node2 = edge
        if "%i_%i" % (node1, node2) in lookup:
            continue
        node1 = graph.nodes[node1]["pos"]
        node2 = graph.nodes[node2]["pos"]
        edge_x.extend([node1[0], node2[0], None])
        edge_y.extend([node1[1], node2[1], None])
        edge_z.extend([node1[2], node2[2], None])

    # Create trace for edges
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(
            color='white', width=2),
        hoverinfo='none'
    )

    # Extract additional_edge coordinates
    selected_edge_x = []
    selected_edge_y = []
    selected_edge_z = []
    for selected_edge in selected_edges:
        node1, node2 = selected_edge
        node1 = graph.nodes[node1]["pos"]
        node2 = graph.nodes[node2]["pos"]
        selected_edge_x.extend([node1[0], node2[0], None])
        selected_edge_y.extend([node1[1], node2[1], None])
        selected_edge_z.extend([node1[2], node2[2], None])

    # Create trace for edges
    selected_edge_trace = go.Scatter3d(
        x=selected_edge_x,
        y=selected_edge_y,
        z=selected_edge_z,
        mode='lines',
        line=dict(color='red', width=2),
        hoverinfo='none'
    )

    # Create figure
    fig = go.Figure(data=[selected_edge_trace, edge_trace]) # node_trace
    fig.update_layout(
        title='Graph Plot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False),
            bgcolor='black'
        )
    )
    fig.update_layout(template='plotly_dark',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)', )
    fig.show()
