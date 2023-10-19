import plotly
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_swc(file_path):
    nodes = []
    edges = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            columns = line.split()
            if len(columns) != 7:
                print(f"Error: line {i+1} in file {file_path} has {len(columns)} columns instead of 7")
                continue
            node_id = int(columns[0])
            node_type = int(columns[1])
            x = float(columns[2])
            y = float(columns[3])
            z = float(columns[4])
            radius = float(columns[5])
            parent = int(columns[6])
            nodes.append((node_id, x, y, z, radius, parent))
            if parent != -1:
                edges.append((node_id, parent))
            
    return nodes, edges

#swc_file_path = '/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/geometric_priors/width_consistency/skeletons/skeleton_2.swc'
swc_file_path = '/home/maisl/data/micro_ct_vesselseg/skeleton_2_test_cycle.swc'
nodes, edges = read_swc(swc_file_path)


fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# add nodes 
x_coords = [node[1] for node in nodes]
y_coords = [node[2] for node in nodes]
z_coords = [node[3] for node in nodes]


node_trace = go.Scatter3d(
    x=x_coords,
    y=y_coords,
    z=z_coords,
    mode='markers',
    marker=dict(size=2, color='blue')
)


fig.add_trace(node_trace)

# to add edges between the nodes
edge_x = []
edge_y = []
edge_z = []

for edge in edges:
    x0, y0, z0 = x_coords[edge[0] - 1], y_coords[edge[0] - 1], z_coords[edge[0] - 1]
    x1, y1, z1 = x_coords[edge[1] - 1], y_coords[edge[1] - 1], z_coords[edge[1] - 1]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])


edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode='lines',
    line=dict(color='red', width=1)
)


fig.add_trace(edge_trace)


fig.update_layout(
    scene=dict(
        aspectmode='cube',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    )
)


fig.show()
