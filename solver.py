import argparse
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm, vonmises
import matplotlib.pyplot as plt
import networkx as nx
from time import time
import utils
import pandas as pd
import multiprocessing
from multiprocessing import pool


def get_width_consistency_cost(x, mean=0, std=1):
    return norm.pdf(x, loc=mean, scale=std)


def get_edge_direction_similarity(u, v, w):
    e1 = u - v
    e2 = v - w
    angle = np.arccos(np.clip(
        np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)),
        -1,
        1))
    return angle


def get_edge_direction_cost(angle, mean=0, kappa=4):
    rad = angle * np.pi / 180
    cost = vonmises.pdf(mean, kappa, 0)
    # check plot

    return cost


def plot_edge_direction_cost(mean=0, kappa=4):
    x = np.linspace(-400, 400, 8000, endpoint=True)
    x = x * np.pi / 180
    fig = plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection='polar')
    vonmises_pdf = vonmises.pdf(mean, kappa, x)

    left.plot(x, vonmises_pdf)
    left.set_title("Cartesian plot")
    left.grid(True)

    right.plot(x, vonmises_pdf, label="PDF")
    right.set_title("Polar plot")
    right.legend(bbox_to_anchor=(0.15, 1.06))
    plt.show()


# def get_unary_cost():
#     pass
#
#
# def get_pairwise_cost():
#     pass


# def conservation_of_flow(nodes, ):

def add_constraints(m, graph, vars, root, multiple_roots=[], num_workers=1):
    # add no-cycle constraint
    # reimplemented from Tueretken 2016 formulas (8-15),
    # https://vcg.seas.harvard.edu/publications/reconstructing-curvilinear-networks-using-path-classifiers-and-integer-programming/paper

    if type(multiple_roots) == int:
        multiple_roots = [multiple_roots]
    # pool = multiprocessing.Pool(num_workers)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_vars = num_nodes * num_edges
    #auxiliary_vars = np.empty(num_vars, dtype=object)
    nodes_array = np.array(list(graph.nodes))
    #print(nodes_array)
    edges_i = []
    edges_j = []
    for i, j in graph.edges():
        edges_i.append(i)
        edges_j.append(j)
    edges_i_array = np.array(edges_i)
    edges_j_array = np.array(edges_j)
    #print(edges_i_array)
    #auxiliary_vars = np.array([f"y_{i}_{j[0]}_{j[1]}" for i in nodes_array for j in zip(edges_i_array, edges_j_array)],
    #                          dtype='U30')

    #variable_names = [f"y_{i}_{j[0]}_{j[1]}" for i in nodes_array for j in zip(edges_i_array, edges_j_array)]
    #print("variable names generated")
    m.addMVar(num_vars, vtype=GRB.CONTINUOUS, name=[f"y_{i}_{j[0]}_{j[1]}" for i in nodes_array for j in zip(edges_i_array, edges_j_array)])

    print("auxiliary variables array is created")

    #m.addMVar(513200693, vtype=GRB.CONTINUOUS, name=[i for i in auxiliary_vars])
    #auxiliary_vars = np.unique(auxiliary_vars)
    #np.set_printoptions(threshold=np.inf)
    #print(auxiliary_vars)
    # element_to_check = 'y_6146_0_192'
    # if element_to_check in auxiliary_vars:
    #     print(f"{element_to_check} is in the array")
    # else:
    #     print(f"{element_to_check} is not in the array")

    # auxiliary_vars = auxiliary_vars.reshape(num_nodes, num_edges)
    # print(auxiliary_vars.shape)

    # for i in range(auxiliary_vars.shape[0]):
    #     for j in range(auxiliary_vars.shape[1]):
    #         var_name = f'{auxiliary_vars[i, j]}'
    #         m.addVar(vtype=GRB.CONTINUOUS, name=var_name)
    #added_var_names = []
    # for i in auxiliary_vars:
    #     #var_name = i
    #     try:
    #        m.addVar(vtype=GRB.BINARY, name=i)
    #        #added_var_names.append(var_name)
    #        #print(f"Added variable '{var_name}': Name={new_var.VarName}, Type={new_var.VType}")
    #     except gp.GurobiError as e:
    #         print(f"Error adding variable '{i}': {e}")
    # print(len(added_var_names))
    # print(added_var_names[5])
    print("variables added to the model")
    m.update()
    #del variable_names
    #print("variable array deleted")
    # cauxvar_index = np.where(auxiliary_vars == 'y_6146_0_192')[0]
    # print(cauxvar_index)
    # print(auxiliary_vars[cauxvar_index][0])

    # var_names = [var.VarName for var in m.getVars()]
    # print(len(var_names))
    #var = m.getVarByName('y_192_13934_13878')
    # Loop through variables and find the matching one
    # for var in m.getVars():
    #     if var.VarName == 'y_192_13934_13878':
    #         print("Found variable:", var)
    #         break

            #print(var_name)

    # # (8) for all nodes j and l without root: sum_j y^l_rj <= 1
    print("add c8")
    start = time()
    for n_l in graph.nodes():
        if n_l == root:
            continue
        tmpexp = None
        for n_j in graph.successors(root):
            if n_j == root or n_j == n_l:
                continue
            cauxvar = 'y_%i_%i_%i' % (n_l, root, n_j)
            #cauxvar_index = 0
            # if auxiliary_vars[cauxvar_index] not in auxiliary_vars:
            #     auxiliary_vars[cauxvar_index] = m.addVar(
            #         vtype=GRB.CONTINUOUS, name=cauxvar)
            #print(cauxvar)
            #cauxvar_index = np.where(auxiliary_vars == cauxvar)[0]
            #var = m.getVarByName(cauxvar)
            # auxiliary_vars[cauxvar] = m.addVar(
            #     vtype=GRB.CONTINUOUS, name=cauxvar)
            if tmpexp is None:
                tmpexp = gp.LinExpr(m.getVarByName(cauxvar))
            else:
                tmpexp.add(m.getVarByName(cauxvar))
        m.addConstr(tmpexp <= 1, "c8_%i" % n_l)
    stop = time()
    print("%s for c8" % (stop - start))


    # (9) for all nodes l without root and all nodes j without l:
    #sum_j y^l_jl <= 1
    print("add c9")
    start = time()
    for n_l in graph.nodes():
        if n_l == root:
            continue
        tmpexp = None
        for n_j in graph.predecessors(n_l):
            if n_j == n_l:
                continue
            cauxvar = "y_%i_%i_%i" % (n_l, n_j, n_l)
            # if cauxvar not in auxiliary_vars:
            #     auxiliary_vars[cauxvar] = m.addVar(
            #         vtype=GRB.CONTINUOUS, name=cauxvar)
            if tmpexp is None:
                tmpexp = gp.LinExpr(m.getVarByName(cauxvar))
            else:
                tmpexp.add(m.getVarByName(cauxvar))
        if tmpexp is not None:
            m.addConstr(tmpexp <= 1, "c9_%i" % n_l)
        else:
            print("c9 is None for ", n_l, list(graph.predecessors(n_l)))
    stop = time()
    print("%s for c9" % (stop - start))

    print("add c10")
    start = time()
    for n_l in graph.nodes():
        if n_l == root:
            continue
        for n_i in graph.nodes():
            if n_i == root or n_i == n_l:
                continue
            tmpexpleft = None
            tmpexpright = None

            for n_j in graph.successors(n_i):
                if n_j == n_i or n_j == root:
                    continue
                cauxvar = "y_%i_%i_%i" % (n_l, n_i, n_j)
                # if cauxvar not in auxiliary_vars:
                #     auxiliary_vars[cauxvar] = m.addVar(
                #         vtype=GRB.CONTINUOUS, name=cauxvar)
                if tmpexpleft is None:
                    tmpexpleft = gp.LinExpr(m.getVarByName(cauxvar))
                else:
                    tmpexpleft.add(m.getVarByName(cauxvar))

            for n_j in graph.predecessors(n_i):
                if n_j == n_i or n_j == n_l:
                    continue
                cauxvar = "y_%i_%i_%i" % (n_l, n_j, n_i)
                # if cauxvar not in auxiliary_vars:
                #     auxiliary_vars[cauxvar] = m.addVar(
                #         vtype=GRB.CONTINUOUS, name=cauxvar)
                if tmpexpright is None:
                    tmpexpright = gp.LinExpr(m.getVarByName(cauxvar))
                else:
                    tmpexpright.add(m.getVarByName(cauxvar))

            if tmpexpleft is not None and tmpexpright is not None:
                m.addConstr(tmpexpleft - tmpexpright == 0, "c10_%i_%i" % (n_l, n_i))
            else:
                print("c10 is None for ", n_l)
    stop = time()
    print("%s for c10" % (stop - start))


    #(11) for all edges and all nodes except r,i,j: y^l_ij <= t_ij
    print("add c11")
    start = time()
    for i, j in graph.edges():
        for n_l in graph.nodes():
            if n_l == root or n_l == i or n_l == j:
                continue
            cauxvar = "y_%i_%i_%i" % (n_l, i, j)
            # if cauxvar not in auxiliary_vars:
            #     auxiliary_vars[cauxvar] = m.addVar(
            #         vtype=GRB.CONTINUOUS, name=cauxvar)
            cedgename = "e_%i_%i" % (i, j)
            m.addConstr(m.getVarByName(cauxvar) <= vars[cedgename],
                        "c11_%i_%i_%i" % (n_l, i, j))
    stop = time()
    print("%s for c11" % (stop - start))

    #(11)
    # print("add c11")
    # start = time()


    # (12) for all edges: y^l_il == t_il
    print("add c12")
    start = time()
    for i, l in graph.edges():
        cauxvar = "y_%i_%i_%i" % (l, i, l)
        # if cauxvar not in auxiliary_vars:
        #     auxiliary_vars[cauxvar] = m.addVar(
        #         vtype=GRB.CONTINUOUS, name=cauxvar)
        cedgename = "e_%i_%i" % (i, l)
        m.addConstr(m.getVarByName(cauxvar) == vars[cedgename],
                    "c12_%i_%i" % (i, l))
    stop = time()
    print("%s for c12" % (stop - start))


    print("add c13")
    start = time()
    for i, j in graph.edges():
        for n_l in graph.nodes():
            if n_l == root or n_l == i:
                continue
            cauxvar = "y_%i_%i_%i" % (n_l, i, j)
            m.addConstr(m.getVarByName(cauxvar) >= 0,
                        "c13_%i_%i_%i" % (n_l, i, j))
    stop = time()
    print("%s for c13" % (stop - start))


    # (14) already done in variable type definition
    # (15) set edge from virtual root to real root to 1
    print("add c15")
    start = time()
    if len(multiple_roots) > 0:
        for r in multiple_roots:
            cedgename = "e_%i_%i" % (root, r)
            if cedgename not in vars:
               vars[cedgename] = m.addVar(
                   vtype=GRB.BINARY, name=cedgename)
            m.addConstr(vars[cedgename] == 1, "c15_%i_%i" % (root, r))
    stop = time()
    print("%s for c15" % (stop - start))

    # each edge can only be selected in one direction
    # x[i, j] + x[j, i] <= 1
    print("add c16")
    start = time()
    added = []
    for i in graph.nodes():
        if i == 0:
            continue
        nnodes = list(graph.successors(i))
        for nn in nnodes:
            if nn in added:
                continue
            cedgename = "e_%i_%i" % (i, nn)
            nedgename = "e_%i_%i" % (nn, i)
            if cedgename in vars and nedgename in vars:
                m.addConstr(vars[cedgename] + vars[nedgename] <= 1,
                            "c16_%i_%i" % (i, nn))
        added.append(i)
    stop = time()
    print("%s for c16" % (stop - start))

    return m


def create_model(graph, root_indices=[], virtual_root_index=None,
                 num_workers=1):
    # define model

    m = gp.Model("OptimalTrees")
    if virtual_root_index is not None:
        root = virtual_root_index
    elif len(root_indices) == 1:
        root = root_indices[0]
    elif len(root_indices) > 1 and virtual_root_index is None:
        raise ValueError('Please specify virtual root if more than one root')
    else:
        # if no roots are given, taking start node with the highest radius
        #endpoints = utils.get_n_degree_nodes(graph, 1)
        # branching_points = utils.get_ge_n_degree_nodes(graph, 3)
        root = None
        cradius = 0
        for p in graph.nodes():
            if graph.nodes[p]["radius"] >= cradius:
                root = p
                cradius = graph.nodes[p]["radius"]
    print("root index: ", root)

    objective = None
    vars = {}
    # iterate over all edges for unary term
    print("add unary terms for edges")
    for u, v in graph.edges():
        cedgename = "e_%i_%i" % (u, v)
        # define gurobi variables, add all edges
        vars[cedgename] = m.addVar(vtype=GRB.BINARY, name=cedgename)
        # compute edge cost and add to graph
        # add width difference and cost as edge attribute
        if u == root:
            width_diff = 0
            width_cost = 0
        else:
            width_diff = graph.nodes[u]["radius"] - graph.nodes[v]["radius"]
            width_cost = get_width_consistency_cost(width_diff, 0, 4)

        graph[u][v]["width_diff"] = width_diff
        graph[u][v]["width_cost"] = width_cost
        # add cost for edge raw intensity here?
        # add unary term to objective
        if objective is None:
            # objective = gp.LinExpr(width_cost * vars[cedgename])
            objective = gp.QuadExpr(width_cost * vars[cedgename])
        else:
            objective.add(width_cost * vars[cedgename])

    # iterate over all edges for pairwise term
    print("add pairwise terms for edge pairs")
    for u, v in graph.edges():
        # check if edge has successor edges
        successor_nodes = graph.successors(v)
        if len(list(successor_nodes)) == 0:
            continue
        cedgename = "e_%i_%i" % (u, v)
        for n in successor_nodes:
            nedgename = "e_%i_%i" % (v, n)
            if u == root:
                edge_dir = 0
                edge_dir_cost = 0
            else:
                edge_dir = get_edge_direction_similarity(
                    graph.nodes[u]["pos"],
                    graph.nodes[v]["pos"],
                    graph.nodes[n]["pos"]
                )
                edge_dir_cost = get_edge_direction_cost(edge_dir)
            print(edge_dir, edge_dir_cost)
            # plot_edge_direction_cost()
            # add edge pair cost to objective
            objective.add(edge_dir_cost * vars[cedgename] * vars[nedgename])
    m.Params.NodeFileStart = 0.05
    #m.setParam('Method', 2)
    m.setParam('BarHomogeneous', 1)

    # add constraints
    m = add_constraints(
        m, graph, vars, root, multiple_roots=root_indices,
        num_workers=num_workers)
    # add objective
    m.setObjective(objective, GRB.MAXIMIZE)

    return m


def run_solver(graph, root_indices, virtual_root_index, num_workers=1):
    # check for loops
    try:
        print("loop found", list(nx.find_cycle(graph, orientation="ignore")))
    except nx.exception.NetworkXNoCycle as e:
        print("no loops found.")
    # check how many connected components
    print("number connected components: ", len(list(nx.connected_components(
        graph.to_undirected()))))

    # create model
    m = create_model(graph, root_indices, virtual_root_index, num_workers)

    # solve
    m.optimize()

    # read result
    cnt = 0
    selected_edges = []
    for v in m.getVars():
        if v.X == 1 and v.VarName.startswith("e_"):
            # print('%s %g' % (v.VarName, v.X))
            selected_edges.append(np.array(v.VarName.split("_")[1:], dtype=int))
            cnt += 1
    print(graph.number_of_edges(), cnt)

    # create graph and check for loops
    solution_graph = utils.create_graph_from_edge_list(selected_edges)

    # node_to_remove = virtual_root_index
    # solution_graph = solution_graph.remove_node(node_to_remove)
    # verify if graph contains loops
    try:
        print("loop found",
              list(nx.find_cycle(solution_graph, orientation="ignore")))
    except nx.exception.NetworkXNoCycle as e:
        print("no loops found.")

    node_to_remove = virtual_root_index
    solution_graph.remove_node(node_to_remove)

    output_edges = []
    for edge in solution_graph.edges():
        node1, node2 = edge
        node1 = graph.nodes[node1]["pos"]
        node2 = graph.nodes[node2]["pos"]
        edge = ((node1[0], node1[1], node1[2]), (node2[0], node2[1], node2[2]))
        output_edges.append(edge)
    df = pd.DataFrame(output_edges, columns=['node1', 'node2'])
    #df.to_csv('/mnt/md0/rajalakshmi/MST_ILP/ILP/skeleton_1_toygraph_LP_output.csv', index=False)


    print("number of connected components: ", len(list(nx.connected_components(
        solution_graph.to_undirected()))))

    print(len(selected_edges))

    utils.plot_graph(graph, selected_edges)


def main():
    # get input parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="input folder or file in csv or swc file format")
    parser.add_argument("--roots", type=str,
                        default="roots.csv",
                        help="input json file with root coordinates"
                        )
    parser.add_argument("--min_radius_diff", type=int,
                        default=None,
                        help="minimum radius difference between two points "
                             "such that edge is constructed"
                        )
    parser.add_argument("--max_radius_diff", type=int,
                        default=None,
                        help="maximum radius difference between two points "
                             "such that edge is constructed"
                        )
    parser.add_argument("--num_workers", type=int,
                        default=10,
                        help="number of workers to construct constraints"
                        )
    args = parser.parse_args()

    # read input
    # assume overcomplete graph here
    if args.input.endswith(".swc"):
        # todo: clean up here
        points = utils.read_swc_file(args.input, directed=False)
    elif args.input.endswith(".csv"):
        points = utils.read_csv_file(args.input)
    else:
        raise NotImplementedError

    start = time()
    # read roots
    roots = utils.read_roots_from_csv(args.roots)
    # create nx graph
    graph, root_indices, virtual_root_index = (
        utils.create_graph_from_point_list(
            points, roots, args.min_radius_diff, args.max_radius_diff))
    # create smaller toy subgraph for developing
    graph, root_indices, virtual_root_index = utils.create_toy_subgraph(
        graph, root_indices, virtual_root_index, 500)
    print(graph.number_of_nodes(), graph.number_of_edges())
    print("time for creating graph % s sec" % (time() - start))

    start = time()
    run_solver(graph, root_indices, virtual_root_index, args.num_workers)
    print("time for solving optimization problem % s sec" % (time() - start))


if __name__ == "__main__":
    main()
