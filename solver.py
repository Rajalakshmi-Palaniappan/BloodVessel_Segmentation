import os
import sys
import argparse
from glob import glob
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm, vonmises
import matplotlib.pyplot as plt
import networkx as nx
import utils


def get_width_consistency_cost(x, mean, std):
    return norm.cdf(x, mean, std)


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


def get_unary_cost():
    pass


def get_pairwise_cost():
    pass


def add_constraints(m, graph, vars, root):
    # add no-cycle constraint
    # reimplemented from Tueretken 2016 formulas (8-15),
    # https://vcg.seas.harvard.edu/publications/reconstructing-curvilinear-networks-using-path-classifiers-and-integer-programming/paper

    auxiliary_vars = {}
    # (8) for all nodes j and l without root: sum_j y^l_rj <= 1
    print("add c8")
    for n_l in graph.nodes():
        if n_l == root:
            continue
        tmpexp = None
        for n_j in graph.nodes():
            if n_j == root or n_j == n_l:
                continue
            cauxvar = "y_%i_%i_%i" % (n_l, root, n_j)
            auxiliary_vars[cauxvar] = m.addVar(
                vtype=GRB.CONTINUOUS, name=cauxvar)
            if tmpexp is None:
                tmpexp = gp.LinExpr(auxiliary_vars[cauxvar])
            else:
                tmpexp.add(auxiliary_vars[cauxvar])
        m.addConstr(tmpexp <= 1, "c8_%i" % n_l)

    # (9) for all nodes l without root and all nodes j without l:
    # sum_j y^l_jl <= 1
    print("add c9")
    for n_l in graph.nodes():
        if n_l == root:
            continue
        tmpexp = None
        for n_j in graph.nodes():
            if n_j == n_l:
                continue
            cauxvar = "y_%i_%i_%i" % (n_l, n_j, n_l)
            auxiliary_vars[cauxvar] = m.addVar(
                vtype=GRB.CONTINUOUS, name=cauxvar)
            if tmpexp is None:
                tmpexp = gp.LinExpr(auxiliary_vars[cauxvar])
            else:
                tmpexp.add(auxiliary_vars[cauxvar])
        m.addConstr(tmpexp <= 1, "c9_%i" % n_l)

    # (10) for all l
    print("add c10")
    for n_l in graph.nodes():
        if n_l == root:
            continue
        for n_i in graph.nodes():
            if n_i == root or n_i == n_l:
                continue
            # left sum
            tmpexpleft = None
            for n_j in graph.nodes():
                if n_j == root or n_j == n_i:
                    continue
                cauxvar = "y_%i_%i_%i" % (n_l, n_i, n_j)
                if cauxvar not in auxiliary_vars:
                    auxiliary_vars[cauxvar] = m.addVar(
                        vtype=GRB.CONTINUOUS, name=cauxvar)
                if tmpexpleft is None:
                    tmpexpleft = gp.LinExpr(auxiliary_vars[cauxvar])
                else:
                    tmpexpleft.add(auxiliary_vars[cauxvar])
            # right sum
            tmpexpright = None
            for n_j in graph.nodes():
                if n_j == n_i or n_j == n_l:
                    continue
                cauxvar = "y_%i_%i_%i" % (n_l, n_j, n_i)
                if cauxvar not in auxiliary_vars:
                    auxiliary_vars[cauxvar] = m.addVar(
                        vtype=GRB.CONTINUOUS, name=cauxvar)
                if tmpexpright is None:
                    tmpexpright = gp.LinExpr(auxiliary_vars[cauxvar])
                else:
                    tmpexpright.add(auxiliary_vars[cauxvar])
            # bring them together
            m.addConstr(tmpexpleft - tmpexpright == 0,
            "c10_%i_%i" % (n_l, n_i))

    # (11) for all edges and all nodes except r,i,j: y^l_ij <= t_ij
    print("add c11")
    for i, j in graph.edges():
        for n_l in graph.nodes():
            if n_l == root or n_l == i or n_l == j:
                continue
            cauxvar = "y_%i_%i_%i" % (n_l, i, j)
            cedgename = "e_%i_%i" % (i, j)
            m.addConstr(auxiliary_vars[cauxvar] <= vars[cedgename],
                        "c11_%i_%i_%i" % (i, j, n_l))

    # (12) for all edges: y^l_il == t_il
    print("add c12")
    for i, l in graph.edges():
        cauxvar = "y_%i_%i_%i" % (l, i, l)
        cedgename = "e_%i_%i" % (i, l)
        m.addConstr(auxiliary_vars[cauxvar] == vars[cedgename],
                    "c12_%i_%i" % (i, l))

    # (13) for all edges e_ij and all nodes except root and i: y^l_ij >= 0
    print("add c13")
    for i, j in graph.edges():
        for n_l in graph.nodes():
            if n_l == root or n_l == i:
                continue
            cauxvar = "y_%i_%i_%i" % (n_l, i, j)
            m.addConstr(auxiliary_vars[cauxvar] >= 0,
                        "c13_%i_%i_%i" % (l, i, j))

    return m, auxiliary_vars


def create_model(graph, roots=None):
    # define model
    m = gp.Model("OptimalTrees")
    # heads up: taking start node with highest radius for now
    endpoints = utils.get_n_degree_nodes(graph, 1)
    #branching_points = utils.get_ge_n_degree_nodes(graph, 3)
    root = None
    cradius = 0
    for n in endpoints:
        if graph.nodes[n]["radius"] >= cradius:
            root = n
            cradius = graph.nodes[n]["radius"]
    print("root: ", root, cradius)

    # check for loops
    try:
        print("loop found", list(nx.find_cycle(graph, orientation="ignore")))
    except nx.exception.NetworkXNoCycle as e:
        print("no loops found.")

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
        width_diff = graph.nodes[u]["radius"] - graph.nodes[v]["radius"]
        graph[u][v]["width_diff"] = width_diff
        width_cost = get_width_consistency_cost(width_diff, 1, 4)
        graph[u][v]["width_cost"] = width_cost
        # add cost for edge raw intensity here?
        # add unary term to objective
        if objective is None:
            #objective = gp.LinExpr(width_cost * vars[cedgename])
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
        #cedge = m.getVarByName(cedgename)
        for n in successor_nodes:
            nedgename = "e_%i_%i" % (v, n)
            #nedge = m.getVarByName(nedgename)
            edge_dir = get_edge_direction_similarity( # maybe edge_pair_angle?
                graph.nodes[u]["pos"],
                graph.nodes[v]["pos"],
                graph.nodes[n]["pos"]
            )
            edge_dir_cost = get_edge_direction_cost(edge_dir)
            print(edge_dir, edge_dir_cost)
            # plot_edge_direction_cost()
            # add edge pair cost to objective
            objective.add(edge_dir_cost * vars[cedgename] * vars[nedgename])

    # add constraints
    m, aux_vars = add_constraints(m, graph, vars, root)

    # add objective
    m.setObjective(objective, GRB.MAXIMIZE)

    m.optimize()
    cnt = 0
    selected_edges = []
    for v in m.getVars():
        if v.X == 1 and v.VarName.startswith("e_"):
            print('%s %g' % (v.VarName, v.X))
            selected_edges.append(np.array(v.VarName.split("_")[1:], dtype=int))
            cnt += 1
    print(graph.number_of_edges(), cnt)

    # create graph and check for loops
    solution_graph = utils.create_graph_from_edge_list(selected_edges)
    try:
        print("loop found", list(nx.find_cycle(solution_graph, orientation="ignore")))
    except nx.exception.NetworkXNoCycle as e:
        print("no loops found.")

    return m


def main():
    # get input parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--swc_folder", type=str, default=None,
                        help="path to input swc folder")
    args = parser.parse_args()

    # read swc files
    # assume overcomplete graph here
    if args.swc_folder.endswith(".swc"):
        swc = utils.read_swc_file(args.swc_folder)
    else:
        raise NotImplementedError

    # create nx graph ?
    graph = utils.create_graph_from_swc(swc)

    # get roots -> heads up: for now just first -1 node
    # roots =
    # create model
    model = create_model(graph)

    # solve

    # read result


if __name__ == "__main__":
    main()
