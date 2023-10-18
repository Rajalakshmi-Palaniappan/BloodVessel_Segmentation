import os
import sys
import argparse
from glob import glob
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm, vonmises
import matplotlib.pyplot as plt
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


def add_constraints():
    pass


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
    print(root, cradius)

    objective = None
    vars = {}
    # iterate over all edges for unary term
    for u, v in graph.edges():
        print(u, v)
        cedgename = "e_%i_%i" % (u, v)
        print(cedgename)
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

    # add objective
    m.setObjective(objective, GRB.MAXIMIZE)

    m.optimize()
    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))

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
