import os
import sys
import argparse
from glob import glob
import utils
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm


def get_width_consistency_cost(x, mean, std):
    return norm.cdf(x, mean, std)


def get_edge_direction_similarity():
    pass


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
    # iterate over all edges
    for u, v in graph.edges():
        print(u, v)
        cedgename = "e_%i_%i" % (u, v)
        print(cedgename)
        # define gurobi variables, add all edges
        cedge = m.addVar(vtype=GRB.BINARY, name=cedgename)
        # compute edge cost and add to graph
        # add width difference and cost as edge attribute
        width_diff = graph.nodes[u]["radius"] - graph.nodes[v]["radius"]
        graph[u][v]["width_diff"] = width_diff
        width_cost = get_width_consistency_cost(width_diff, 1, 4)
        graph[u][v]["width_cost"] = width_cost
        # add cost for edge raw intensity here?
        # add unary term to objective
        if objective is None:
            objective = gp.LinExpr(width_cost * cedge)
        else:
            objective.add(width_cost * cedge)

    # add all edge pairs

    # compute pairwise costs

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
