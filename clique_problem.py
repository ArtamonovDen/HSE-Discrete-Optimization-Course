import argparse
import logging
import io
import math
import networkx as nx
import numpy as np
import operator
import random
import time

from docplex.mp.model import Model
from itertools import combinations, compress
from collections import defaultdict


best_sol = 0
best_x = []

EPS = 1e-09

def get_model_info(model):
    '''
        Return short model info
    '''
    output = io.StringIO()
    output.write(f'Model: {model.name}\n')
    output.write(model.get_statistics().to_string())
    model_info = output.getvalue()
    output.close()
    return model_info

def is_integer(x):
    '''
    Check if element is integer with Eps
    '''
    return math.isclose(x, np.round(x),rel_tol=EPS)

def is_int_list(l):
    '''
        Check if each elemnt in list is integer with Eps
    '''
    return all(map(is_integer,l))


def branching(vars_list):
    '''
        Return index of x to start branching with.
        The strategy: random select index among not integers vars
    '''
    int_mask = list(map(is_integer, vars_list))
    int_vars = {index:x for index, x in enumerate(vars_list) if not is_integer(x)}
    if not int_vars:
        raise Exception('All vars are integers')

    return random.choice([*int_vars]) # choice random index


def bnb(model):
    '''
    Main BnB function. Called recurcively
    '''
    global best_sol
    global best_x

    local_sol = model.solve()

    if not local_sol: # No solution found
        return

    x = local_sol.get_all_values()
    ub = local_sol.get_objective_value()

    if (best_sol - ub > EPS):
        return 
    
    if (abs(ub-best_sol) < EPS or ub < best_sol):
        return

    if (is_int_list(x)):
        best_sol = math.floor(ub)
        best_x = list(map(round,x))
        logging.info(f'Updated solution: {best_sol}, {best_x}\n')
        return
    
    i = branching(x)
    branching_var = model.get_var_by_index(i)

    logging.debug(f'Branching var is {branching_var}. Branching by value {x[i]}')
    left_int_constraint = model.add_constraint(
            branching_var <= math.floor(x[i])
    )
    logging.debug(f'Added LEFT constraint: {left_int_constraint}')
    bnb(model)
    model.remove_constraint(left_int_constraint)
    logging.debug(f'Remove LEFT constraint: {left_int_constraint}')

    right_int_constraint = model.add_constraint(
            branching_var >= math.ceil(x[i])
    )
    logging.debug(f'Added RIGHT constraint: {right_int_constraint}')
    bnb(model)
    model.remove_constraint(right_int_constraint)
    logging.debug(f'Remove RIGHT constraint: {right_int_constraint}')




def check_painting_correct(nodes_by_colors):
    '''
        Check the vertex to be painted with a single color
    '''
    check_list=[]
    for color,nodes_list in nodes_by_colors.items():
        check_list.extend(nodes_list)
    assert len(check_list) == len(set(check_list))   


def apply_coloring_constraints(model, G, painting_steps=10):
    '''
       Apply coloring to graph several times and add constraints to model.
       Constraints: if nodes have same colors they can not be included in the same clique
    '''

    logging.info(f'Applying Coloring constraints {painting_steps} times')
    
    for step in range(painting_steps):

        var = model.get_var_by_name # get variable by name function

        # Apply coloring algorithm to graph 
        colors_by_nodes = nx.algorithms.coloring.greedy_color(G,'random_sequential')
        
        nodes_by_colors = defaultdict(list)
        for node, color in colors_by_nodes.items():
            nodes_by_colors[color].append(node)
            
        check_painting_correct(nodes_by_colors)
        
        # Add painting constarints to the model
        for color, nodes_list in nodes_by_colors.items():
            for c_num in range(2, len(nodes_list)+1): # Add different constraints like x1+x2<=1 or x1+x2+x3<=1 etc.
                model.add_constraints(
                    [model.le_constraint( sum(var(f'x_{i}') for i in combo ), 1) for combo in combinations(nodes_list,c_num)]
                )


def is_clique(C,G):
    '''
      Test that C is a real clique
    '''
    for v in C:
        C_v = C.copy()
        C_v.remove(v)
        assert all(elem in G.neighbors(v) for elem in  C_v)

def init_solution(C, G):
    '''
      Init solution by found heuristic
    '''
    global best_sol
    global best_x
    best_sol = len(C)
    best_x = {k:0 for k in sorted(G.nodes)}
    for c in C:
        best_x[c] = 1
    best_x = best_x.values()


def clique_heuristic(G):
    '''
      Init solution by heuristic for maximum clique problem
    '''
    logging.info('Applying Clique heuristic')

    C_best=set()
    for v in G.nodes:
        C = set()
        C.add(v)
        M = {n:G.degree(n) for n in G.neighbors(v)}
        while True:
            local_Clique = []
            w = max(M.items(), key=operator.itemgetter(1))[0] # get node with max degree
            C.add(w)
            N_w =  {n:G.degree(n) for n in G.neighbors(w)}
            M = {k:M[k] for k in set(M).intersection(set(N_w))}
            if not M:
                break

        is_clique(C,G)
        if len(C) > len(C_best):
            C_best = C
    return C_best

def init_graph(problem):
    '''
      Read graph from file
    '''
    # Read graph from file
    logging.info('Initializing Graph')
    with open(problem, 'r') as f:
        edges = [
            tuple(map(int,line.split()[1:3])) for line in f if line.startswith('e')
        ]

    G = nx.Graph()
    G.add_edges_from(edges)

    return G


def init_model(problem, G):
    '''
      Init LP model and add basic clique constraints
    '''

    logging.info('Initializing Model')
    nodes = sorted(G.nodes)

    # Init LP Model
    model = Model(name=f'Max Clique-{problem}')
    x = {i : model.continuous_var(name= f'x_{i}') for i in nodes}

    # Basic constrains
    # model.add_constraints([x[i] <=1 for i in nodes])
    model.add_constraints([(x[i]+x[j]) <= 1 for i,j in combinations(nodes,2) if not G.has_edge(i,j)])

    model.maximize(model.sum(x))

    return model


def main(args):

    global best_sol 
    global best_x

    problem_file = args.problem # 'cliques_problems/san200_0.7_1.clq'  
    problem_name = problem_file.split('/')[1]

    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%I:%M:%S %p', 
        filename=f'reports/{problem_name}.report', level=logging.INFO)
    logging.info(f'Problem: {problem_name}')

    G = init_graph(problem_file)
    model = init_model(problem_name,G)
    apply_coloring_constraints(model, G, painting_steps=args.color_step)
    C = clique_heuristic(G)
    init_solution(C,G)

    logging.info(f'Init solution is {best_sol}: {best_x}')
    logging.info(f'Model Information: {get_model_info(model)}')
    logging.info('Starting Branch and Bounds...')
    start = time.time()
    bnb(model)
    end = time.time()
    logging.info('Branch and Bounds have finished')
    logging.info(f'BnB time is {(end-start)/60} minutes')

    best_clique = [i for i,j in enumerate(best_x) if j > 0]
    logging.info(f'Best solution is {best_sol}: {best_clique}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BnB Solver for Maximum Clique Problem')
    parser.add_argument('-p', '--problem', dest="problem")
    parser.add_argument('-c', '--color-step', dest="color_step", default=10, type=int)
    args = parser.parse_args()
    main(args)