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


best_sol = 0.0
best_x = []

EPS = 1e-09
EPS_OBJECTIVE_MOVE = 0.01
TAIL_OFF_BOUND = 10


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
    return math.isclose(x, round(x), rel_tol=EPS)


def is_int_list(l):
    '''
        Check if each elemnt in list is integer with Eps
    '''
    return all(map(is_integer, l))


def compare_with(x):
    '''
       Return nearest int if x is int. Prepare to compare with integers
    '''
    return round(x) if is_integer(x) else x

def separation(G, sol, model, constraint_num=5):
    '''
      Find the most violated constraint
      by creating Maximum Independent set using coloring.

      Return found constraint of None if |S| <= 1
    '''
    logging.debug('Start Separation')
    Cs = []  # Constraints list
    var = model.get_var_by_name   # get variable by name function

    # Apply coloring algorithm to graph
    colors_by_nodes = nx.algorithms.coloring.greedy_color(G, 'random_sequential')
    nodes_by_colors = defaultdict(list)
    weights_by_color = defaultdict(int)  # weights of each colored independent set

    for node, color in colors_by_nodes.items():
        nodes_by_colors[color].append(node)
        weights_by_color[color] += sol.get_value(var(f'x_{node}'))

    max_weighted_colors = sorted(weights_by_color, key=weights_by_color.get)[-constraint_num:]  # Get constraint_num of max weighted sets
    for color in reversed(max_weighted_colors):
        independent_set = nodes_by_colors[color]
        if compare_with(weights_by_color[color]) > 1:
            C = model.le_constraint(sum(var(f'x_{i}') for i in independent_set), 1)  # TODO return several max constraints? 
            Cs.append(C)
            logging.debug(f'Found Max violated constraint: {str(C)}. Weight is {weights_by_color[color]}')
        else:
            C = model.le_constraint(sum(var(f'x_{i}') for i in independent_set), 1)
            logging.debug(f'Found constraint {str(C)} has weight {weights_by_color[color]} which is less or equal to 1. Stop separation...')
            break

    logging.debug('Seration finished')
    return Cs


def branching(vars_list, strategy='max'):
    '''
        Return index of x to start branching with.
        The strategy to chose:
          - random select index among not integers vars
          - select max x
        If all vars are integer return -1
    '''
    non_int_vars = {index: x for index, x in enumerate(vars_list) if not is_integer(x)}
    if not non_int_vars:
        return -1
    if strategy == 'max':
        return max(non_int_vars, key=non_int_vars.get)
    if strategy == 'random':
        return random.choice([*non_int_vars])


def bnc_branch_left(model, G, branching_var, x):
    left_int_constraint = model.add_constraint(
            branching_var <= math.floor(x)
    )
    logging.debug(f'Added LEFT constraint: {left_int_constraint}')
    bnc(model, G)
    model.remove_constraint(left_int_constraint)
    logging.debug(f'Remove LEFT constraint: {left_int_constraint}')


def bnc_branch_right(model, G, branching_var, x):
    right_int_constraint = model.add_constraint(
            branching_var >= math.ceil(x)
    )
    logging.debug(f'Added RIGHT constraint: {right_int_constraint}')
    bnc(model, G)
    model.remove_constraint(right_int_constraint)
    logging.debug(f'Remove RIGHT constraint: {right_int_constraint}')


def is_best_solution(sol):
    '''
        Check if current solution is better than best INT solution
    '''
    global best_sol
    sol = round(sol) if is_integer(sol) else math.floor(sol)  # sol is int now
    return sol >= best_sol


def is_changed_enough(cur_val, last_val):
    '''
       Check if objective finction is changed enough
    '''

    return abs(cur_val - last_val) > EPS_OBJECTIVE_MOVE


def bnc(model, G):
    '''
        Main Branch-and-Cut function Called recurcively
    '''
    local_sol = model.solve()

    global best_sol
    global best_x

    if not local_sol:  # No solution found
        logging.debug('No solution found. Skip branch')
        return

    x = local_sol.get_all_values()
    obj = local_sol.get_objective_value()
    logging.debug(f'Found solution {obj}, {x}')

    if not is_best_solution(obj):
        logging.debug('Solution is not better. Skip branch')
        return

    tail_off_counter = 0
    last_obj = 0
    logging.debug('Start Separation loop')
    while True:

        if tail_off_counter > TAIL_OFF_BOUND:
            logging.debug('TAIL_OFF_BOUND is reached. Break loop')
            break

        Cs = separation(G, local_sol, model, constraint_num=100)

        if not Cs:
            logging.debug('Separation loop: no found constraint after separation. Break loop')
            break

        model.add_constraints(Cs)
        local_sol = model.solve()

        if not local_sol:
            logging.debug('Separation loop: no solution found. Skip branch')
            return

        x = local_sol.get_all_values()
        obj = local_sol.get_objective_value()
        logging.debug(f'Separation loop: found solution {obj}, {x}')

        if not is_best_solution(obj):
            logging.debug('Separation loop: solution is not better. Skip branch')
            return

        if is_changed_enough(obj, last_obj):
            last_obj = obj
            tail_off_counter = 0
            logging.debug('Objective function change is noticable. Resume counter')
        else:
            tail_off_counter += 1
            logging.debug(f'Objective function change is small. Attempt is increased to {tail_off_counter}')

    i = branching(x)
    if i < 0:
        possible_clique = list(np.where(np.array(x) > 0.5)[0] + 1)  # from index 0..n-1 to nodes 1..n
        logging.debug(f'Solution is integer. Possible clique is {possible_clique}')

        broken_constraint = check_solution(possible_clique, G)
        if not broken_constraint:
            logging.debug('It is a real clique. Update solution')
            best_sol = obj
            best_x = x
            return

        else:
            var = model.get_var_by_name
            c = model.le_constraint(
                var(f'x_{broken_constraint[0]}') + var(f'x_{broken_constraint[1]}'), 1
                )
            logging.debug(f'It is not a real clique. Add new constraint: {c}')
            model.add_constraint(c)
            bnc(model, G)
            logging.debug('Up back')
        return

    branching_var = model.get_var_by_index(i)  # TODO get by name? x_i+1 ?
    logging.debug(f'Branching var is {branching_var}. Branching by value {x[i]}')

    if round(x[i]):
        # choose left branch if x[0] is closer to 0 and right one otherwise
        bnc_branch_right(model, G, branching_var, x[i])
        bnc_branch_left(model, G, branching_var, x[i])
    else:
        bnc_branch_left(model, G, branching_var, x[i])
        bnc_branch_right(model, G, branching_var, x[i])


def check_solution(C, G):
    '''
      Check if C is a clique. Return one missing edge if it's not
      and None if it is
    '''
    _C = C.copy()
    _is_clique = True
    while _C and _is_clique:
        v = _C.pop()
        _is_clique = all(elem in G.neighbors(v) for elem in _C)

    if not _is_clique:
        for elem in _C:
            if elem not in G.neighbors(v):
                return (v, elem)
    return None


def assert_is_clique(C, G):
    '''
      Check if C is a real clique
    '''
    _C = C.copy()
    while _C:
        v = _C.pop()
        assert all(elem in G.neighbors(v) for elem in _C)


def init_solution(C, G):
    '''
      Init solution by found heuristic
    '''
    global best_sol
    global best_x
    best_sol = len(C)
    best_x = {k: 0 for k in sorted(G.nodes)}
    for c in C:
        best_x[c] = 1
    best_x = best_x.values()


def clique_heuristic(G):
    '''
      Init solution by heuristic for maximum clique problem
    '''
    logging.info('Applying Clique heuristic')

    C_best = set()
    for v in G.nodes:
        C = set()
        C.add(v)
        M = {n: G.degree(n) for n in G.neighbors(v)}
        while True:
            w = max(M.items(), key=operator.itemgetter(1))[0]  # get node with max degree
            C.add(w)
            N_w = {n: G.degree(n) for n in G.neighbors(w)}
            M = {k: M[k] for k in set(M).intersection(set(N_w))}
            if not M:
                break

        if len(C) > len(C_best):
            C_best = C
    assert_is_clique(C, G)
    return C_best


def apply_coloring_constraints(model, G, painting_steps=10):
    '''
       Apply coloring to graph several times and add constraints to model.
       Constraints: if nodes have same colors they can not be included in the same clique
    '''
    logging.info(f'Applying Coloring constraints {painting_steps} times')
    for step in range(painting_steps):

        var = model.get_var_by_name  # get variable by name function
        # Apply coloring algorithm to graph
        colors_by_nodes = nx.algorithms.coloring.greedy_color(G, 'random_sequential')

        nodes_by_colors = defaultdict(list)
        for node, color in colors_by_nodes.items():
            nodes_by_colors[color].append(node)

        # Add painting constarints to the model
        for color, nodes_list in nodes_by_colors.items():
            model.add_constraint(
                model.le_constraint(sum(var(f'x_{i}') for i in nodes_list), 1)
            )


def init_graph(problem):
    '''
      Read graph from file
    '''
    logging.info('Initializing Graph')
    with open(problem, 'r') as f:
        edges = [
            tuple(map(int, line.split()[1:3])) for line in f if line.startswith('e')
        ]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def init_model(problem, G):
    '''
      Init LP model just with target function and light constraints. 
      This function should be followed by adding independet set constraints
    '''

    logging.info('Initializing Model')
    nodes = sorted(G.nodes)

    model = Model(name=f'Max Clique-{problem}')
    x = {i: model.continuous_var(name=f'x_{i}') for i in nodes}

    model.add_constraints([x[i] <= 1 for i in nodes])
    model.add_constraints([x[i] >= 0 for i in nodes])
    model.maximize(model.sum(x))

    return model


def init_logger(level, filename):
    '''
      Init logger with Log Level
    '''
    levels = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO}
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%I:%M:%S %p', filename=filename, level=levels[level.upper()])


def main(args):

    global best_sol 
    global best_x

    problem_file = args.problem  # 'cliques_problems/san200_0.7_1.clq'
    problem_name = problem_file.split('/')[1]
    init_logger(args.log_lvl, f'reports/bnc/{problem_name}.report')

    logging.info(f'Problem: {problem_name}')

    G = init_graph(problem_file)
    model = init_model(problem_name, G)
    apply_coloring_constraints(model, G, painting_steps=args.color_step)

    C = clique_heuristic(G)
    init_solution(C, G)

    logging.info(f'Init solution is {best_sol}: {best_x}')
    logging.info(f'Model Information: {get_model_info(model)}')
    logging.info('Starting Branch and Cuts...')
    start = time.time()
    bnc(model, G)
    end = time.time()
    logging.info('Branch and Cuts have finished')
    logging.info(f'BnC time is {(end-start)/60} minutes')
    best_clique = [i for i, j in enumerate(best_x) if j > 0]
    logging.info(f'Best solution is {best_sol}: {best_clique}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BnC Solver for Maximum Clique Problem')
    parser.add_argument('-p', '--problem', dest="problem")
    parser.add_argument('-l', '--log-lvl', dest="log_lvl", default='INFO')
    parser.add_argument('-c', '--color-step', dest="color_step", default=10, type=int)
    args = parser.parse_args()
    main(args)
