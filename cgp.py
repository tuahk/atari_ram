from dcgpy import *
from dcgpy import expression_double, kernel_set_double
from pyaudi import gdual_double as gdual

import operator
import math
from typing import Dict, Sequence
import networkx as nx

# Setup parameters
KERNELS = ['sum', 'diff', 'mul', 'div']
ARITY = 2
SEED = 23
MUTATION_RATE = 0.05

def generate_individual(n_inputs, n_outputs, n_cols, arity, kernels):
    return expression_double(
      inputs = n_inputs, 
      outputs = n_outputs, 
      rows = 1, 
      cols = n_cols, 
      levels_back = n_cols+1, 
      arity = arity, 
      kernels = kernels, 
      n_eph = 0, 
      # seed = SEED,
    )

def generate_population(n_inputs, n_outputs, n_cols, arity, kernels, n=5):
  pop = []
  for i in range(n):
    pop.append(
      generate_individual(
      n_inputs, 
      n_outputs, 
      n_cols, 
      arity, 
      kernels, 
    ))
  return pop

def mu_lambda(pop, parent):
  num_of_mutations = int(len(parent.get())*MUTATION_RATE)
  # print(num_of_mutations)
  for i,ind in enumerate(pop):
    # Kaikkien yksilöiden geeneiksi vanhemman geeit
    pop[i].set(parent.get())
    if i != 0:
      # Mutatoidaan muiden yksilöiden geenejä - lähtökohtana vahnemman geenit
      pop[i].mutate_random(num_of_mutations)
  return pop

def save_ind(ind,f_name):
  with open(f_name, 'w') as f:
    for gene in ind.get():
        f.write('%s\n' % gene)

def load_genes(ind, f_name):
  genes = []
  with open(f_name, 'r') as f:
    for line in f:
      # remove linebreak which is the last character of the string
      gene = line[:-1]
      # add item to the list
      genes.append(int(gene))
  ind.set(genes)

def draw_ind(ind):
  input_names = ['x'+str(i) for i in range(0,ind.get_n())]
  simp = ind.simplify(input_names)
  print(simp)
  # graph = extract_computational_subgraph(ind, KERNELS)
  # visualize(graph, 'img/last_best.pdf', ind=ind)
  # graph = ind.visualize(input_names, False, False)
  # print("graph active done")
  # graph.render('img/best_ind_active')
  # print("render active done")
  # graph = ind.visualize(input_names, True, False)
  # print("graph all done")
  # graph.render('img/best_ind_all')
  # print("render all done")

if __name__ == "__main__":
  n_inputs = 4 
  n_outputs = 2 
  n_cols = 15
  arity = ARITY
  kernels = kernel_set_double(KERNELS)()

  ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
  draw_ind(ind)


# This does not currently work
# def extract_computational_subgraph(ind, kernels):
#     """Extract a computational subgraph of the CGP graph `ind`, which only contains active nodes.
#     Args:
#         ind (cgp.Individual): an individual in CGP  
#     Returns:
#         nx.DiGraph: a acyclic directed graph denoting a computational graph
#     See https://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/ and 
#     http://www.cs.columbia.edu/~mcollins/ff2.pdf for knowledge of computational graphs.
#     """
#     print(ind.get())
#     print(ind.get_active_nodes())
#     print(ind.get_gene_idx())
#     print(ind.get_arity())
#     # in the digraph, each node is identified by its index in `ind.nodes`
#     # if node i depends on node j, then there is an edge j->i
#     g = nx.MultiDiGraph()  # possibly duplicated edges
#     order = 1
#     for i in ind.get_active_nodes():
#       i_gene = ind.get_gene_idx()[i]
#       f_index = ind.get()[i_gene]
#       f = kernels[f_index]

#       g.add_node(i, func=f)
#       if i >= ind.get_n():
#         arity = ind.get_arity(i)
#       else:
#         arity = 0
#       for j in range(arity):
#         input_index = ind.get()[i_gene+j+1]
#         # w = node.weights[j]
#         g.add_edge(input_index, i, order=order)
#         order += 1
#     # add output nodes
#     for i in range(ind.get_m()):
#       n_nodes = ind.get_n() + ind.get_cols()
#       g.add_node(n_nodes+i)
#       input_index = ind.get()[n_nodes+i]
#       g.add_edge(input_index, n_nodes+i, order=order)
#     return g



# def visualize(g: nx.MultiDiGraph, to_file: str, ind, input_names: Sequence = None, operator_map: Dict = None):
#     """Visualize an acyclic graph `g`.
#     Args:
#         g (nx.MultiDiGraph): a graph
#         to_file (str): file path
#         input_names (Sequence, optional): a list of names, each for one input. If `None`, then a default name "vi" is used
#             for the i-th input. Defaults to None.
#         operator_map (Dict, optional): Denote a function by an operator symbol for conciseness. Defaults to None. If `None`,
#             then +-*/ are used.
#     """

#     from networkx.drawing.nx_agraph import to_agraph
#     import pygraphviz
#     layout = 'dot'
#     # label each function node with an operator
#     if operator_map is None:
#         operator_map = {'sum': '+',
#                         'diff': '-',
#                         'div': '/',
#                         'mul': '*',
#                         operator.add.__name__: '+',
#                         operator.neg.__name__: '-',
#                         operator.mul.__name__: '*',
#                         "protected_div": '/',
#                         'sin': 'sin',
#                         'cos': 'cos',
#                         'min': 'min',
#                         'max': 'max',
#                         }
#     for n in g.nodes:
#       print(n)
#       attr = g.nodes[n]
#       print(attr)
#       if n >= ind.get_n() + ind.get_cols(): # output node
#           attr['color'] = 'red'
#           attr['label'] = 'o_' + str(n)
#       elif n >= ind.get_n():  # function node
#           if attr['func'] not in operator_map:
#               print(
#                   f"Operator notation of '{attr['func']}'' is not available. The node id is shown instead.")
#           attr['label'] = operator_map.get(attr['func'], n)
#           if g.out_degree(n) == 0:  # the unique output node
#               attr['color'] = 'red'
#       else:  # input node
#           attr['color'] = 'green'
#           attr['label'] = input_names[-n -
#                                       1] if input_names is not None else f'v{-n}'

#     ag: pygraphviz.agraph.AGraph = to_agraph(g)
#     ag.layout(layout)
#     ag.draw(to_file)