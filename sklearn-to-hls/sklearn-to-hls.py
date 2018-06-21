from sklearn.externals import joblib
import numpy as np
import os
import argparse
import yaml
import sys
filedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filedir, "..", "hls-writer"))
from hls_writer import bdt_writer, parse_config

class Node:
  def __init__(self, i=None, parent=None, lChild=None, rChild=None):
    self._parent = parent
    self._lChild = lChild
    self._rChild = rChild
    self._i = i

def nParents(node):
  n = 0
  while node._parent is not None:
    node = node._parent
    n += 1
  return n

def makeBalancedTreeLeftFirst(depth=3):
  iN = 0
  node = Node(i=iN)
  iN += 1
  n_nodes = 2**(depth + 1) - 1
  children_left = [-1] * n_nodes
  children_right = [-1] * n_nodes
  parents = [-1] * n_nodes
  while iN < 2**(depth+1) -1:
    if node._lChild is None and nParents(node) < depth:
      new_node = Node(i=iN, parent=node)
      node._lChild = new_node
      children_left[node._i] = new_node._i
      parents[new_node._i] = node._i
      node = new_node
      iN += 1
    elif node._rChild is None and nParents(node) < depth:
      new_node = Node(i=iN, parent=node)
      node._rChild = new_node
      children_right[node._i] = new_node._i
      parents[new_node._i] = node._i
      node = new_node
      iN += 1
    else:
      climb = True
      while climb:# node._parent._rChild is not None and node._parent._i != 0:
        if node._parent is None:
          climb = False
        elif node._rChild is None and nParents(node) < depth:
          climb = False
        else:
          node = node._parent
      if node._i == 0 and node._lChild is not None and node._rChild is not None:
        done = True
  while node._parent is not None:
    node = node._parent
  return {'children_left' : children_left, 'children_right' : children_right, 'parents' : parents}


def ensembleToDict(bdt):
  ensembleDict = {'max_depth' : bdt.max_depth, 'n_trees' : bdt.n_estimators,
                  'n_features' : len(bdt.feature_importances_),
                  'n_classes' : bdt.n_classes_, 'trees' : [],
                  'init_predict' : bdt.init_.predict(np.array([0]))[0].tolist()}
  for trees in bdt.estimators_:
    treesl = []
    for tree in trees:
      tree = treeToDict(bdt, tree.tree_)
      tree = padTree(ensembleDict, tree)
      treesl.append(tree)
    ensembleDict['trees'].append(treesl)
    ensembleDict['base_tree'] = makeBalancedTreeLeftFirst(bdt.max_depth)
  return ensembleDict

def treeToDict(bdt, tree):
  # Extract the relevant tree parameters
  # NB node values are multiplied by the learning rate here, saving work in the FPGA
  treeDict = {'feature' : tree.feature.tolist(), 'threshold' : tree.threshold.tolist(), 'value' : (tree.value[:,0,0] * bdt.learning_rate).tolist()}
  treeDict['children_left'] = tree.children_left.tolist()
  treeDict['children_right'] = tree.children_right.tolist()
  # add the parent index
  n = len(tree.children_left) # number of nodes
  parents = [0] * n
  for i in range(n):
    j = tree.children_left[i]
    if j != -1:
      parents[j] = i
    k = tree.children_right[i]
    if k != -1:
      parents[k] = i
  parents[0] = -1
  treeDict['parent'] = parents
  # Add the depth info
  treeDict['depth'] = [0] * n
  for i in range(n):
    depth = 0
    parent = treeDict['parent'][i]
    while parent != -1:
      depth += 1
      parent = treeDict['parent'][parent]
    treeDict['depth'][i] = depth
  return treeDict

def padTree(ensembleDict, treeDict):
  '''Pad a tree with dummy nodes if not perfectly balanced or depth < max_depth'''
  n_nodes = len(treeDict['children_left'])
  # while th tree is unbalanced
  while n_nodes != 2 ** (ensembleDict['max_depth'] + 1) - 1:
    for i in range(n_nodes):
      if treeDict['children_left'][i] == -1 and treeDict['depth'][i] != ensembleDict['max_depth']:
        treeDict['children_left'].extend([-1, -1])
        treeDict['children_right'].extend([-1, -1])
        treeDict['parent'].extend([i, i])
        treeDict['feature'].extend([-2, -2])
        treeDict['threshold'].extend([-2.0, -2.0])
        val = treeDict['value'][i]
        treeDict['value'].extend([val, val])
        newDepth = treeDict['depth'][i] + 1
        treeDict['depth'].extend([newDepth, newDepth])
        iRChild = len(treeDict['children_left']) - 1
        iLChild = iRChild - 1
        treeDict['children_left'][i] = iLChild
        treeDict['children_right'][i] = iRChild
    n_nodes = len(treeDict['children_left'])
  return treeDict

def main():
  
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='')
  parser.add_argument("-c", action='store', dest='config',
                      help="Configuration file.")
  args = parser.parse_args()
  if not args.config: parser.error('A configuration file needs to be specified.')
  configDir  = os.path.abspath(os.path.dirname(args.config))
  yamlConfig = parse_config(args.config)
  if not os.path.isabs(yamlConfig['OutputDir']):
    yamlConfig['OutputDir'] = os.path.join(configDir, yamlConfig['OutputDir'])
  if not os.path.isabs(yamlConfig['sklearnPkl']):
    yamlConfig['sklearnPkl'] = os.path.join(configDir, yamlConfig['sklearnPkl'])

  if not (yamlConfig["IOType"] == "io_parallel"):
    raise Exception('ERROR: Invalid IO type (serial not yet supported)')

  ######################
  ##  Do translation
  ######################
  if not os.path.isdir("{}/firmware".format(yamlConfig['OutputDir'])):
    os.makedirs("{}/firmware".format(yamlConfig['OutputDir']))

  bdt = joblib.load(yamlConfig['sklearnPkl'])
  ensembleDict = ensembleToDict(bdt)
  bdt_writer(ensembleDict, yamlConfig)

if __name__ == "__main__":
  main()
