import joblib
import numpy as np
from hls4ml.model import HLSBDT

def sklearn_to_hls(config):
    bdt = joblib.load(config['SklearnModel'])
    ensembleDict = {'max_depth' : bdt.max_depth, 'n_trees' : bdt.n_estimators,
                    'n_features' : len(bdt.feature_importances_),
                    'n_classes' : bdt.n_classes_, 'trees' : [],
                    'init_predict' : bdt.init_.predict(np.array([0]))[0].tolist(),
                    'norm' : 1}
    for trees in bdt.estimators_:
        treesl = []
        for tree in trees:
            tree = treeToDict(bdt, tree.tree_)
            treesl.append(tree)
        ensembleDict['trees'].append(treesl)

    return HLSBDT(config, ensembleDict)

def treeToDict(bdt, tree):
    # Extract the relevant tree parameters
    # NB node values are multiplied by the learning rate here, saving work in the FPGA
    treeDict = {'feature' : tree.feature.tolist(), 'threshold' : tree.threshold.tolist(), 'value' : (tree.value[:,0,0] * bdt.learning_rate).tolist()}
    treeDict['children_left'] = tree.children_left.tolist()
    treeDict['children_right'] = tree.children_right.tolist()
    return treeDict
