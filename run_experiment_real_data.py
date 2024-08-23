# Based on code from the paper "Classification with Valid and Adaptive Coverage" by Romano, Sesia and Candès (Neurips 2020)
# See https://github.com/msesia/arc


import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os.path
from os import path
import random
import torch


import sys
sys.path.insert(0, '..')
import source

from source.misc import train_test_split_wrapper

def assess_predictions(S, X, y, selection_type, null_class=None):
    # Marginal coverage
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    # Average length
    length = np.mean([len(S[i]) for i in range(len(y))])
    # Average length conditional on coverage
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    length_cover = np.mean([len(S[i]) for i in idx_cover])


    K=np.max(y)+1
    if selection_type=='min': # List of informative sets 
        sel= [(len(S[i]) < K) for i in range(len(y))]
        m1 = len(y)
    elif selection_type=='nonull':
        sel= [(len(S[i]) < K) and (null_class not in S[i]) for i in range(len(y))]
        m1 = np.sum([y[i]==1 or y[i]==2 for i in range(len(y))])
    else: raise ValueError('selection type')

    n_rej=np.sum(sel)
    if n_rej>0: 
        # False coverage proportion 
        FCP = np.sum([y[i] not in S[i] and sel[i] for i in range(len(y))]) / n_rej 
        # Resolution-adjusted power
        power=np.sum([1/len(S[i]) if (y[i] in S[i] and sel[i]) else 0 for i in range(len(y))])  / m1
        # Length of informative sets
        length_info = np.sum([len(S[i]) if sel[i] else 0 for i in range(len(y))]) / n_rej
    else: FCP=0; power =0;length_info=0


    # Combine results
    out = pd.DataFrame({'Coverage': [coverage], 
                        'Length': [length], 'Length cover': [length_cover],
                        'FCR':[FCP],
                        'Selection rate':[n_rej/len(y)],
                        'Length info':[length_info],
                        'Power':[power]},)

    
    return out



def run_experiment(X,y, out_dir, n_train, alpha, experiment_nb, 
                   selection_type='min', null_class=None):

    """
    X is of type np.ndarray or torch.utils.data.Dataset
    y is array-like 

    See notebook example
    """
    num_classes = np.max(y)+1

    # Determine output file
    out_file = out_dir + "/summary_" +selection_type+".csv"
    
    # Random state for this experiment
    random_state = 2020 + experiment_nb
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    
    # List of calibration methods to be compared
    
    if selection_type=='min':
    
        methods = {
                'CC': source.methods.StandardSplitConformal,
                'InfoSP': source.methods.minSel,
                'InfoSCOP': source.methods.minSelpreproc,
            }
        
    elif selection_type=='nonull':
        methods = {
                'CC': source.methods.StandardSplitConformal,
                'InfoSP': source.methods.nonullSelbasic,
                'InfoSCOP': source.methods.nonullSelpreproc,
            }
            
    else: raise ValueError('selection type')
    
    

    # List of black boxes to be compared
    black_boxes = {
                   #'SVC': source.black_boxes.SVC(random_state=random_state),
                   #'RFC': source.black_boxes.RFC(n_estimators=100,
                                              #criterion="gini",
                                              #max_depth=None,
                                              ##max_features="sqrt",
                                              #min_samples_leaf=3,
                                              #random_state=random_state),
                   #'NNet': source.black_boxes.NNet(hidden_layer_sizes=64,
                                                #batch_size=128,
                                                #learning_rate_init=0.01,
                                                #max_iter=20,
                                                #random_state=random_state), 

                    'CNN': source.black_boxes.CNN(in_channels=3, num_classes=num_classes)
                  }

    
    # Total number of samples
    #n_test = min( X.shape[0] - n_train, 5000) 
    n_test = min( len(X) - n_train, 1000) 
    if n_test<=0:
        return

    # Split data into train/test sets   
    X_train, X_test, y_train, y_test = train_test_split_wrapper(X, y, test_size=n_test, random_state=random_state) 
    
    #X_train = X_train[:n_train] #to do
    if isinstance(X, np.ndarray):
        X_train = X_train[:n_train]
    elif isinstance(X, torch.utils.data.Dataset):
        if n_train < len(X_train): 
            X_train = torch.utils.data.Subset(X_train,np.arange(n_train)) 
    y_train = y_train[:n_train] 

    # Load pre-computed results
    if path.exists(out_file) & path.exists(out_file):
        results = pd.read_csv(out_file)
    else:
        results = pd.DataFrame()
        
    for box_name in black_boxes:
        black_box = black_boxes[box_name]
        for method_name in methods:
            # Skip if this experiment has already been run
            if results.shape[0] > 0:
                found  = (results['Method']==method_name)
                found &= (results['Black box']==box_name)
                found &= (results['Experiment']==experiment_nb)
                found &= (results['Nominal']==1-alpha)
                found &= (results['n_train']==n_train)
                found &= (results['n_test']==n_test)
            else:
                found = 0
            if np.sum(found) > 0:
                print("Skipping experiment with black-box {} and method {}...".format(box_name, method_name))
                sys.stdout.flush()
                continue

            print("Running experiment with black-box {} and method {}...".format(box_name, method_name))
            sys.stdout.flush()

            # Train classification method
            method = methods[method_name](X_train, y_train, black_box, alpha, random_state=random_state,
                                          verbose=True,
                                          null_class=null_class)
            # Apply classification method
            S = method.predict(X_test)

            # Evaluate results
            res = assess_predictions(S, X_test, y_test, selection_type=selection_type, null_class=null_class)
            # Add information about this experiment
            res['Method'] = method_name
            res['Black box'] = box_name
            res['Experiment'] = experiment_nb
            res['Nominal'] = 1-alpha
            res['n_train'] = n_train
            res['n_test'] = n_test


            # Add results to the list
            results = pd.concat([results, res])

            # Write results on output files
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            
            results.to_csv(out_file, index=False, float_format="%.4f")
            print("Updated summary of results on\n {}".format(out_file))

    return results

