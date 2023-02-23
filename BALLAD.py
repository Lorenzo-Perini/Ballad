import numpy as np
from skopt import gp_minimize
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import pandas as pd
from src.anomatools.models import SSDO
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

def load_dataset(dataset_name, path, random_state):
    dslist = ['ALOI_withoutdupl_norm.arff',
              'Annthyroid_withoutdupl_norm_07.arff',
              'Arrhythmia_withoutdupl_norm_10_v01.arff',
              'Cardiotocography_withoutdupl_norm_05_v02.arff',
              'Glass_withoutdupl_norm.arff',
              'InternetAds_withoutdupl_norm_05_v01.arff',
              'KDDCup99_withoutdupl_norm_catremoved.arff',
              'PageBlocks_norm_10.arff',
              'Parkinson_withoutdupl_norm_10_v01.arff',
              'PenDigits_withoutdupl_norm_v01.arff',
              'Pima_withoutdupl_norm_05_v01.arff',
              'Shuttle_withoutdupl_norm_v02.arff',
              'SpamBase_withoutdupl_norm_05_v01.arff',
              'Stamps_withoutdupl_norm_09.arff',
              'Waveform_withoutdupl_norm_v01.arff',
              'Wilt_withoutdupl_norm_02_v01.arff',
              'WBC_withoutdupl_norm_v01.arff',
              'WDBC_withoutdupl_norm_v02.arff',
              'WPBC_withoutdupl_norm_05.arff']

    if dataset_name in dslist:
        np.random.seed(331)
        data = arff.loadarff(path+dataset_name)
        df = pd.DataFrame(data[0])
        df['outlier'] = [string.decode("utf-8") for string in df['outlier'].values]
        y = np.asarray([1 if string == 'yes' else -1 for string in df['outlier'].values])
        X = df[df.columns[:-2]].values
    else:
        np.random.seed(331)
        n_samples = 900
        n_anom = 100
        X = make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0]-np.array([0.5, 0.25])
        X = np.concatenate([X, np.random.uniform(low=-3, high=3, size=(n_anom, 2))], axis=0)
        y = -1*np.ones(n_samples + n_anom, np.int)
        y[-n_anom:] = 1
        
    idx_norm = y == -1
    idx_out = y == 1
    contamination = sum(idx_out)/len(idx_out)
        
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],
                                                                            test_size=0.6,
                                                                            random_state=random_state)

    X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],
                                                                        test_size=0.6,
                                                                        random_state=random_state)

    X_val_norm, X_test_norm, y_val_norm, y_test_norm = train_test_split(X_test_norm, y_test_norm,
                                                                        test_size=0.333,
                                                                        random_state=random_state)

    X_val_out, X_test_out, y_val_out, y_test_out = train_test_split(X_test_out, y_test_out,
                                                                    test_size=0.333,
                                                                    random_state=random_state)

    X_train = np.concatenate((X_train_norm, X_train_out))
    X_test = np.concatenate((X_test_norm, X_test_out))
    y_train = np.concatenate((y_train_norm, y_train_out))
    y_test = np.concatenate((y_test_norm, y_test_out))
    X_val = np.concatenate((X_val_norm, X_val_out))
    y_val = np.concatenate((y_val_norm, y_val_out))
    scaler = StandardScaler().fit(X_train)
    X_train_stand = scaler.transform(X_train)
    X_test_stand = scaler.transform(X_test)
    X_val_stand = scaler.transform(X_val)

    # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(X_train_stand)
    X_train_scaled = minmax_scaler.transform(X_train_stand)
    X_test_scaled = minmax_scaler.transform(X_test_stand)
    X_val_scaled = minmax_scaler.transform(X_val_stand)
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, contamination


def initialize_loop(X_train, contamination, random_state, c_r, c_fp, c_fn, metric):
    detector = SSDO()
    detector.fit(X_train, np.zeros(len(X_train), np.int))
    
    if metric == 'cosine':
        oldpred_AL = detector.predict(X_train)
        oldpred_LR = np.zeros(len(X_train), np.int)
        
    elif metric == 'entropy':
        probs = detector.predict_proba(X_train)[:,1]
        oldpred_AL = -np.nan_to_num(probs*np.log(probs), nan=0.0, posinf=0.0, neginf=0.0)
        rejection_threshold = 0.1
        reject_probs = np.exp(np.log(.5)*np.power(np.divide(np.array(2*abs(probs - 0.5)), rejection_threshold), 2))
        oldpred_LR = -np.nan_to_num(reject_probs*np.log(reject_probs), nan=0.0, posinf=0.0, neginf=0.0)
        
    return oldpred_AL, oldpred_LR


def add_validation_labels(y_val_semitargets, y_val, budget):
    notusedIndx = np.where(y_val_semitargets == 0)[0]
    indeces = np.random.choice(notusedIndx, budget, replace = False)
    y_val_semitargets[indeces] = y_val[indeces]
    return y_val_semitargets

def set_rejection_threshold(val_probs, y_val, c_r, c_fp, c_fn):
    if len(val_probs) == 0:
        rejection_threshold = 0.0
        print("##### NO VALIDATION LABELS. SETTING THE REJECTION THRESHOLD TO 0.")
    else:
        rej_scores = 2*abs(val_probs - 0.5)
        max_quantile = np.quantile(rej_scores, 0.5)
        min_quantile = np.quantile(rej_scores, 0.01)
        if min_quantile == max_quantile:
            min_quantile = max(0, max_quantile - 0.1)
            max_quantile = min(1, max_quantile +0.1)
        predictions = np.where(val_probs >= 0.5, 1, 0)
    
    def optimize_cost_function(thr):
        non_rejected = np.where(rej_scores >= thr)[0]

        y_pred = predictions[non_rejected]
        labels = y_val[non_rejected]
        
        false_positives = np.shape(np.intersect1d(np.where(y_pred == 1)[0], np.where(labels == -1)[0]))[0]
        false_negatives = np.shape(np.intersect1d(np.where(y_pred == -1)[0], np.where(labels == 1)[0]))[0]
        
        cost = (false_positives*c_fp + false_negatives*c_fn + (len(y_val) - len(non_rejected)) *c_r)/len(y_val)
        return cost
    
    res = gp_minimize(func = optimize_cost_function, dimensions = [(min_quantile,max_quantile)], n_calls = 20,
                      random_state = 331, x0 = [np.quantile(rej_scores, 0.10)])
    rejection_threshold = res.x
    return rejection_threshold[0]

def measure_rewards(predictions, probs, rejection_threshold, oldpred_AL = [], oldpred_LR = [], metric = 'cosine'):
    if metric == 'cosine':
        newpred_AL = np.copy(predictions)
        newpred_LR = np.zeros(len(predictions), np.int)
        reject_indx = reject(predictions, probs, rejection_threshold)
        newpred_LR[reject_indx] = 1
        ALreward = cosine(oldpred_AL, newpred_AL)
        LRreward = cosine(oldpred_LR, newpred_LR)

    elif metric == 'entropy':
        newpred_AL = -np.nan_to_num(probs*np.log2(probs), nan=0.0, posinf=0.0, neginf=0.0)
        reject_probs = np.exp(np.log(.5)*np.power(np.divide(np.array(2*abs(probs - 0.5)),rejection_threshold), 2))
        newpred_LR = -np.nan_to_num(reject_probs*np.log(reject_probs), nan=0.0, posinf=0.0, neginf=0.0)
        ALreward = np.sum(abs(oldpred_AL - newpred_AL))/len(newpred_AL)
        LRreward = np.sum(abs(oldpred_LR - newpred_LR))/len(newpred_LR)
        
    return ALreward, LRreward, newpred_AL, newpred_LR

def cost_function(labels, predictions, c_r, c_fp, c_fn):
    if np.shape(predictions)[0] != np.shape(labels)[0]:
        print("Predictions and true labels do not have the same size.")
        return
    false_positives = np.shape(np.intersect1d(np.where(predictions == 1)[0], np.where(labels == -1)[0]))[0]
    false_negatives = np.shape(np.intersect1d(np.where(predictions == -1)[0], np.where(labels == 1)[0]))[0]
    nrejections = np.shape(np.where(predictions == 2)[0])[0]
    cost = (false_positives*c_fp + false_negatives*c_fn + nrejections*c_r)/len(labels)
    return cost, false_positives/len(labels), false_negatives/len(labels), nrejections/len(labels)

def reject(predictions, probs, rejection_threshold):
    reject_idx = np.where(2*abs(probs - 0.5) < rejection_threshold)[0]
    return reject_idx


def add_training_labels(y_train_semitargets, y_train, rankingAL, poolBudget):
    unlabeledIdx = np.where(y_train_semitargets == 0)[0]
    if len(rankingAL) == 0:
        indeces = np.random.choice(unlabeledIdx, poolBudget, replace = False)
        y_train_semitargets[indeces] = y_train[indeces]
    else:
        commonIdxs = np.array([x for x in rankingAL if x in unlabeledIdx])
        indeces = commonIdxs[:poolBudget]
        y_train_semitargets[indeces] = y_train[indeces]

    return y_train_semitargets

