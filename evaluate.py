import numpy as np
import torch as t
import csv
import random
import math


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    TPTN = TP * TN
    FPFN = FP * FN
    aa = TPTN - FPFN
    bb = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC_list = aa/bb
    MCC = MCC_list[max_index]
    return [auc[0, 0], aupr[0, 0], recall, precision, f1_score, MCC]


def test_Set(association, one_index_test, zero_index_test, train_data, score):
    test_index = [[], []]
    test_index[0].extend(one_index_test[0])
    test_index[1].extend(one_index_test[1])
    test_index[0].extend(zero_index_test[0])
    test_index[1].extend(zero_index_test[1])

    real = np.array(association[test_index])
    pred = np.array(score[test_index].detach().cpu())
    location_one = np.where(association[test_index] == 1)
    location_zero = np.where(association[test_index] == 0)
    real_one = real[location_one]
    temp = len(real_one)

    random.seed(1)

    rz = real[location_zero]
    random.shuffle(rz)
    real_zero = rz[0:temp]
    pred_one = pred[location_one]
    pz = pred[location_zero]
    random.shuffle(pz)
    pred_zero = pz[0:temp]

    real_score = np.append(real_one, real_zero)
    predict_score = np.append(pred_one, pred_zero)
    return real_score, predict_score


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def cv_model_evaluate(one_index_test, zero_index_test, train_data, score):

    association = read_csv('dr-predict.csv')
    real_set, predict_set = test_Set(association, one_index_test, zero_index_test, train_data, score)
    return get_metrics(real_set, predict_set)

