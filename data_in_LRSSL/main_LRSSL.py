from torch import nn, optim
import numpy as np
import csv
from evaluate_LRSSL import cv_model_evaluate
import torch as t
from torch import nn
from torch_geometric.nn import conv
import random


class Paramater(object):
    def __init__(self):
        self.validation = 5
        self.epoch = 1000
        self.alpha = 0.03


class losses(nn.Module):
    def __init__(self):
        super(losses, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1 - opt.alpha) * loss_sum[one_index].sum() + opt.alpha * loss_sum[zero_index].sum()


class data(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index):
        return (self.data_set['drug'],
                self.data_set['disease'],
                self.data_set['drug_dis_train']['train'],
                self.data_set['drug_dis'])


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()

        self.d = dataset['disease']['data'].size(0)
        self.r = dataset['drug']['data'].size(0)
        self.IN = 256
        self.OUT = 256
        self.first_GCN_disease = conv.GCNConv(self.IN, self.OUT)
        self.second_GCN_disease = conv.GCNConv(self.IN, self.OUT)
        self.first_GCN_drug = conv.GCNConv(self.IN, self.OUT)
        self.second_GCN_drug = conv.GCNConv(self.IN, self.OUT)

        self.first_linear_disease = nn.Linear(self.IN, 256)
        self.second_linear_disease = nn.Linear(256, 128)
        self.third_linear_disease = nn.Linear(128, 64)
        self.first_linear_drug = nn.Linear(self.IN, 256)
        self.second_linear_drug = nn.Linear(256, 128)
        self.third_linear_drug = nn.Linear(128, 64)

    def forward(self, input):
        t.cuda.manual_seed(0)
        t.manual_seed(0)
        disease_random = t.randn(self.d, self.OUT)
        drug_random = t.randn(self.r, self.OUT)

        disease_GCN_1 = t.relu(self.first_GCN_disease(disease_random.cuda("cuda:0"),
                                input[1]['edge_index'].cuda("cuda:0"),
                                input[1]['data'][input[1]['edge_index'][0],
                                                 input[1]['edge_index'][1]].cuda("cuda:0")))
        disease_GCN_2 = t.relu(self.second_GCN_disease(disease_GCN_1.cuda("cuda:0"),
                               input[1]['edge_index'].cuda("cuda:0"),
                               input[1]['data'][input[1]['edge_index'][0],
                                                input[1]['edge_index'][1]].cuda("cuda:0")))

        drug_GCN_1 = t.relu(self.first_GCN_drug(drug_random.cuda("cuda:0"),
                                input[0]['edge_index'].cuda("cuda:0"),
                                input[0]['data'][input[0]['edge_index'][0],
                                                 input[0]['edge_index'][1]].cuda("cuda:0")))
        drug_GCN_2 = t.relu(self.second_GCN_drug(drug_GCN_1.cuda("cuda:0"),
                               input[0]['edge_index'].cuda("cuda:0"),
                               input[0]['data'][input[0]['edge_index'][0],
                                                input[0]['edge_index'][1]].cuda("cuda:0")))

        disease_linear_1 = t.relu(self.first_linear_disease(disease_GCN_2))
        disease_linear_2 = t.relu(self.second_linear_disease(disease_linear_1))
        disease = t.relu(self.third_linear_disease(disease_linear_2))

        drug_linear_1 = t.relu(self.first_linear_drug(drug_GCN_2))
        drug_linear_2 = t.relu(self.second_linear_drug(drug_linear_1))
        drug = t.relu(self.third_linear_drug(drug_linear_2))
        return disease.mm(drug.t())


def epochLossAndScore(model, one_index, zero_index, train_data, optimizer):
    model.zero_grad()
    score = model(train_data)
    myloss = losses()
    loss = myloss(one_index, zero_index, train_data[3].cuda(), score)

    loss.backward()
    optimizer.step()
    return loss, score


def train(model, train_data, data_test, association_matrix, optimizer, opt):
    one_index = train_data[2][0].cuda().t().tolist()
    zero_index = train_data[2][1].cuda().t().tolist()
    one_index_test = data_test[0].cuda().t().tolist()
    zero_index_test = data_test[1].cuda().t().tolist()

    model.train()
    for epoch in range(1, opt.epoch + 1):
        train_reg_loss, score = epochLossAndScore(model, one_index, zero_index, train_data, optimizer)
        np.savetxt('score_LRSSL.csv', score.detach().cpu().numpy(), fmt='%.6f', delimiter=',')
        if epoch % 100 == 0:
            metric = cv_model_evaluate(one_index_test, zero_index_test, train_data[3], score)
            print(epoch)
            print(metric)

    return metric, score


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


opt = Paramater()


def main():
    dataset = dict()

    diseaseSim = read_csv('lrssl_wrr_GS.csv')
    drugSim = read_csv('lrssl_wdd_GS.csv')
    dataset['drug_dis'] = read_csv('lrssl_dr.csv')

    drug_edge_index = edge_index(drugSim)
    dataset['drug'] = {'data': drugSim,
                       'edge_index': drug_edge_index}
    disease_edge_index = edge_index(diseaseSim)
    dataset['disease'] = {'data': diseaseSim,
                          'edge_index': disease_edge_index}

    k_folds = 10
    metric_validation = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(opt.validation):
        print('validation:', i + 1)
        seed = i
        value = dataset['drug_dis']
        index_matrix = np.mat(np.where(value > 0.1))
        association_nam = index_matrix.shape[1]
        random_index = index_matrix.T.tolist()
        random.seed(seed)
        random.shuffle(random_index)
        CV_size = int(association_nam / k_folds)
        temp = np.array(random_index[:association_nam - association_nam %
                                      k_folds]).reshape(k_folds, CV_size, -1).tolist()
        temp[k_folds - 1] = temp[k_folds - 1] + \
                            random_index[association_nam - association_nam % k_folds:]
        random_index = temp

        index_matrix_negative = np.mat(np.where(value <= 0.1))
        association_nam_negative = index_matrix_negative.shape[1]
        random_index_negative = index_matrix_negative.T.tolist()
        random.seed(seed)
        random.shuffle(random_index_negative)
        CV_size = int(association_nam_negative / k_folds)
        temp_negative = np.array(random_index_negative[:association_nam_negative - association_nam_negative %
                                                        k_folds]).reshape(k_folds, CV_size, -1).tolist()
        temp_negative[k_folds - 1] = temp_negative[k_folds - 1] + \
                                     random_index_negative[
                                     association_nam_negative - association_nam_negative % k_folds:]
        random_index_negative = temp_negative

        dataset['drug_dis'] = read_csv('lrssl_dr.csv')
        drug_dis = dataset['drug_dis']
        num = 0
        metric = []
        metric_fold = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for k in range(k_folds):
            print('k_folds:', k + 1)
            num += 1
            train_matrix = np.matrix(drug_dis.detach().numpy(), copy=True)
            association_matrix = np.matrix(drug_dis.detach().numpy(), copy=True)
            train_matrix[tuple(np.array(random_index[k]).T)] = 0
            dataset['drug_dis'] = t.FloatTensor(train_matrix)

            one_index_test = random_index[k]
            zero_index_test = random_index_negative[k]
            zero_tensor_test = t.LongTensor(zero_index_test)
            one_tensor_test = t.LongTensor(one_index_test)
            date_test = [one_tensor_test, zero_tensor_test]

            one_index_train = []
            zero_index_train = []
            for m in range(k_folds):
                if k != m:
                    one_index_train.extend(random_index[m])
                    zero_index_train.extend((random_index_negative[m]))
            one_tensor_train = t.LongTensor(one_index_train)
            zero_tensor_train = t.LongTensor(zero_index_train)
            dataset['drug_dis_train'] = dict()
            dataset['drug_dis_train']['train'] = [one_tensor_train, zero_tensor_train]

            model = Model(dataset)
            model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.002)
            train_data = data(opt, dataset)

            metric2, score = train(model, train_data[i], date_test, association_matrix, optimizer, opt)
            metric.append(metric2)
            metric_fold += metric2

        print('result=', metric)
        print('metric_fold=')
        print(np.divide(metric_fold, k + 1))
        metric_validation += metric_fold
    print()
    print('metric_validation', np.divide(metric_validation, (i + 1) * (k + 1)))


if __name__ == "__main__":
    main()
