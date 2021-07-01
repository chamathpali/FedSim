# This is our proposed method FedSim.

import copy
import math
import time

import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import random
import utils.csv_log as csv_log

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad

from sklearn.cluster import KMeans
import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        self.num_groups = params["num_groups"]
        self.groups = []

        # Setup Log
        self.params_log = params
        self.run_name = str(params["ex_name"])+"_fedsim"
        self.log_main = []
        self.log_groups = { g:[] for g in range(self.num_groups)}
        csv_log.log_start('sim',params, self.num_groups, self.run_name)

        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        elapsed = []
        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()
                train_loss, train_acc, test_acc = self.arrange_stats(stats_train, stats)

                tqdm.write('At round {} accuracy: {}'.format(i, test_acc ))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i,train_acc ))
                tqdm.write('At round {} training loss: {}'.format(i,train_loss ))

                self.log_main.append([i, train_loss, train_acc, test_acc])
                self.test_groups(i)

                if i % 10 == 0:
                    csv_log.write_all('sim', self.log_main, self.log_groups, self.num_groups, self.run_name)
                    csv_log.graph_print('sim',self.params_log, self.num_groups, self.run_name)

            start_time = time.time()
            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            self.groups = self.ClusterGroups(active_clients.tolist(), i)


            csolns = []  # buffer for receiving client solutions
            cs = {}
            client_sols = {}

            # Locally Train the clients first
            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)
                # solve minimization locally - soln #samples, weights
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                # gather solutions from client
                csolns.append(soln)
                cs[idx] = soln
                client_sols[c.id] = soln

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # Create group solutions - Groups will be taken
            self.IntraTrainAggregate(client_sols, i)

            gsolns = [(self.groups[g]["num_samples"], self.groups[g]["model"] ) for id,g in enumerate(self.groups)]

            # update models
            self.latest_model = self.aggregate_group(gsolns) # BE CAREFUL

            elapsed_time = time.time() - start_time
            elapsed.append(elapsed_time)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)

        test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])

        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, test_acc ))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, train_acc))

        self.log_main.append([self.num_rounds, train_loss, train_acc, test_acc])
        self.test_groups(self.num_rounds)

        csv_log.write_all('sim', self.log_main, self.log_groups, self.num_groups, self.run_name)
        csv_log.graph_print('sim', self.params_log, self.num_groups, self.run_name)
        print("Time Taken Each Round: ")
        print(elapsed)
        print(np.mean(elapsed))
        csv_log.write_time_taken(elapsed, self.run_name)


    def IntraTrainAggregate(self, client_sols, round ):
        for idx,g in enumerate(self.groups):
            g_clients = self.groups[g]["clients"]
            gsolns = []
            for id,c in enumerate(g_clients):
                gsolns.append(client_sols[ c.id ]) # c.id is a string
                self.groups[g]["num_samples"] += c.num_samples
            self.groups[g]["model"] = self.aggregate(gsolns)

        return self.groups  # Updated groups with models aggregated with FedAvg

    def ClusterGroups(self, S, round):
        self.groups = []
        groups = ["group_" + str(i) for i in range(self.num_groups)]
        groups = {g: {"model": 0, "clients": [], "num_samples":0, "id": idx} for idx, g in enumerate(groups)}

        if round == 0:
            assign_idx = 0
            for idx, c in enumerate(S):
                groups["group_" + str(assign_idx)]["clients"].append(c)
                assign_idx += 1
                if assign_idx == self.num_groups:
                    assign_idx = 0

            return groups
        else:
            model_len = process_grad(self.latest_model).size
            X = []
            for idx, c in enumerate(S):
                # solve minimization locally - soln #samples, weights
                num, client_grad = c.get_grads(model_len)
                X.append(client_grad)

            X = np.array(X)

            pca = PCA(n_components=0.95, svd_solver= 'full')
            X_reduced = pca.fit_transform(X)

            # print(pca.get_params())
            # print(X_reduced.shape)
            # print(pca.explained_variance_ratio_)

            # FOR PRINTING THE PCA GRAPHS USE THIS
            # cumsum = np.cumsum(pca.explained_variance_ratio_)
            # fig, ax = plt.subplots(2, 1, figsize=[8, 12])
            # ax[0].plot(np.arange(1, pca.n_components_ + 1), cumsum, linewidth=3.0, color="#17becf")
            # ax[0].grid()
            # ax[0].set_xlabel("# Components", fontsize=18)
            # ax[0].set_ylabel('Variance', fontsize=18)
            # plt.show()

            km = KMeans(
                n_clusters= self.num_groups,
            )
            y_km = km.fit(X_reduced)
            print(km.labels_)
            csv_log.write_clusters(y_km.labels_, self.run_name)
            groups_predicted = y_km.labels_

            for idx, c in enumerate(S):
                groups["group_"+str(groups_predicted[idx])]["clients"].append(c)

            remove = [g for g in groups.keys() if len(groups[g]["clients"]) == 0]
            for k in remove: del groups[k]

            return groups


    def aggregate_group(self, wsolns):
        total_weight = 0.0
        base = [0] * len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of total samples in a group
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += v.astype(np.float64)

        averaged_soln = [v / self.num_groups for v in base]
        return averaged_soln

    def test_groups(self, round):
        if round == 0:
            for idx in range(self.num_groups):
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()
                train_loss, train_acc, test_acc = self.arrange_stats(stats_train, stats)

                self.log_groups[idx].append([round, train_loss, train_acc, test_acc])
                tqdm.write('Group {} at round {} accuracy: {}'.format(idx, round, test_acc))  # testing accuracy

        for idx,g in enumerate(self.groups):
            stats = self.test_group_model(self.groups[g]["model"])  # have set the latest model for all clients
            stats_train = self.train_error_and_loss()
            train_loss, train_acc, test_acc = self.arrange_stats(stats_train, stats)

            self.log_groups[idx].append([round, train_loss, train_acc, test_acc])
            tqdm.write('Group {} at round {} accuracy: {}'.format(self.groups[g]["id"],round, test_acc))  # testing accuracy
            tqdm.write('Group {} at round {} train accuracy: {}'.format(self.groups[g]["id"],round, train_acc))  # testing accuracy

    def test_group_model(self, group_model):
        '''tests group model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(group_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def aggregate_avg(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += v.astype(np.float64)
        averaged_soln = [v / len(wsolns) for v in base]

        return averaged_soln

    def arrange_stats(self,stats_train, stats):
        train_loss = np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])
        train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])

        return train_loss, train_acc, test_acc