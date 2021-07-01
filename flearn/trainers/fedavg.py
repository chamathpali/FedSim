import datetime
import time

import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from utils import csv_log
from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

        # Setup Log
        self.params_log = params
        # self.run_name = str(params["ex_name"])+"_fedavg_"+ str(datetime.datetime.now().strftime("%m%d-%H%M%S"))
        self.run_name = str(params["ex_name"])+"_fedavg"
        self.log_main = []
        csv_log.log_start('avg',params,1, self.run_name)

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

                train_loss = np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])
                train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
                test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])

                self.log_main.append([i, train_loss, train_acc, test_acc])

                tqdm.write('At round {} accuracy: {}'.format(i, test_acc ))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i,train_acc ))
                tqdm.write('At round {} training loss: {}'.format(i,train_loss ))

            start_time = time.time()

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            csolns = []  # buffer for receiving client solutions

            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            self.latest_model = self.aggregate(csolns)
            elapsed_time = time.time() - start_time
            elapsed.append(elapsed_time)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)

        test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])

        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, test_acc))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, train_acc))

        self.log_main.append([self.num_rounds, train_loss, train_acc, test_acc])
        csv_log.write_all('avg', self.log_main, [], 1, self.run_name)
        csv_log.graph_print('avg',self.params_log, 1, self.run_name)

        print("Time Taken Each Round: ")
        print(elapsed)
        print(np.mean(elapsed))
        csv_log.write_time_taken(elapsed, self.run_name)

