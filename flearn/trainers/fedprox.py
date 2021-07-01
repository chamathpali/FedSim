import datetime
import time

import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from utils import csv_log
from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])

        # Setup Log
        self.params_log = params
        # self.run_name = str(params["ex_name"])+"_fedprox_"+ str(datetime.datetime.now().strftime("%m%d-%H%M%S"))
        self.run_name = str(params["ex_name"])+"_fedprox"
        self.log_main = []
        csv_log.log_start('prox',params,1, self.run_name)

        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        elapsed = []

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
                train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
                train_loss = np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])

                tqdm.write('At round {} accuracy: {}'.format(i, test_acc ))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i,train_acc ))
                tqdm.write('At round {} training loss: {}'.format(i,train_loss ))

                self.log_main.append([i, train_loss, train_acc, test_acc])

            start_time = time.time()

            model_len = process_grad(self.latest_model).size
            global_grads = np.zeros(model_len)
            client_grads = np.zeros(model_len)
            num_samples = []
            local_grads = []

            for c in self.clients:
                num, client_grad = c.get_grads(model_len)
                local_grads.append(client_grad)
                num_samples.append(num)
                global_grads = np.add(global_grads, client_grad * num)
            global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

            difference = 0
            for idx in range(len(self.clients)):
                difference += np.sum(np.square(global_grads - local_grads[idx]))
            difference = difference * 1.0 / len(self.clients)
            tqdm.write('gradient difference: {}'.format(difference))

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)

            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.latest_model, self.client_model)

            for idx, c in enumerate(selected_clients.tolist()):
                # communicate the latest model
                c.set_params(self.latest_model)

                total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)

                # solve minimization locally
                if c in active_clients:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                else:
                    #soln, stats = c.solve_iters(num_iters=np.random.randint(low=1, high=total_iters), batch_size=self.batch_size)
                    soln, stats = c.solve_inner(num_epochs=np.random.randint(low=1, high=self.num_epochs), batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)
        
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)
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
        csv_log.write_all('prox', self.log_main, [], 1, self.run_name)
        csv_log.graph_print('prox',self.params_log, 1, self.run_name)

        # tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        # tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
        print("Time Taken Each Round: ")
        print(elapsed)
        print(np.mean(elapsed))
        csv_log.write_time_taken(elapsed, self.run_name)
