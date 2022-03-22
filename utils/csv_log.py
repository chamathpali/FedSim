import json
import os

import csv
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)

def log_start(method, params, num_groups = 1, name="non"):
    logdir = "logs/"+name
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    with open(logdir+"/params" + '.json', 'w') as json_file:
        json.dump(params, json_file)

def write_dataset(arr, name="non"):
    logdir = "logs/"+name
    with open(logdir + '/dataset_analysis.csv', mode='a+', newline='') as log_file:
        writer = csv.DictWriter(log_file, fieldnames=arr[0].keys())
        writer.writeheader()
        for data in arr:
            writer.writerow(data)


def write_clusters(arr, name="non"):
    logdir = "logs/"+name
    with open(logdir + '/clusters.csv', mode='a+', newline='') as log_file:
        writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(arr)

def write_all(method, log_data, log_groups, num_groups = 1, name="non"):
    logdir = "logs/"+name
    # with open(logdir + '/fed_' +method +'_'+str(num_groups)+'.csv', mode='w', newline='') as log_file:
    with open(logdir + '/' + name + '.csv', mode='w', newline='') as log_file:
        writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['round', 'train_loss', 'train_acc', 'test_acc'])
        for line in log_data:
            writer.writerow(line)

    for  idx,group in enumerate(log_groups):
        with open(logdir + '/fed_' +method +'_g_'+str(idx)+'.csv', mode='w', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['round', 'train_loss', 'train_acc', 'test_acc'])
            for line in log_groups[idx]:
                writer.writerow(line)

def graph_print(method, params, num_groups = 1, name="non"):
    sim_rounds = []
    sim_test_acc = []
    avg_rounds = []
    avg_test_acc = []
    prox_rounds = []
    prox_test_acc = []
    groups = []
    logdir = "logs/"+name

    with open(logdir + '/' + name + '.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            sim_rounds.append(float(row["round"]))
            sim_test_acc.append(float(row["test_acc"]))
            line_count += 1
        # print(f'Fed Processed {line_count} lines.')

    if method == "sim":
        for i in range(num_groups):
            group_data = []
            with open(logdir + '/fed_sim_g_'+ str(i)+'.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    group_data.append(float(row["test_acc"]))
                    line_count += 1
            groups.append(group_data)
            # print(f'FedSim Groups {line_count} lines.')

    with open('fedavg_original/'+str(params["dataset"])+'_'+str(params["clients_per_round"])+'.csv', mode='r') as csv_file:
    # with open('fedavg_original/seeds/'+str(params["dataset"])+'_'+str(params["seed"])+'_fedavg/'+str(params["dataset"])+'_'+str(params["seed"])+'_fedavg.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            avg_rounds.append(float(row["round"]))
            avg_test_acc.append(float(row["test_acc"]))
            line_count += 1
        # print(f'FedAvg log Processed {line_count} lines.')

    with open('fedavg_original/'+str(params["dataset"])+'_'+str(params["clients_per_round"])+'_prox.csv', mode='r') as csv_file:
    # with open('fedavg_original/seeds/'+str(params["dataset"])+'_'+str(params["seed"])+'_fedprox/'+str(params["dataset"])+'_'+str(params["seed"])+'_fedprox.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            prox_rounds.append(float(row["round"]))
            prox_test_acc.append(float(row["test_acc"]))
            line_count += 1
        # print(f'FedProx log Processed {line_count} lines.')

    fig, ax = plt.subplots(2,1, figsize=[12, 16])

    ax[0].plot(sim_rounds, sim_test_acc, linewidth=3.0, color="#17becf", label="FedSim - G - "+str(num_groups))
    ax[0].plot(avg_rounds, avg_test_acc, ":",alpha=0.6, linewidth=3.0, color="#ff7f0e", label="FedAvg")
    ax[0].plot(prox_rounds, prox_test_acc, "-",alpha=0.6, linewidth=3.0, color="#90C978", label="FedProx")

    ax[1].plot(sim_rounds, sim_test_acc,  "-",linewidth=1.0, alpha=0.8, color="#0000ff", label="FedSim - G - "+str(num_groups))
    ax[1].plot(avg_rounds, avg_test_acc,  "-",linewidth=1.0, alpha=0.8, color="#ff0000", label="Fed Avg")
    for idx,g in enumerate(groups):
        ax[1].plot(sim_rounds, g, linewidth=1.5,  alpha=0.3, label="Group - " + str(idx))

    ax[0].set_xlabel("# Rounds", fontsize=22)
    ax[0].set_ylabel('Testing Accuracy', fontsize=22)
    ax[0].set_title("FedSim comparision - Data:"+str(params["dataset"])
                    + " Clients/round: "+ str(params["clients_per_round"])
                    + " E: " + str(params["num_epochs"])
                    + " Groups: " + str(params["num_groups"]), fontsize=18)

    ax[0].legend(fontsize=22, loc='lower center')
    ax[0].grid()

    ax[1].set_xlabel("# Rounds", fontsize=22)
    ax[1].set_ylabel('Testing Accuracy', fontsize=22)
    ax[1].set_title("Group Accuracies", fontsize=22)
    ax[1].legend(fontsize=22, loc='lower right')
    ax[1].grid()

    # plt.xticks(fontsize=17)
    # plt.yticks(fontsize=17)
    # ax.tick_params(color='#dddddd')
    # ax.spines['bottom'].set_color('#dddddd')
    # ax.spines['top'].set_color('#dddddd')
    # ax.spines['right'].set_color('#dddddd')
    # ax.spines['left'].set_color('#dddddd')

    # fig.showfig.show()()
    # fig.savefig(logdir+"/fed"+method+"_acc_"+str(num_groups)+".pdf")
    fig.savefig(logdir+"/"+name+".pdf")


    plt.close(fig)

def write_time_taken(elapsed, name="non"):
    logdir = "logs/"+name
    with open(logdir + '/timetaken.csv', mode='w', newline='') as log_file:
        writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in elapsed:
            writer.writerow([line])
