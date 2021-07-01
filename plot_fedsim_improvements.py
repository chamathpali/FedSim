# USED TO GENERATE THE compare_results_real.pdf GRAPH

import json
import os
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

color_avg ="#ff7f0e"
color_prox ="#13CA91"
color_sim ="#17becf"

color_avg ="#ff7f0e"
color_prox ="#fb99bc"
color_sim ="#17becf"
linewidth = 1.8

ROUNDS = 101

rounds = [i for i in range(ROUNDS)]

# RANDOM COMMPARISION GRAPH
# dataset = "logs/FINAL/random/"
# sim_rounds = []
# sim_test_acc = []
# random_rounds = []
# random_test_acc = []
# with open(dataset+'original.csv',mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             line_count += 1
#         sim_rounds.append(float(row["round"]))
#         sim_test_acc.append(float(row["test_acc"]))
#         line_count += 1
#     print(f'FedAvg log Processed {line_count} lines.')
#
# with open(dataset + 'random.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             line_count += 1
#         random_rounds.append(float(row["round"]))
#         random_test_acc.append(float(row["test_acc"]))
#         line_count += 1
#     print(f'FedAvg log Processed {line_count} lines.')
#
#
#     fig, ax = plt.subplots(1, 1, figsize=[5, 3])
#     # linewidth = 1.2
#     linewidth = 1
#
#     ax.plot(rounds, sim_test_acc, linewidth=linewidth, color=color_sim, label="FedSim")
#     ax.plot(rounds, random_test_acc,"--", alpha=0.7, linewidth=linewidth, color="#000000", label="FedSim Random")
#     # ax.set_title("FEMNIST Comparision ",fontweight='bold')
#
#     plt.subplots_adjust(hspace=0.5)
#     ax.set_xlabel("# Rounds")
#     ax.set_ylabel('Test Accuracy')
#
#     for i in range(1):
#         # ax[i].set_xlabel("# Rounds")
#
#         ax.spines['bottom'].set_color('#dddddd')
#         ax.spines['top'].set_color('#dddddd')
#         ax.spines['right'].set_color('#dddddd')
#         ax.spines['left'].set_color('#dddddd')
#         ax.tick_params(color='#dddddd')
#         ax.set_xlim(0, ROUNDS)
#     plt.legend(frameon=False, loc='lower right',
#                 prop=dict(weight='normal'),
#                )  # note: different from plt.legend
#
#     # print(all)
#
#     # ax[0].set_title("Synthetic", fontsize=22)
#     plt.tight_layout()
#     # plt.subplots_adjust(bottom=0.2)
#
#     plt.show()
#     fig.savefig("random_compare.pdf")
#
# exit(0)

# datasets = ["IID","00", "2525","0505","7575" ,"11", "mnist","femnist", "mex"]
# datasets = ["IID","00","2525", "0505","7575", "11"]
datasets = ["mnist","femnist", "mex","goodreads"]
all = {}
for ds in datasets:
    dataset = "docs/analysis/outputs/"+ds+".csv"
    avg_rounds = []
    avg_test_acc = []
    prox_rounds = []
    prox_test_acc = []
    sim_rounds = []
    sim_test_acc = []
    with open(dataset,
              mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            avg_rounds.append(float(row["round"]))
            avg_test_acc.append(float(row["avg"]))
            prox_rounds.append(float(row["round"]))
            prox_test_acc.append(float(row["prox"]))
            sim_rounds.append(float(row["round"]))
            sim_test_acc.append(float(row["sim"]))

            line_count += 1
        # print(f'FedAvg log Processed {line_count} lines.')

    all[ds] = {"avg": [], "sim": [], "prox": []}
    all[ds]["avg"]  = avg_test_acc
    all[ds]["sim"]  = sim_test_acc
    all[ds]["prox"] = prox_test_acc

    all[ds]["diff_avg"]  = (np.array(sim_test_acc) - np.array(avg_test_acc)) *100
    all[ds]["diff_prox"] = (np.array(sim_test_acc) - np.array(prox_test_acc)) *100



# PRINT ALL SYNTHETIC
if(False):
    # fig, ax = plt.subplots(2, 3, figsize=[7.8,8.5])
    fig, ax = plt.subplots(2, 3, figsize=[10,6])
    ax[0,0].plot(rounds, all["IID"]["sim"], linewidth=linewidth, color=color_sim, label="FedSim")
    ax[0,0].plot(rounds, all["IID"]["avg"],":", linewidth=linewidth, color=color_avg, label="FedAvg")
    ax[0,0].plot(rounds, all["IID"]["prox"],"-.",  linewidth=linewidth, color=color_prox, label="FedProx")
    ax[0,0].set_title("Synthetic IID", fontweight='bold')

    ax[0,1].plot(rounds, all["00"]["sim"], linewidth=linewidth, color=color_sim)
    ax[0,1].plot(rounds, all["00"]["avg"],":", linewidth=linewidth, color=color_avg)
    ax[0,1].plot(rounds, all["00"]["prox"],"-.",  linewidth=linewidth, color=color_prox)
    ax[0,1].set_title("Synthetic(0,0)", fontweight='bold')

    ax[0,2].plot(rounds, all["2525"]["sim"], linewidth=linewidth, color=color_sim)
    ax[0,2].plot(rounds, all["2525"]["avg"], ":", linewidth=linewidth, color=color_avg)
    ax[0,2].plot(rounds, all["2525"]["prox"],"-.", linewidth=linewidth, color=color_prox)
    ax[0,2].set_title("Synthetic(0.25,0.25)", fontweight='bold')


    ax[1,0].plot(rounds, all["0505"]["sim"], linewidth=linewidth, color=color_sim)
    ax[1,0].plot(rounds, all["0505"]["avg"],":",  linewidth=linewidth, color=color_avg)
    ax[1,0].plot(rounds, all["0505"]["prox"],"-.", linewidth=linewidth, color=color_prox)
    ax[1,0].set_title("Synthetic(0.5,0.5)", fontweight='bold')
    #
    ax[1,1].plot(rounds, all["7575"]["sim"], linewidth=linewidth, color=color_sim)
    ax[1,1].plot(rounds, all["7575"]["avg"],":", linewidth=linewidth, color=color_avg)
    ax[1,1].plot(rounds, all["7575"]["prox"],"-.",  linewidth=linewidth, color=color_prox)
    ax[1,1].set_title("Synthetic(0.75,0.75)", fontweight='bold')

    ax[1,2].plot(rounds, all["11"]["sim"], linewidth=linewidth, color=color_sim)
    ax[1,2].plot(rounds, all["11"]["avg"], ":", linewidth=linewidth, color=color_avg)
    ax[1,2].plot(rounds, all["11"]["prox"],"-.", linewidth=linewidth, color=color_prox)
    ax[1,2].set_title("Synthetic(1,1)", fontweight='bold')

    # ax[0,2].plot(rounds, all["mnist"]["sim"], linewidth=linewidth, color=color_sim)
    # ax[0,2].plot(rounds, all["mnist"]["avg"], alpha=0.6, linewidth=linewidth, color=color_avg)
    # ax[0,2].plot(rounds, all["mnist"]["prox"], alpha=0.6, linewidth=linewidth, color=color_prox)
    # ax[0,2].set_title("MNIST", fontweight='bold')
    #
    # ax[1,2].plot(rounds, all["femnist"]["sim"], linewidth=linewidth, color=color_sim)
    # ax[1,2].plot(rounds, all["femnist"]["avg"], alpha=0.6, linewidth=linewidth, color=color_avg)
    # ax[1,2].plot(rounds, all["femnist"]["prox"], alpha=0.6, linewidth=linewidth, color=color_prox)
    # ax[1,2].set_title("FEMNIST", fontweight='bold')


    # plt.subplots_adjust(hspace=0.2)

    for i in range(3):
        for j in range(2):
            if(False):
                ax[j,i].set_xlabel("# Rounds")
                ax[j,i].set_ylabel('Test Accuracy')

            # ax[j,i].set_xlabel("# Rounds")
            # ax[j,i].set_ylabel('Test Accuracy')
            ax[j, i].spines['bottom'].set_color('#dddddd')
            ax[j, i].spines['top'].set_color('#dddddd')
            ax[j, i].spines['right'].set_color('#dddddd')
            ax[j, i].spines['left'].set_color('#dddddd')
            ax[j, i].tick_params(color='#dddddd')
            ax[j, i].set_xlim(0, ROUNDS)
    fig.legend(frameon=False, loc='lower center',
               bbox_to_anchor=(0.53, 0),
               ncol=3, prop=dict(weight='normal'), borderaxespad=-0.3)  # note: different from plt.legend
    ax[0, 0].set_xlabel("# Rounds")
    ax[0, 0].set_ylabel('Test Accuracy')
    ax[1, 0].set_xlabel("# Rounds")
    ax[1, 0].set_ylabel('Test Accuracy')

    # print(all)

    # ax[0].set_title("Synthetic", fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)
    plt.subplots_adjust(left=0.055, wspace=0.13)

    plt.show()
    fig.savefig("full_results_synthetic.pdf")
else:
    fig, ax = plt.subplots(2, 2, figsize=[10, 6])
    # linewidth = 1.2

    # ax[0,0].plot([i for i in range(31)], all["mnist"]["diff_avg"], linewidth=linewidth, color=color_sim, label="FedSim")
    ax[0,0].plot([i for i in range(31)], all["mnist"]["diff_avg"],":", alpha=1, linewidth=linewidth, color=color_avg, label="(FedSim - FedAvg)")
    ax[0,0].plot([i for i in range(31)], all["mnist"]["diff_prox"], "-.", alpha=1, linewidth=linewidth, color=color_prox, label="(FedSim - FedProx)")
    ax[0,0].set_title("MNIST",fontweight='bold')
    # ax[0,0].axvspan(0, 1, alpha=0.2, color='#787878')
    ax[0,0].set_xlim(0, 31)
    # ax[0,0].set_ylim(0.2, 0.92)
    ax[0,0].axvspan(0, 0, alpha=0.2, color='#787878')

    # ax[0,1].plot([i for i in range(501)], all["femnist"]["sim"], linewidth=linewidth, color=color_sim)
    ax[0,1].plot([i for i in range(501)], all["femnist"]["diff_avg"],":", alpha=1, linewidth=linewidth, color=color_avg)
    ax[0,1].plot([i for i in range(501)], all["femnist"]["diff_prox"], "-.", alpha=1, linewidth=linewidth, color=color_prox)
    ax[0,1].set_title("FEMNIST", fontweight='bold')
    ax[0,1].set_xlim(0, 501)
    ax[0,1].axvspan(0, 1, alpha=0.2, color='#787878')

    # ax[0,1].set_ylim(0.3, 0.83)


    # ax[1,0].plot([i for i in range(201)], all["mex"]["sim"], linewidth=linewidth, color=color_sim)
    ax[1,0].plot([i for i in range(201)], all["mex"]["diff_avg"],":", alpha=1, linewidth=linewidth, color=color_avg)
    ax[1,0].plot([i for i in range(201)], all["mex"]["diff_prox"], "-.", alpha=1, linewidth=linewidth, color=color_prox)
    ax[1,0].set_title("Fed-MEx", fontweight='bold')
    ax[1,0].set_xlim(0, 201)
    ax[1,0].axvspan(0, 37, alpha=0.2, color='#787878')

    # ax[1,0].set_ylim(0.6, 0.96)


    # ax[1,1].plot([i for i in range(251)], all["goodreads"]["sim"], linewidth=linewidth, color=color_sim)
    ax[1,1].plot([i for i in range(251)], all["goodreads"]["diff_avg"],":", alpha=1, linewidth=linewidth, color=color_avg)
    ax[1,1].plot([i for i in range(251)], all["goodreads"]["diff_prox"], "-.", alpha=1, linewidth=linewidth, color=color_prox)
    ax[1,1].set_title("Fed-Goodreads", fontweight='bold')
    ax[1,1].set_xlim(0, 251)
    # [0, 1, 2, 3, 4, 5, 12, 15, 16, 17, 24, 25, 40, 106, 122, 140]
    ax[1,1].axvspan(0, 5, alpha=0.2, color='#787878')
    ax[1,1].axvspan(12, 12, alpha=0.2, color='#787878')
    ax[1,1].axvspan(15, 17, alpha=0.2, color='#787878')
    ax[1,1].axvspan(24, 25, alpha=0.2, color='#787878')
    ax[1,1].axvspan(40, 40, alpha=0.2, color='#787878')
    ax[1,1].axvspan(106, 106, alpha=0.2, color='#787878')
    ax[1,1].axvspan(122, 122, alpha=0.2, color='#787878')
    ax[1,1].axvspan(140, 140, alpha=0.2, color='#787878')

    # ax[1,1].set_ylim(0.53, 0.63)


    plt.subplots_adjust(hspace=0.5)
    ax[0,0].set_xlabel("# Rounds")
    ax[0,0].set_ylabel('Test Acc. Improvement (%)')
    ax[1,0].set_xlabel("# Rounds")
    ax[1,0].set_ylabel('Test Acc. Improvement (%)')

    for i in range(2):
        for j in range(2):
            ax[j, i].spines['bottom'].set_color('#dddddd')
            ax[j, i].spines['top'].set_color('#dddddd')
            ax[j, i].spines['right'].set_color('#dddddd')
            ax[j, i].spines['left'].set_color('#dddddd')
            ax[j, i].tick_params(color='#dddddd')
            ax[j, i].axhline(y=0, color='r', linestyle='-')

            # ax[j, i].set_xlim(0, ROUNDS)

    # for i in range(3):
    #     # ax[i].set_xlabel("# Rounds")
    #
    #     ax[ i].spines['bottom'].set_color('#dddddd')
    #     ax[ i].spines['top'].set_color('#dddddd')
    #     ax[ i].spines['right'].set_color('#dddddd')
    #     ax[ i].spines['left'].set_color('#dddddd')
    #     ax[i].tick_params(color='#dddddd')
    # ax[ i].set_xlim(0, ROUNDS)

    fig.legend(frameon=False, loc='lower center', ncol=3, prop=dict(weight='normal', size=13),
               borderaxespad=-0.3)  # note: different from plt.legend

    # print(all)

    # ax[0].set_title("Synthetic", fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)
    plt.subplots_adjust(left=0.062, wspace=0.08)

    plt.show()
    fig.savefig("compare_results_real.pdf")

exit(0)


def graph_print(method, params, num_groups = 1, name="non"):
    sim_rounds = []
    sim_test_acc = []
    avg_rounds = []
    avg_test_acc = []
    prox_rounds = []
    prox_test_acc = []
    groups = []
    logdir = "logs/"+name

    with open(logdir + '/fed_'+method + '_' + str(num_groups) + '.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            sim_rounds.append(float(row["round"]))
            sim_test_acc.append(float(row["test_acc"]))
            line_count += 1
        print(f'Fed Processed {line_count} lines.')

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
            print(f'FedSim Groups {line_count} lines.')

    with open('fedavg_original/'+str(params["dataset"])+'_'+str(params["clients_per_round"])+'.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            avg_rounds.append(float(row["round"]))
            avg_test_acc.append(float(row["test_acc"]))
            line_count += 1
        print(f'FedAvg log Processed {line_count} lines.')

    with open('fedavg_original/'+str(params["dataset"])+'_'+str(params["clients_per_round"])+'_prox.csv', mode='r') as csv_file:
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


    # fig.showfig.show()()
    fig.savefig(logdir+"/fed"+method+"_acc_"+str(num_groups)+".pdf")
    plt.close(fig)

