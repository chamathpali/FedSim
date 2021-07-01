# Used to generate - full_results_real_other.pdf

import json
import os
import csv
import matplotlib.pyplot as plt
import matplotlib

color_avg ="#ff7f0e"
color_prox ="#13CA91"
color_sim ="#17becf"

color_avg ="#ff7f0e"
color_prox ="#fb99bc"
color_sim ="#17becf"
linewidth = 1.8

ROUNDS = 501
rounds = [i for i in range(ROUNDS)]

datasets = ["mnist","femnist", "mex","goodreads"]

all = {}
for ds in datasets:
    dataset = "results/other/"+ds+".csv"
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

    all[ds] = {"avg": [], "sim": [], "prox": []}
    all[ds]["avg"]  = avg_test_acc
    all[ds]["sim"]  = sim_test_acc
    all[ds]["prox"] = prox_test_acc


if(False):
    print(False)
else:
    fig, ax = plt.subplots(2, 2, figsize=[10, 8])
    # linewidth = 1.2

    ax[0,0].plot([i for i in range(31)], all["mnist"]["sim"], linewidth=linewidth, color=color_sim, label="FedSim")
    ax[0,0].plot([i for i in range(31)], all["mnist"]["avg"],":", alpha=1, linewidth=linewidth, color=color_avg, label="FedAvg")
    ax[0,0].plot([i for i in range(31)], all["mnist"]["prox"], "-.", alpha=1, linewidth=linewidth, color=color_prox, label="FedProx")
    ax[0,0].set_title("MNIST - CNN",fontweight='bold')
    ax[0,0].set_xlim(0, 31)
    ax[0,0].set_ylim(0.2, 0.9)

    ax[0,1].plot([i for i in range(501)], all["femnist"]["sim"], linewidth=linewidth, color=color_sim)
    ax[0,1].plot([i for i in range(501)], all["femnist"]["avg"],":", alpha=1, linewidth=linewidth, color=color_avg)
    ax[0,1].plot([i for i in range(501)], all["femnist"]["prox"], "-.", alpha=1, linewidth=linewidth, color=color_prox)
    ax[0,1].set_title("FEMNIST - CNN", fontweight='bold')
    ax[0,1].set_xlim(0, 501)
    ax[0,1].set_ylim(0.6, 0.95)

    #
    ax[1,0].plot([i for i in range(201)], all["mex"]["sim"], linewidth=linewidth, color=color_sim)
    ax[1,0].plot([i for i in range(201)], all["mex"]["avg"],":", alpha=1, linewidth=linewidth, color=color_avg)
    ax[1,0].plot([i for i in range(201)], all["mex"]["prox"], "-.", alpha=1, linewidth=linewidth, color=color_prox)
    ax[1,0].set_title("Fed-MEx - MLP", fontweight='bold')
    ax[1,0].set_xlim(0, 201)
    ax[1,0].set_ylim(0.68, 0.98)
    #
    #
    ax[1,1].plot([i for i in range(251)], all["goodreads"]["sim"], linewidth=linewidth, color=color_sim)
    ax[1,1].plot([i for i in range(251)], all["goodreads"]["avg"],":", alpha=1, linewidth=linewidth, color=color_avg)
    ax[1,1].plot([i for i in range(251)], all["goodreads"]["prox"], "-.", alpha=1, linewidth=linewidth, color=color_prox)
    ax[1,1].set_title("Fed-Goodreads - RNN", fontweight='bold')
    ax[1,1].set_xlim(0, 251)
    ax[1,1].set_ylim(0.45, 0.6)


    plt.subplots_adjust(hspace=0.5)
    ax[0,0].set_xlabel("# Rounds")
    ax[0,0].set_ylabel('Test Accuracy')
    ax[1,0].set_xlabel("# Rounds")
    ax[1,0].set_ylabel('Test Accuracy')

    for i in range(2):
        for j in range(2):
            ax[j, i].spines['bottom'].set_color('#dddddd')
            ax[j, i].spines['top'].set_color('#dddddd')
            ax[j, i].spines['right'].set_color('#dddddd')
            ax[j, i].spines['left'].set_color('#dddddd')
            ax[j, i].tick_params(color='#dddddd')
            # ax[j, i].set_xlim(0, ROUNDS)

    fig.legend(frameon=False, loc='lower center', ncol=3, prop=dict(weight='normal', size=13),
               borderaxespad=-0.3)  # note: different from plt.legend

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)
    plt.subplots_adjust(left=0.061, wspace=0.11)

    plt.show()
    fig.savefig("full_results_real_cnn.pdf")

exit(0)