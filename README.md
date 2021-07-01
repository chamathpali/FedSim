# FedSim
_Similarity Guided Model Aggregation for Federated Learning_

## Usage
```shell
bash run_fedsim.sh DATASET_NAME DROP_PERC NUM_CLUSTERS NUM_CLIENTS Run_Name
bash run_fedsim.sh mnist 0 9 20 mnist_run
```

FedSim algorithm implementation is available in `flearn/trainers/fedsim.py`.

## Reproduce results

The experiments performed on all the datasets were carried out with 35 random seeds (from 0 to 34 incremented by 1) to empirically demonstrate the significance. Repetition of the same experiment with different random seeds helps to reduce the sampling error of our experiments.

For a single dataset, run rounds of FedSim, FedAvg and FedProx where each run will generate a single folder in the `logs` folder, as a reference we have added a sample log folder with the results in `logs/sample/`

Used hyper parameters for the experiments are presented in Table 2.

Once the experiments are completed, create the summary log files with the 3 methods as in `results/` folder. We have added our results in here which can help to refer and use FedSim.

1. Figure 3 - Results on real datasets - `plot_fedsim_main.py`

2. Figure 5 - Accuracy improvements of FedSim - `plot_fedsim_improvements.py`

3. Figure 6 - Results on synthetic datasets - `plot_fedsim_main.py` change line #123 to `if(True)`

4. Figure 7 - Results on other learning models - `plot_fedsim_other.py`


## Experiment setup
We have adapted the experiment setup from [FedProx](https://github.com/litian96/FedProx) and [Leaf Benchmark](https://github.com/TalwalkarLab/leaf) work. Thanks for the support by [Tian Li](https://github.com/litian96).

### Dataset generation

For all datasets, see the `README` files in separate `data/$dataset` folders for instructions on preprocessing and/or sampling data.

For further clarifications follow the guides on [FedProx](https://github.com/litian96/FedProx) and [Leaf](https://github.com/TalwalkarLab/leaf)

The two datasets produced with this work is published with the generation source code.
- [Fed-Mex](https://github.com/chamathpali/Fed-MEx/)
- [Fed-Goodreads](https://github.com/chamathpali/Fed-Goodreads/)

### Downloading dependencies

```
pip3 install -r requirements.txt  
```
### Run FedSim Experiments

```shell
bash run_fedsim.sh DATASET_NAME 0 NUM_CLUSTERS NUM_CLIENTS Run_Name
bash run_fedavg.sh mnist 0 9 20 mnist_run
```
or direcly use the python command
```shell
python3  -u main.py --dataset='goodreads' --optimizer='fedsim' --learning_rate=0.0001 
--num_rounds=250 --clients_per_round=20 --eval_every=1 --batch_size=10 --num_epochs=10 
--model='rnn' --drop_percent=0 --num_groups=11 --ex_name=goodreads_rnn_0 --seed=0
```

When running on GPU specify the id and then run the experiments
```
export CUDA_VISIBLE_DEVICES=GPU_ID
```
