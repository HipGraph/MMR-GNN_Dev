# Development Pipeline for MMR-GNN
## Authors
	Nicholas Majeske
## Contact
	nmajeske@iu.edu

## Cloning
The pipeline contains many submodules (git repos) for various ML models and some data processing. Cloning must include the --recurse-submodules flag to pull the complete pipeline.

```bash
git clone git@github.com:HipGraph/MMR-GNN_Dev.git --recurse-submodules
```

## Environment
The pipeline uses Anaconda and pip to manage dependencies. Issue the following to create the environment:
```bash
conda create -n MMRGNN python=3.9
```
The environment may be installed using setup.py. Run the following to install all python packages and integrate submodules (correct imports, etc.):
```bash
conda activate MMRGNN
python setup.py install_dependencies
python setup.py integrate_submodules
```
##  Experiments
All experimental settings are defined in experiment/pakdd2024.py. The following commands execute the pipeline for various experiments:

### Main Experiments
Main experiments train and test all models for 10 trials (initialization seeds) on a dataset.
```bash
python driver.py --E \"pakdd2024\" --e 1.1  # Little River
python driver.py --E \"pakdd2024\" --e 1.2  # Wabash River
python driver.py --E \"pakdd2024\" --e 2.1  # METR-LA
python driver.py --E \"pakdd2024\" --e 2.11 # E-METR-LA
python driver.py --E \"pakdd2024\" --e 2.2  # PEMS-BAY
python driver.py --E \"pakdd2024\" --e 2.21 # E-PEMS-BAY
python driver.py --E \"pakdd2024\" --e 2.1  # Solar-Energy
python driver.py --E \"pakdd2024\" --e 2.1  # Electricity
```
Results are cached under experiment/PAKDD2024/[experiment name]/. For example, results from Little River will be under experiments/PAKDD2024/LittleRiver/. Sub-directories 'Checkpoints' and 'Evaluations' contain model checkpoint and evaluation results respectively.

Once completed, results may be automatically collected into csv and latex tables via their associated analysis'. Issue any of the following to generate the result table for a dataset:
```bash
python driver.py --A \"pakdd2024\" --a 1.1  # Little River
python driver.py --A \"pakdd2024\" --a 1.2  # Wabash River
python driver.py --A \"pakdd2024\" --a 2.1  # METR-LA
python driver.py --A \"pakdd2024\" --a 2.11 # E-METR-LA
python driver.py --A \"pakdd2024\" --a 2.2  # PEMS-BAY
python driver.py --A \"pakdd2024\" --a 2.21 # E-PEMS-BAY
python driver.py --A \"pakdd2024\" --a 2.1  # Solar-Energy
python driver.py --A \"pakdd2024\" --a 2.1  # Electricity
```
You will find the resulting tables under analysis/PAKDD2024/[analysis name]/. For example, results from Little River will be under analysis/PAKDD2024/LittleRiver/.

### Ablation Experiments
Many ablation studies exist for MMR-GNN. An exhaustive list includes:
```bash
python driver.py --E \"pakdd2024\" --e 0.2  # stGRU vs. vanilla GRU
python driver.py --E \"pakdd2024\" --e 0.31  # Implicit modality count
python driver.py --E \"pakdd2024\" --e 0.4  # Graph augmentation rate
python driver.py --E \"pakdd2024\" --e 0.5  # Increasing embeding dimension
python driver.py --E \"pakdd2024\" --e 0.107  # Long-term Solar-Energy forecasting
python driver.py --E \"pakdd2024\" --e 0.7  # Alternative RNN cells
python driver.py --E \"pakdd2024\" --e 0.8  # Alternative fusion methods
python driver.py --E \"pakdd2024\" --e 0.9  # Various clustering algorithms
```
Refer to the name of each experiment (defined in function name()) for location of results.
