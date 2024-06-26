# Development Pipeline for MMR-GNN
## Authors
	Nicholas Majeske
## Contact
	nmajeske@iu.edu or azad@iu.edu

## Cloning
The pipeline contains many submodules (git repos) for various ML models and some data processing. Cloning must include the --recurse-submodules flag to pull the complete pipeline.

```bash
git clone git@github.com:HipGraph/MMR-GNN_Dev.git --recurse-submodules
```

## Environment
The pipeline uses Anaconda and pip to manage dependencies. Issue the following to create the environment:
```bash
conda create -n MMR-GNN_Dev python=3.9
```
Pipeline dependencies are handled via setup.py. Run the following to install all python packages and integrate submodules (correct imports, etc.):
```bash
conda activate MMR-GNN_Dev
python setup.py install_dependencies
python setup.py integrate_submodules
```
## Datasets
All datasets are hosted on NERSC at [datasets](https://portal.nersc.gov/project/m4012/data/). Each dataset is organized into several files for its different modalities with the data file structure of this repository appearing as follows:
```bash
data
└───LittleRiver
│   │   Spatial.csv             # static node features
│   │   Spatiotemporal.csv.gz   # dynamic node features
│   │   Graph.csv               # dependency structure
│   │   TemporalLabels.csv      # time-step labels
│   │   SpatialLabels.csv       # node labels
│   │   variables.py            # dataset variables module
│   
└───WabashRiver
│   │   Spatial.csv             # static node features
│   │   Spatiotemporal.csv.gz   # dynamic node features
│   │   ...
│   │   variables.py            # dataset variables module
│   
└───...
│   
│   __init__.py                 # module 1
│   dataset.py                  # module 2
│   ...                         # ...
```
Before running any experiments, all data files (.csv and .csv.gz) must be added to the associated dataset folder. For example, to run any experiment on Little River the files Spatial.csv, Spatiotemporal.csv.gz, Graph.csv, TemporalLabels.csv, and SpatialLabels.csv must be downloaded and added to data/LittleRiver/ of the repository.

### Original Dataset Sources
The datasets provided above represent our integrated versions converted and generated by routines defined in data/integration.py and data/generation.py. The following commands may be issued to recreate any of the datasets from their original source:
```bash
python driver.py --I \"LittleRiver\"
python driver.py --I \"WabashRiver\"
python driver.py --I \"METR_LA\"
python driver.py --I \"PREMS_BAY\"
python driver.py --I \"E_METR_LA\"  # downloads+collates Caltrans PeMS files
python driver.py --G \"E_METR_LA\"  # generates Graph.csv
python driver.py --I \"E_PEMS_BAY\" # downloads+collates Caltrans PeMS files
python driver.py --G \"E_PEMS_BAY\" # generates Graph.csv
python driver.py --I \"Solar_Energy\"
python driver.py --I \"Electricity\"
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
python driver.py --E \"pakdd2024\" --e 3.1  # Solar-Energy
python driver.py --E \"pakdd2024\" --e 3.2  # Electricity
```
Results are cached under experiment/PAKDD2024/[experiment name]/. For example, results from Little River will be under experiments/PAKDD2024/LittleRiver/. Sub-directories *Checkpoints* and *Evaluations* contain model checkpoint and evaluation results respectively.

Once completed, results may be automatically collected into csv and LaTeX tables via their associated analysis class. Issue any of the following to generate the result table for a dataset:
```bash
python driver.py --A \"pakdd2024\" --a 1.1  # Little River
python driver.py --A \"pakdd2024\" --a 1.2  # Wabash River
python driver.py --A \"pakdd2024\" --a 2.1  # METR-LA
python driver.py --A \"pakdd2024\" --a 2.11 # E-METR-LA
python driver.py --A \"pakdd2024\" --a 2.2  # PEMS-BAY
python driver.py --A \"pakdd2024\" --a 2.21 # E-PEMS-BAY
python driver.py --A \"pakdd2024\" --a 3.1  # Solar-Energy
python driver.py --A \"pakdd2024\" --a 3.2  # Electricity
```
You will find the resulting tables under analysis/PAKDD2024/[analysis name]/. For example, results from Little River will be under analysis/PAKDD2024/LittleRiver/.

### Ablation Experiments
Many ablation studies exist for MMR-GNN. The following experiments cover all published ablation studies:
```bash
python driver.py --E \"pakdd2024\" --e 0.2   # stGRU vs. vanilla GRU
python driver.py --E \"pakdd2024\" --e 0.31  # implicit modality count
python driver.py --E \"pakdd2024\" --e 0.4   # graph augmentation rate
python driver.py --E \"pakdd2024\" --e 0.5   # increasing embeding dimension
python driver.py --E \"pakdd2024\" --e 0.102 # long-term Wabash River forecasting
python driver.py --E \"pakdd2024\" --e 0.107 # long-term Solar-Energy forecasting
python driver.py --E \"pakdd2024\" --e 0.7   # alternative RNN cells
python driver.py --E \"pakdd2024\" --e 0.8   # alternative fusion methods
python driver.py --E \"pakdd2024\" --e 0.9   # various clustering algorithms
```
Refer to the name of each experiment (defined in function name()) for location of results.

## Citation
```
@inproceedings{majeske2024multi,
  title={Multi-modal Recurrent Graph Neural Networks for Spatiotemporal Forecasting},
  author={Majeske, Nicholas and Azad, Ariful},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={144--157},
  year={2024},
  organization={Springer}
}
```
