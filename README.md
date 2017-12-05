# State Representation Learning with Robotic Priors in PyTorch

Related papers:
- "Learning State Representations with Robotic Priors" (Jonschkowski and Brock, 2015), paper: [http://tinyurl.com/gly9sma](http://tinyurl.com/gly9sma)
- "Unsupervised state representation learning with robotic priors: a robustness benchmark" (Lesort, Seurin et al., 2017), paper: [https://arxiv.org/pdf/1709.05185.pdf](https://arxiv.org/pdf/1709.05185.pdf)

### Config files

#### Base config
Config common to all dataset can found in [configs/default.json](configs/default.json).

#### Dataset config
All dataset must be placed in the `data/` folder.
Each dataset should can contain a `dataset_config.json` file, an example can be found [here](configs/example_dataset_config.json).
This config file describes variables specifics to this dataset.


#### Experiment config
Experiment config file is generated by the `pipeline.py` script. An example can be found [here](configs/example_exp_config.json))

### Launch script
Located [here](launch.sh), it is a shell script that launches multiple grid searches, train the baselines and call the report script.
You have to edit `$data_folder` and make sure of the parameters for knn evaluation before running it:
```
./launch.sh
```

### Pipeline Script
It preprocess data, learn a state representation and evaluate it using knn.

Baxter data used in the paper are not public yet. However you can generate new data using [Baxter Simulator](https://github.com/araffin/arm_scenario_simulator) and [Baxter Experiments](https://github.com/NataliaDiaz/arm_scenario_experiments)

```
python pipeline.py [-h] [-c EXP_CONFIG] [--data_folder DATA_FOLDER]
                   [--base_config BASE_CONFIG]
-c EXP_CONFIG, --exp_config EXP_CONFIG
                     Path to an experiment config file
--data_folder DATA_FOLDER
                     Path to a dataset folder
--base_config BASE_CONFIG
                     Path to overall config file, it contains variables
                     independent from datasets (default:
                     /configs/default.json)
```

#### Examples

Grid search:
```
python pipeline.py --data_folder data/staticButtonSimplest/
```

Reproducing an experiment:
```
python pipeline.py -c path/to/exp_config.json
```


### Preprocessing

```
python -m preprocessing.preprocess [--data_folder DATA_FOLDER] [--mode MODE] [--no-warnings]

--data_folder DATA_FOLDER
                      Dataset folder name
--mode MODE           Preprocessing mode: One of "image_net", "tf".
--no-warnings         disables warnings
```

e.g.
```
python -m preprocessing.preprocess --data_folder staticButtonSimplest
```

### Learn a state representation

Usage:
```
python train.py [--epochs N] [--seed S] [--state_dim STATE_DIM]
                [-bs BATCH_SIZE] [-lr LEARNING_RATE] [--l1_reg L1_REG]
                [--no-cuda] [--no-plots] [--model_type MODEL_TYPE]
                [--data_folder DATA_FOLDER]
                [--log_folder LOG_FOLDER]

--epochs N            number of epochs to train (default: 50)
--seed S              random seed (default: 1)
--state_dim STATE_DIM
                      state dimension (default: 2)
-bs BATCH_SIZE, --batch_size BATCH_SIZE
                      batch_size (default: 256)
-lr LEARNING_RATE, --learning_rate LEARNING_RATE
                      learning rate (default: 0.005)
--l1_reg L1_REG       L1 regularization coeff (default: 0.0)
--no-cuda             disables CUDA training
--no-plots            disables plots
--model_type MODEL_TYPE
                      Model architecture (default: "resnet")
--data_folder DATA_FOLDER
                      Dataset folder
--log_folder LOG_FOLDER
                      Folder within logs/ where the experiment model and plots will be saved

```


Example:
```
python train.py --data_folder data/path/to/dataset
```

### Evaluation and Plotting


#### Create a report
After a report you can create a csv report file using:
```
python evaluation/create_report.py -d logs/nameOfTheDataset/
```
#### Plot a learned representation
You can plot a learned representation with:
```
python plotting/representation_plot.py -i path/to/states_rewards.npz
```

You can also plot ground truth states with:
```
python plotting/representation_plot.py --data_folder path/to/datasetFolder/
```

#### Interactive Plot

You can have an interactive plot of a learned representation using:
```
python plotting/interactive_plot.py --data_folder path/to/datasetFolder/ -i path/to/states_rewards.npz
```
When you click on a state in the representation plot (left click for 2D, **right click for 3D plots**!), it shows the corresponding image along with the reward and the coordinates in the space.

You can also plot ground truth states when you don't specify a npz file:
```
python plotting/interactive_plot.py --data_folder path/to/datasetFolder/
```

#### Create a knn plot and compute KNN-MSE

Usage:
```
python plotting/knn_images.py [-h] --log_folder LOG_FOLDER
                     [--seed SEED] [-k N_NEIGHBORS] [-n N_SAMPLES]

KNN plot and KNN MSE

--log_folder LOG_FOLDER
                      Path to a log folder
--seed SEED           random seed (default: 1)
-k N_NEIGHBORS, --n_neighbors N_NEIGHBORS
                      Number of nearest neighbors (default: 5)
-n N_SAMPLES, --n_samples N_SAMPLES
                      Number of test samples (default: 10)

```

Example:
```
python plotting/knn_images.py --log_folder path/to/an/experiment/log/folder
```

### Baselines

Baseline models are saved in `logs/nameOfTheDataset/baselines/` folder.

#### Supervised learning

Example:
```
python -m baselines.supervised --data_folder path/to/data/folder
```

#### Autoencoder

Gaussian noise is added to the input with a factor `0.1`.

Example:
```
python -m baselines.autoencoder --data_folder path/to/data/folder --state_dim 3 --noise_factor 0.1
```

### Dependencies

Recommended configuration: Ubuntu 16.04 with python 2.7 or 3.6
(should work with python3 though it was only thoroughly tested with python2)


- OpenCV (version >= 2.4)
```
conda install -c menpo opencv3
```
or
```
sudo apt-get install python-opencv (opencv 2.4 - python2)
```

- PyTorch
- PyTorchVision
- Numpy
- Scikit-learn

For plotting:
- matplotlib
- seaborn
- Pillow

For display enhancement:
- termcolor
- tqdm

### Example Data
You can reproduce Rico Jonschkowski's results by downloading npz files from the original [github repository](https://github.com/tu-rbo/learning-state-representations-with-robotic-priors) and placing them in the `data/` folder.

It was tested with the following commit (checkout this one to be sure it will work): [https://github.com/araffin/srl-robotic-priors-pytorch/commit/5175b88a891c240f393b717dd1866435c73ebbda](https://github.com/araffin/srl-robotic-priors-pytorch/commit/5175b88a891c240f393b717dd1866435c73ebbda)

Then run (for example):
```
python main.py --path data/slot_car_task_train.npz
```


## Troubleshooting

### CUDA out of memory error

1.  python train.py --data_folder data/staticButtonSimplest
```
RuntimeError: cuda runtime error (2) : out of memory at /b/wheel/pytorch-src/torch/lib/THC/generic/THCStorage.cu:66
```

SOLUTION 1: CUDA_VISIBLE_DEVICES – Masking GPUs

CUDA_VISIBLE_DEVICES=0 Only Device 0 will be visible
NOTE: Do not set a device id superior to the number of GPU you have, or it will run on your CPU and may freeze your display.


SOLUTION 2: Decrease the batch size, e.g. 32-64 in GPUs with little memory. Warning: computing the priors might not work

SOLUTION 3: Use simple 2-layers neural network model
python train.py --data_folder data/staticButtonSimplest --model_type mlp


### Links
Plugin to work with conda environments and jupyter notebooks:

- [nb_conda plugin](https://docs.anaconda.com/anaconda/user-guide/tasks/use-jupyter-notebook-extensions)
- [nb_conda plugin check](https://stackoverflow.com/questions/37085665/in-which-conda-environment-is-jupyter-executing)
