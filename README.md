# Tackling Uncertainties in Multi-Agent Reinforcement Learning through Integration of Agent Termination Dynamics
The source code containing the implementation of Tackling Uncertainties in Multi-Agent Reinforcement Learning through Integration of Agent Termination Dynamics accepted at AAMAS 2025

[PyMARL](https://github.com/oxwhirl/pymarl) is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning. We have used the same code-base structure and used the [**QPLEX**: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/pdf/2008.01062) MARL algorithm.

Written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

Set up Environment and install dependencies 
```shell
bash install_dependencies.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The `requirements.txt` file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
python src/main.py --config=dmix_cbf --env-config=sc2 with env_args.map_name=5m_vs_6m
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `results` folder.

## Saving and loading models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep.
