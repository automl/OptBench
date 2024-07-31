# OptBench

Optimization tasks:
Synthetic functions: Ackley, Levy, Schwefel

Compatible with [carps](https://github.com/automl/CARP-S)

## Installation
```bash
# Create conda env
conda create -n optbench python=3.12 -c conda-forge

# Activate env
conda activate optbench

# Clone repo
git clone git@github.com:automl/OptBench.git
cd OptBench

# Install 
pip install -e .
```

## Running an optimizer from carps
The command for running an optimizer, e.g., random search from carps on OptBench looks as follows:
```bash
python -m carps.run 'hydra.searchpath=[pkg://optbench/configs]' +optimizer/randomsearch=config +problem/OptBench=synth_Ackley_2 seed=1
```
The breakdown of the command:
- `'hydra.searchpath=[pkg://optbench/configs]'`: Let hydra know where to find the configs of the OptBench package. For this, `optbench` needs to be installed. It is in general: pkg://PACKAGE_NAME/PATH_INSIDE_PACKAGE_TO_CONFIGS_FOLDER .
- `+optimizer/randomsearch=config`: select an optimizer from `carps`. Follows the config folder structure in `carps`. Beware, for other optimizers you need to install dependencies (check the [repo](https://github.com/automl/CARP-S)).
- `+problem/OptBench=synth_Ackley_2`: Select a problem. Follows the configs folder structure in this package, starting from `optbench/configs`.
- `seed=1`: Set the seed to 1. 🙂

Of course, you can also specify the run dir etc.
For more hydra overrides and parallelization, check their [docs/tutorials](https://hydra.cc/docs/advanced/override_grammar/basic/).