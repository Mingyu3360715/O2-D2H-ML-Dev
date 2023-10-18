# Codes for BDT trainings and working point optimisation to be used in D2H
## Requirements
In order to execute the code in this folder, the following python libraries are needed:
- [hipe4ml](https://github.com/hipe4ml/hipe4ml)
- [hipe4ml_converter](https://github.com/hipe4ml/hipe4ml_converter)
- [alive_progress](https://github.com/rsalmei/alive-progress)

All configurables needed for sample preparation and ML training (resp. working point optimisation) are embedded in a YAML configuration file such as `config_training_DplusToPiKPi.yml` (resp. `config_optimisation_DplusToPiKPi.yml`) . In the following, we give this file an arbitrary name: `config.yml`.

## Training
### Download training samples from Hyperloop
Download the derived `AO2D.root` (produced via a treeCreator task) and save in a dedicated directory (one directory per workflow). These directories will be used as labeled inputs for ML training.

### Comment on the configurables

### Prepare samples
In order to prepare the samples the following script can be used:
```python3 prepare_samples.py config.yml```

### Perform training
In order to perform the training and produce the BDT models to be used in D2H, the following script can be used:
```python3 train_d2h.py config.yml```
Given the output directory set in `config.yml`, a directory is created for each pT bin (i.e. each model trained) and filled with:
- plots at data preparation level: variables distributions and correlations
- plots at training-testing level: BDT output scores, ROC curves, precision recall, features importance
- trained models in files containing `ModelHandler` prefix (in `pickle` and `onnx` formats). *The `.onnx` can be used for ML inference in O2Physics selection task.*
- model applied to test set in file containing `ModelApplied` suffix

## Working point optimisation
The goal is to perform a scan on BDT output scores while computing relevant quantities such as efficiencies (preselections and BDT), yield fractions, expected signal, expected background, allowing to build expected signal-over-background and expected significance.

### Required input
To do so, the needed input gathers the:
- preselections acceptance-times-efficiency (i.e. without cuts on BDT scores): this can be obtained using `efficiency.py` macro;
- pT-differential cross section predicted by FONLL, both for prompt and non-prompt: see [this](http://www.lpthe.jussieu.fr/~cacciari/fonll/fonllform.html) for instance.

### Perform ML output scan
The following script can be used:
```python3 optimisation.py config.yml```
Given the output directory set in `config.yml`, a single directory is created for all pT bins (i.e. for all models trained) and filled with:
- a `.root` file containing information on background fits, significance distributions as a function of other quantities of interest, quantities of interest distributions as a function of ML output score(s), with a `TDirectory` set for each model;
- a canvas containing quantities of interest distributions as a function of ML output score(s), with configurable extensions (`.pdf` by default).



