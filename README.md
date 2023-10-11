# Code for BDT trainings to be used in D2H
## Requirements
In order to execute the code in this folder, the following python libraries are needed:
- [hipe4ml](https://github.com/hipe4ml/hipe4ml)
- [hipe4ml_converter](https://github.com/hipe4ml/hipe4ml_converter)
- [alive_progress](https://github.com/rsalmei/alive-progress)

## Main steps
### Download training samples from Hyperloop
Download the derived `AO2D.root` (produced via a treeCreator task) and save in a dedicated directory (one directory per workflow). These directories will be used as labeled inputs for ML training.

### Comment on the configurables
All configurables needed for sample preparation and ML training are embedded in a YAML configuration file (such as `config_training_DplusToPiKPi.yml`). In the following, we give this file an arbitrary name: `config.yml`.

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
