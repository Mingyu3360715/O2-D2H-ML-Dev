"""
Script for the training of ML models to be used in HF triggers
It requires hipe4ml and hipe4ml_converter to be installed:
  pip install hipe4ml
  pip install hipe4ml_converter

\author Fabrizio Grosa <fabrizio.grosa@cern.ch>, CERN
\author Alexandre Bigot <alexandre.bigot@cern.ch>, Strasbourg University
\author Biao Zhang <biao.zhang@cern.ch>, CCNU
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yaml

from hipe4ml import plot_utils
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml_converter.h4ml_converter import H4MLConverter

def get_list_input_files(indirs, channel):
    """
    function that returns the list of files

    Parameters
    -----------------
    - indirs: dictionary with lists of input directories for prompt, nonprompt, and bkg
    - channel: decay channel, options:
        D0ToKPi, DPlusToPiKPi, DsToKKPi, LcToPKPi, XicToPKPi

    Outputs
    -----------------
    - file_lists: dictionary with lists of input files for prompt, nonprompt, and bkg
    """

    if channel not in ["D0ToKPi", "DplusToPiKPi", "DsToKKPi", "LcToPKPi", "XicToPKPi"]:
        print(f"ERROR: channel {channel} not implemented, return None")
        return {"Prompt": None, "Nonprompt": None, "Bkg": None}

    file_lists = {}
    for cand_type in indirs:  # pylint: disable=too-many-nested-blocks
        file_lists[cand_type] = []
        for indir in indirs[cand_type]:
            subdirs = os.listdir(indir)
            for subdir in subdirs:
                subdir = os.path.join(indir, subdir)
                if os.path.isdir(subdir):
                    for subsubdir in os.listdir(subdir):
                        subsubdir = os.path.join(subdir, subsubdir)
                        if os.path.isdir(subsubdir):
                            file = os.path.join(
                                subsubdir, f"{cand_type}_{channel}.parquet.gzip")
                            if os.path.isfile(file):
                                file_lists[cand_type].append(file)
                            else:
                                for subsubsubdir in os.listdir(subsubdir):
                                    subsubsubdir = os.path.join(subsubdir, subsubsubdir)
                                    if os.path.isdir(subsubsubdir):
                                        file = os.path.join(
                                            subsubsubdir, f"{cand_type}_{channel}.parquet.gzip")
                                        if os.path.isfile(file):
                                            file_lists[cand_type].append(file)

    return file_lists

def data_prep(inputConfig, iBin, ptBin, outputDirPt, promptDf, fdDf, bkgDf): #pylint: disable=too-many-statements, too-many-branches
    """
    function for data preparation
    """
    nPrompt = len(promptDf)
    nFD = len(fdDf)
    nBkg = len(bkgDf)
    if fdDf.empty:
        out = f'\n     Signal: {nPrompt}\n     Bkg: {nBkg}'
    else:
        out = f'\n     Prompt: {nPrompt}\n     FD: {nFD}\n     Bkg: {nBkg}'
    print(f'Number of available candidates in {ptBin[0]} < pT < {ptBin[1]} GeV/c:{out}')

    datasetOption = inputConfig['data_prep']['class_balance']['option']
    seedSplit = inputConfig['data_prep']['seed_split']
    testFrac = inputConfig['data_prep']['test_fraction']

    if datasetOption == 'equal':
        if fdDf.empty:
            nCandToKeep = min([nPrompt, nBkg])
            out = 'signal'
            out2 = 'signal'
        else:
            nCandToKeep = min([nPrompt, nFD, nBkg])
            out = 'prompt, FD'
            out2 = 'prompt'
        print((f'Keep same number of {out} and background (minimum) for training and '
               f'testing ({1 - testFrac}-{testFrac}): {nCandToKeep}'))
        print(f'Fraction of real data candidates used for ML: {nCandToKeep/nBkg:.5f}')

        if nPrompt > nCandToKeep:
            print((f'Remaining {out2} candidates ({nPrompt - nCandToKeep})'
                   'will be used for the efficiency together with test set'))
        if nFD > nCandToKeep:
            print((f'Remaining FD candidates ({nFD - nCandToKeep}) will be used for the '
                   'efficiency together with test set'))

        TotalDf = pd.concat([bkgDf.iloc[:nCandToKeep], promptDf.iloc[:nCandToKeep], fdDf.iloc[:nCandToKeep]], sort=True)
        if fdDf.empty:
            labelsArray = np.array([0]*nCandToKeep + [1]*nCandToKeep)
        else:
            labelsArray = np.array([0]*nCandToKeep + [1]*nCandToKeep + [2]*nCandToKeep)
        if testFrac < 1:
            trainSet, testSet, yTrain, yTest = train_test_split(TotalDf, labelsArray, test_size=testFrac,
                                                                random_state=seedSplit)
        else:
            trainSet = pd.DataFrame()
            testSet = TotalDf.copy()
            yTrain = pd.Series()
            yTest = labelsArray.copy()

        trainTestData = [trainSet, yTrain, testSet, yTest]
        # promptDfSelForEff = pd.concat([promptDf.iloc[nCandToKeep:], testSet[pd.Series(yTest).array == 1]], sort=True)
        # if fdDf.empty:
        #     fdDfSelForEff = pd.DataFrame()
        # else:
        #     fdDfSelForEff = pd.concat([fdDf.iloc[nCandToKeep:], testSet[pd.Series(yTest).array == 2]], sort=True)
        del TotalDf

    elif datasetOption == 'max_signal':
        nCandBkg = round(inputConfig['data_prep']['class_balance']['bkg_factor'][iBin] * (nPrompt + nFD))
        out = 'signal' if fdDf.empty else 'prompt and FD'
        print((f'Keep all {out} and use {nCandBkg} bkg candidates for training and '
               f'testing ({1 - testFrac}-{testFrac})'))
        if nCandBkg >= nBkg:
            nCandBkg = nBkg
            print('\033[93mWARNING: using all bkg available, not good!\033[0m')
        print(f'Fraction of real data candidates used for ML: {nCandBkg/nBkg:.5f}')

        TotalDf = pd.concat([bkgDf.iloc[:nCandBkg], promptDf, fdDf], sort=True)
        if fdDf.empty:
            labelsArray = np.array([0]*nCandBkg + [1]*nPrompt)
        else:
            labelsArray = np.array([0]*nCandBkg + [1]*nPrompt + [2]*nFD)
        if testFrac < 1:
            trainSet, testSet, yTrain, yTest = train_test_split(TotalDf, labelsArray, test_size=testFrac,
                                                                random_state=seedSplit)
        else:
            trainSet = pd.DataFrame()
            testSet = TotalDf.copy()
            yTrain = pd.Series()
            yTest = labelsArray.copy()

        trainTestData = [trainSet, yTrain, testSet, yTest]
        # promptDfSelForEff = testSet[pd.Series(yTest).array == 1]
        # fdDfSelForEff = pd.DataFrame() if fdDf.empty else testSet[pd.Series(yTest).array == 2]
        del TotalDf

    else:
        print(f'\033[91mERROR: {datasetOption} is not a valid option!\033[0m')
        sys.exit()

    # plots
    varsToDraw = inputConfig['plots']['plotting_columns']
    legLabels = [inputConfig['output']['leg_labels']['Bkg'],
                 inputConfig['output']['leg_labels']['Prompt']]
    if inputConfig['output']['leg_labels']['FD'] is not None:
        legLabels.append(inputConfig['output']['leg_labels']['FD'])
    outputLabels = [inputConfig['output']['out_labels']['Bkg'],
                    inputConfig['output']['out_labels']['Prompt']]
    if inputConfig['output']['out_labels']['FD'] is not None:
        outputLabels.append(inputConfig['output']['out_labels']['FD'])
    listDf = [bkgDf, promptDf] if fdDf.empty else [bkgDf, promptDf, fdDf]
    #_____________________________________________
    plot_utils.plot_distr(listDf, varsToDraw, 100, legLabels, figsize=(12, 7),
                          alpha=0.3, log=True, grid=False, density=True)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(f'{outputDirPt}/DistributionsAll_pT_{ptBin[0]}_{ptBin[1]}.pdf')
    plt.close('all')
    #_____________________________________________
    corrMatrixFig = plot_utils.plot_corr(listDf, varsToDraw, legLabels)
    for fig, lab in zip(corrMatrixFig, outputLabels):
        plt.figure(fig.number)
        plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
        fig.savefig(f'{outputDirPt}/CorrMatrix{lab}_pT_{ptBin[0]}_{ptBin[1]}.pdf')

    return trainTestData #, promptDfSelForEff, fdDfSelForEff

def train_test(inputConfig, ptBin, outputDirPt, trainTestData, iBin): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for model training and testing
    '''
    nClasses = len(np.unique(trainTestData[3]))
    channel = inputConfig["data_prep"]["channel"]
    modelClf = xgb.XGBClassifier(use_label_encoder=False)
    trainCols = inputConfig['ml']['training_vars']
    hyperPars = inputConfig['ml']['hyper_pars'][iBin]
    if not isinstance(trainCols, list):
        print('\033[91mERROR: training columns must be defined!\033[0m')
        sys.exit()
    if not isinstance(hyperPars, dict):
        print('\033[91mERROR: hyper-parameters must be defined or be an empty dict!\033[0m')
        sys.exit()
    modelHandler = ModelHandler(modelClf, trainCols, hyperPars)

    # hyperparams optimization
    if inputConfig['ml']['hyper_pars_opt']['activate']:
        print('Perform optuna optimization')

        optunaConfig = inputConfig['ml']['hyper_pars_opt']['hyper_par_ranges']
        if not isinstance(optunaConfig, dict):
            print('\033[91mERROR: hyper_pars_opt_config must be defined!\033[0m')
            sys.exit()

        if nClasses > 2:
            averageMethod = inputConfig['ml']['roc_auc_average']
            rocMethod = inputConfig['ml']['roc_auc_approach']
            if not (averageMethod in ['macro', 'weighted'] and rocMethod in ['ovo', 'ovr']):
                print('\033[91mERROR: selected ROC configuration is not valid!\033[0m')
                sys.exit()

            if averageMethod == 'weighted':
                metric = f'roc_auc_{rocMethod}_{averageMethod}'
            else:
                metric = f'roc_auc_{rocMethod}'
        else:
            metric = 'roc_auc'

        print('Performing hyper-parameters optimisation: ...', end='\r')
        outFileHyperPars = open(f'{outputDirPt}/HyperParsOpt_pT_{ptBin[0]}_{ptBin[1]}.txt', 'wt')
        sys.stdout = outFileHyperPars
        modelHandler.optimize_params_optuna(trainTestData, optunaConfig, cross_val_scoring='roc_auc_ovo', timeout=inputConfig['ml']['hyper_pars_opt']['timeout'], n_jobs=inputConfig['ml']['hyper_pars_opt']['njobs'], n_trials=inputConfig['ml']['hyper_pars_opt']['ntrials'], direction='maximize')
        outFileHyperPars.close()
        sys.stdout = sys.__stdout__
        print('Performing hyper-parameters optimisation: Done!')
        print(f'Output saved in {outputDirPt}/HyperParOpt_pT_{ptBin[0]}_{ptBin[1]}.txt')
        print(f'Optuna hyper-parameters:\n{modelHandler.get_model_params()}')
    else:
        modelHandler.set_model_params(hyperPars)

    # train and test the model with the updated hyper-parameters
    yPredTest = modelHandler.train_test_model(trainTestData, True, output_margin=inputConfig['ml']['raw_output'],
                                            average=inputConfig['ml']['roc_auc_average'],
                                            multi_class_opt=inputConfig['ml']['roc_auc_approach'])
    yPredTrain = modelHandler.predict(trainTestData[0], inputConfig['ml']['raw_output'])

    # save model handler in pickle
    modelHandler.dump_model_handler(f"{outputDirPt}/ModelHandler_{channel}_pT_{ptBin[0]}_{ptBin[1]}.pickle")
    modelConv = H4MLConverter(modelHandler)
    modelConv.convert_model_onnx(1)
    modelConv.dump_model_onnx(f"{outputDirPt}/ModelHandler_onnx_{channel}_pT_{ptBin[0]}_{ptBin[1]}.onnx")
    # modelConv.convert_model_hummingbird("onnx", 1)
    # modelConv.dump_model_hummingbird(
    #     f"{outputDirPt}/ModelHandler_onnx_hummingbird_{channel}_pT_{ptBin[0]}_{ptBin[1]}")


    #plots
    legLabels = [inputConfig['output']['leg_labels']['Bkg'],
                 inputConfig['output']['leg_labels']['Prompt']]
    if inputConfig['output']['leg_labels']['FD'] is not None:
        legLabels.append(inputConfig['output']['leg_labels']['FD'])
    outputLabels = [inputConfig['output']['out_labels']['Bkg'],
                    inputConfig['output']['out_labels']['Prompt']]
    if inputConfig['output']['out_labels']['FD'] is not None:
        outputLabels.append(inputConfig['output']['out_labels']['FD'])
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 7)
    outputFigML = plot_utils.plot_output_train_test(modelHandler, trainTestData, 80, inputConfig['ml']['raw_output'],
                                                    legLabels, inputConfig['plots']['train_test_log'], density=True)
    if nClasses > 2:
        for fig, lab in zip(outputFigML, outputLabels):
            fig.savefig(f'{outputDirPt}/MLOutputDistr{lab}_pT_{ptBin[0]}_{ptBin[1]}.pdf')
    else:
        outputFigML.savefig(f'{outputDirPt}/MLOutputDistr_pT_{ptBin[0]}_{ptBin[1]}.pdf')
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    rocCurveFig = plot_utils.plot_roc(trainTestData[3], yPredTest, None, legLabels, inputConfig['ml']['roc_auc_average'],
                                      inputConfig['ml']['roc_auc_approach'])
    rocCurveFig.savefig(f'{outputDirPt}/ROCCurveAll_pT_{ptBin[0]}_{ptBin[1]}.pdf')
    pickle.dump(rocCurveFig, open(f'{outputDirPt}/ROCCurveAll_pT_{ptBin[0]}_{ptBin[1]}.pkl', 'wb'))
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    rocCurveTTFig = plot_utils.plot_roc_train_test(trainTestData[3], yPredTest, trainTestData[1], yPredTrain, None,
                                                   legLabels, inputConfig['ml']['roc_auc_average'],
                                                   inputConfig['ml']['roc_auc_approach'])
    rocCurveTTFig.savefig(f'{outputDirPt}/ROCCurveTrainTest_pT_{ptBin[0]}_{ptBin[1]}.pdf')
    #_____________________________________________
    precisionRecallFig = plot_utils.plot_precision_recall(trainTestData[3], yPredTest, legLabels)
    precisionRecallFig.savefig(f'{outputDirPt}/PrecisionRecallAll_pT_{ptBin[0]}_{ptBin[1]}.pdf')
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (12, 7)
    # featuresImportanceFig = plot_utils.plot_feature_imp(trainTestData[2][trainTestData[0].columns], trainTestData[3], modelHandler,
    #                                                     legLabels)
    # fig_feat_importance = plot_utils.plot_feature_imp(
    #     # train_test_data[2][train_test_data[0].columns],
    #     train_test_data[2][training_vars],
    #     train_test_data[3],
    #     model_hdl,
    #     leg_labels
    # nPlot = nClasses if nClasses > 2 else 1
    # for iFig, fig in enumerate(featuresImportanceFig):
    #     if iFig < nPlot:
    #         label = outputLabels[iFig] if nClasses > 2 else ''
    #         fig.savefig(f'{outputDirPt}/FeatureImportance{label}_pT_{ptBin[0]}_{ptBin[1]}.pdf')
    #     else:
    #         fig.savefig(f'{outputDirPt}/FeatureImportanceAll_pT_{ptBin[0]}_{ptBin[1]}.pdf')

    return modelHandler

def apply(inputConfig, ptBin, outputDirPt, modelHandler, dataDfPtSel, promptDfPtSelForEff, fdDfPtSelForEff):
    outputLabels = [inputConfig['output']['out_labels']['Bkg'],
                    inputConfig['output']['out_labels']['Prompt']]
    if inputConfig['output']['out_labels']['FD'] is not None:
        outputLabels.append(inputConfig['output']['out_labels']['FD'])
    print('Applying ML model to prompt dataframe: ...', end='\r')
    yPredPromptEff = modelHandler.predict(promptDfPtSelForEff, inputConfig['ml']['raw_output'])
    dfColumnToSaveList = inputConfig['apply']['column_to_save_list']
    if not isinstance(dfColumnToSaveList, list):
        print('\033[91mERROR: df_column_to_save_list must be defined!\033[0m')
        sys.exit()
    if 'fM' not in dfColumnToSaveList:
        print('\033[93mWARNING: inv_mass is not going to be saved in the output dataframe!\033[0m')
    if 'fPt' not in dfColumnToSaveList:
        print('\033[93mWARNING: pt_cand is not going to be saved in the output dataframe!\033[0m')
    promptDfPtSelForEff = promptDfPtSelForEff.loc[:, dfColumnToSaveList]
    if fdDfPtSelForEff.empty:
        out = 'Signal'
        promptDfPtSelForEff['ML_output'] = yPredPromptEff
    else:
        out = 'Prompt'
        for pred, lab in enumerate(outputLabels):
            promptDfPtSelForEff[f'ML_output_{lab}'] = yPredPromptEff[:, pred]
    promptDfPtSelForEff.to_parquet(f'{outputDirPt}/{out}_pT_{ptBin[0]}_{ptBin[1]}_ModelApplied.parquet.gzip')
    print('Applying ML model to prompt dataframe: Done!')

    if not fdDfPtSelForEff.empty:
        print('Applying ML model to FD dataframe: ...', end='\r')
        yPredFDEff = modelHandler.predict(fdDfPtSelForEff, inputConfig['ml']['raw_output'])
        fdDfPtSelForEff = fdDfPtSelForEff.loc[:, dfColumnToSaveList]
        for pred, lab in enumerate(outputLabels):
            fdDfPtSelForEff[f'ML_output_{lab}'] = yPredFDEff[:, pred]
        fdDfPtSelForEff.to_parquet(f'{outputDirPt}/FD_pT_{ptBin[0]}_{ptBin[1]}_ModelApplied.parquet.gzip')
        print('Applying ML model to FD dataframe: Done!')

    print('Applying ML model to data dataframe: ...', end='\r')
    yPredData = modelHandler.predict(dataDfPtSel, inputConfig['ml']['raw_output'])
    dfColumnToSaveListData = dfColumnToSaveList
    # if 'pt_B' in df_column_to_save_list_data:
    #     df_column_to_save_list_data.remove('pt_B') # only in MC
    dataDfPtSel = dataDfPtSel.loc[:, dfColumnToSaveListData]
    if fdDfPtSelForEff.empty:
        dataDfPtSel['ML_output'] = yPredData
    else:
        for pred, lab in enumerate(outputLabels):
            dataDfPtSel[f'ML_output_{lab}'] = yPredData[:, pred]
    dataDfPtSel.to_parquet(f'{outputDirPt}/Data_pT_{ptBin[0]}_{ptBin[1]}_ModelApplied.parquet.gzip')
    print('Applying ML model to data dataframe: Done!')

def main(): #pylint: disable=too-many-statements
    """
    Main function

    """
    # read config file
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    parser.add_argument("--train", help="perform only training and testing", action="store_true")
    parser.add_argument("--apply", help="perform only application", action="store_true")
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlConfigFile:
        inputConfig = yaml.load(ymlConfigFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')

    print('Loading and preparing data files: ...', end='\r')
    promptHandler = TreeHandler(inputConfig['input']['prompt'], inputConfig['input']['treename'])
    fdHandler = None if inputConfig['input']['FD'] is None else TreeHandler(inputConfig['input']['FD'],
                                                                         inputConfig['input']['treename'])
    dataHandler = TreeHandler(inputConfig['input']['data'], inputConfig['input']['treename'])

    if inputConfig['data_prep']['filt_bkg_mass']:
        bkgHandler = dataHandler.get_subset(inputConfig['data_prep']['filt_bkg_mass'], frac=1.,
                                            rndm_state=inputConfig['data_prep']['seed_split'])
    else:
        bkgHandler = dataHandler

    ptBins = [[a, b] for a, b in zip(inputConfig['pt_ranges']['min'], inputConfig['pt_ranges']['max'])]
    promptHandler.slice_data_frame('fPt', ptBins, True)
    if fdHandler is not None:
        fdHandler.slice_data_frame('fPt', ptBins, True)
    dataHandler.slice_data_frame('fPt', ptBins, True)
    bkgHandler.slice_data_frame('fPt', ptBins, True)
    print('Loading and preparing data files: Done!')

    for iBin, ptBin in enumerate(ptBins):
        print(f'\n\033[94mStarting ML analysis --- {ptBin[0]} < pT < {ptBin[1]} GeV/c\033[0m')

        outputDirPt = os.path.join(os.path.expanduser(inputConfig['output']['dir']), f'pt{ptBin[0]}_{ptBin[1]}')
        if os.path.isdir(outputDirPt):
            print((f'\033[93mWARNING: Output directory \'{outputDirPt}\' already exists,'
                   ' overwrites possibly ongoing!\033[0m'))
        else:
            os.makedirs(outputDirPt)

        # data preparation
        #_____________________________________________
        fdDfPt = pd.DataFrame() if fdHandler is None else fdHandler.get_slice(iBin)
        # trainTestData, promptDfSelForEff, fdDfSelForEff = data_prep(inputConfig, iBin, ptBin, outputDirPt,
                                                                    # promptHandler.get_slice(iBin), fdDfPt,
                                                                    # bkgHandler.get_slice(iBin))
        trainTestData = data_prep(inputConfig, iBin, ptBin, outputDirPt,
                                                                    promptHandler.get_slice(iBin), fdDfPt,
                                                                    bkgHandler.get_slice(iBin))
        if args.apply and inputConfig['data_prep']['test_fraction'] < 1.:
            print('\033[93mWARNING: Using only a fraction of the MC for the application! Are you sure?\033[0m')

        # training, testing
        #_____________________________________________
        if not args.apply:
            modelHandler = train_test(inputConfig, ptBin, outputDirPt, trainTestData, iBin)
        else:
            modelList = inputConfig['ml']['saved_models']
            modelPath = modelList[iBin]
            if not isinstance(modelPath, str):
                print('\033[91mERROR: path to model not correctly defined!\033[0m')
                sys.exit()
            modelPath = os.path.expanduser(modelPath)
            print(f'Loaded saved model: {modelPath}')
            modelHandler = ModelHandler()
            modelHandler.load_model_handler(modelPath)

        # model application
        #_____________________________________________
        if not args.train:
            apply(inputConfig, ptBin, outputDirPt, modelHandler, dataHandler.get_slice(iBin),
                 promptHandler.get_slice(iBin), fdDfPt)

        # delete dataframes to release memory
        # for data in trainTestData:
        #     del data
        # del promptDfSelForEff, fdDfSelForEff

main()
