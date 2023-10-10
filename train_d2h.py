#!/usr/bin/env python3

"""
file: train_d2h.py
brief: script for the training of ML models to be used in D2H
note: inspired by EventFiltering/PWGHF/Macros/train_hf_triggers.py and Run2 macros
usage: python3 train_d2h.py CONFIG
author: Alexandre Bigot <alexandre.bigot@cern.ch>, Strasbourg University
author: Mingyu Zhang <mingyu.zang@cern.ch>
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from hipe4ml import plot_utils
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml_converter.h4ml_converter import H4MLConverter
from sklearn.model_selection import train_test_split

MAX_BKG_FRAC = 0.4 # max of bkg fraction to keep for training

#pylint: disable= too-many-instance-attributes, too-few-public-methods
class MlTrainer:
    """
    Class for ML training and testing
    """

    def __init__(self, config):
        self.channel = config['channel']
        self.seed_split = config['seed_split']
        self.labels = config['labels']

        pt_bins_limits = config['data_prep']['pt_bins_limits']
        self.pt_bins = [[a, b] for a, b in zip(pt_bins_limits[:-1], pt_bins_limits[1:])]
        self.extension = config['plots']['extension']

        self.indirs = config['data_prep']['indirs']
        self.binary = True if self.indirs['Nonprompt'] is None else False
        self.outdir = config['output']['dir']
        self.share = config['data_prep']['class_balance']['share']
        self.bkg_factor = config['data_prep']['class_balance']['bkg_factor']
        self.test_frac = config['data_prep']['test_fraction']
        self.vars_to_draw = config['plots']['extra_columns'] + config['ml']['training_vars']
        self.name_pt_var = config['data_prep']['name_pt_var']

        self.raw_output = config['ml']['raw_output']
        self.roc_auc_approach = config['ml']['roc_auc_approach']
        self.roc_auc_average = config['ml']['roc_auc_average']
        self.score_metric = 'roc_auc'
        self.training_vars = config['ml']['training_vars']
        self.hyper_pars = config['ml']['hyper_pars']
        self.hyper_pars_opt = config['ml']['hyper_pars_opt']

        self.column_to_save_list = config['output']['column_to_save_list']
        self.log_file = config['output']['log_file']

        self.file_lists = {}

    def __check_input_consistency(self):
        """
        Helper method to check self consistency of inputs
        """

        # class balance
        if self.share not in ('equal','all_signal'):
            print(f'\033[91mERROR: class_balance option {self.share} not implemented\033[0m')
            sys.exit()
        if self.share == 'all_signal' and len(self.bkg_factor) != len(self.pt_bins):
            print('\033[91mERROR: bkg_factor must be defined for each pT bin!\033[0m')
            sys.exit()
        # training
        if not isinstance(self.training_vars, list):
            print('\033[91mERROR: training columns must be defined!\033[0m')
            sys.exit()
        # hyper-parameters options
        if not isinstance(self.hyper_pars, list):
            print('\033[91mERROR: hyper-parameters must be defined ' \
                  'or be a list containing an empty dict!\033[0m')
            sys.exit()
        if not isinstance(self.hyper_pars[0], dict):
            print('\033[91mERROR: hyper-parameters must be a list of dict!\033[0m')
            sys.exit()
        if len(self.hyper_pars) != len(self.pt_bins):
            print('\033[91mERROR: hyper-parameters definition does not match pT binning!\033[0m')
            sys.exit()
        if not isinstance(self.hyper_pars_opt['hyper_par_ranges'], dict):
            print('\033[91mERROR: hyper_pars_opt_config must be defined!\033[0m')
            sys.exit()
        if not self.binary:
            if not (self.roc_auc_average in ['macro', 'weighted']
                    and self.roc_auc_approach in ['ovo', 'ovr']):
                print('\033[91mERROR: selected ROC configuration is not valid!\033[0m')
                sys.exit()

            if self.roc_auc_average == 'weighted':
                self.score_metric += f'_{self.roc_auc_approach}_{self.roc_auc_average}'
            else:
                self.score_metric += f'_{self.roc_auc_approach}'

    def __fill_list_input_files(self):
        """
        Helper method to fill a dictionary with lists of input files for each class
        """

        file_lists = {}
        for cand_type in self.indirs:
            file_lists[cand_type] = []
            if self.indirs[cand_type] is None:
                continue
            for indir in self.indirs[cand_type]:
                file = os.path.join(
                        indir, f"{cand_type}_{self.channel}.parquet.gzip")
                if os.path.isfile(file):
                    file_lists[cand_type].append(file)
                else:
                    print('\033[91mERROR: missing file, did you prepare the samples?\n' \
                          'If not, do python3 prepare_samples.py CONFIG\033[0m')
                    sys.exit()

        self.file_lists = file_lists


    def __get_sliced_dfs(self):
        """
        Helper method to get pT-sliced dataframes for each class
        """

        print('Loading and preparing data files: ...', end='\r')

        self.__fill_list_input_files()

        hdl_bkg = TreeHandler(self.file_lists['Bkg'])
        hdl_prompt = TreeHandler(self.file_lists['Prompt'])
        hdl_nonprompt = None if self.binary else TreeHandler(self.file_lists['Nonprompt'])

        hdl_prompt.slice_data_frame(self.name_pt_var, self.pt_bins, True)
        if not self.binary:
            hdl_nonprompt.slice_data_frame(self.name_pt_var, self.pt_bins, True)
        hdl_bkg.slice_data_frame(self.name_pt_var, self.pt_bins, True)

        print('Loading and preparing data files: Done!')
        return hdl_bkg, hdl_prompt, hdl_nonprompt

    #pylint: disable=too-many-statements, too-many-branches, too-many-arguments, too-many-locals
    def __data_prep(self, df_bkg, df_prompt, df_nonprompt,
                    pt_bin, out_dir, bkg_factor):
        """
        Helper method for pt-dependent data preparation
        """

        n_prompt = len(df_prompt)
        n_nonprompt = len(df_nonprompt)
        n_bkg = len(df_bkg)
        if self.binary:
            log_available_cands = f"\nNumber of available candidates " \
                f"in {pt_bin[0]} < pT < {pt_bin[1]} GeV/c: \n   " \
                f"Signal: {n_prompt}\n   Bkg: {n_bkg}"
        else:
            log_available_cands = f"\nNumber of available candidates " \
                f"in {pt_bin[0]} < pT < {pt_bin[1]} GeV/c: \n   " \
                f"Prompt: {n_prompt}\n   Nonprompt: {n_nonprompt}\n   Bkg: {n_bkg}"
        print(log_available_cands)

        if self.share == 'equal':
            if self.binary:
                n_cand_min = min([n_prompt, n_bkg])
                bkg_fraction = n_cand_min/n_bkg
                n_bkg = n_prompt = n_cand_min
            else:
                n_cand_min = min([n_prompt, n_nonprompt, n_bkg])
                bkg_fraction = n_cand_min/n_bkg
                n_bkg = n_prompt = n_nonprompt = n_cand_min
            log_share = '\nKeep the same number of candidates for each class, ' \
                'chosen as the minimal number of candidates among all classes.'

        elif self.share == 'all_signal':
            n_cand_bkg = int(min(
                [n_bkg, (n_prompt + n_nonprompt) * bkg_factor]))
            if self.binary:
                log_share = f'\nKeep all signal and use {n_cand_bkg} bkg candidates ' \
                    f'for training and testing ({1 - self.test_frac}-{self.test_frac})'
            else:
                log_share = f'\nKeep all prompt and nonprompt and use {n_cand_bkg} ' \
                    'bkg candidates for training and testing ' \
                    f'({1 - self.test_frac}-{self.test_frac})'
            bkg_fraction = n_cand_bkg/n_bkg
            n_bkg = n_cand_bkg

        else:
            print(f'\033[91mERROR: class_balance option {self.share} not implemented\033[0m')
            sys.exit()

        print(log_share)
        print(f'Fraction of bkg candidates used for ML: {100*bkg_fraction:.2f}%')
        if (1-self.test_frac)*bkg_fraction > MAX_BKG_FRAC:
            print(f'\033[93mWARNING: using more than {100*MAX_BKG_FRAC:.0f}% ' \
                  'of bkg available for training, not good!\033[0m')

        if self.binary:
            log_training_cands = "\nNumber of candidates used for training and testing: \n   " \
                        f"Signal: {n_prompt}\n   Bkg: {n_bkg}\n"
        else:
            log_training_cands = "\nNumber of candidates used for training and testing: \n   " \
                        f"Prompt: {n_prompt}\n   Nonprompt: {n_nonprompt}\n   Bkg: {n_bkg}\n"

        print(log_training_cands)

        # write logs in log file
        with open(os.path.join(out_dir, self.log_file), "w", encoding="utf-8") as file:
            file.write(log_available_cands)
            file.write(log_share)
            file.write(log_training_cands)

        df_tot = pd.concat(
            [df_bkg[:n_bkg],
            df_prompt[:n_prompt],
            df_nonprompt[:n_nonprompt]],
            sort=True
        )

        labels_array = np.array([0]*n_bkg + [1]*n_prompt + [2]*n_nonprompt)
        if 0 < self.test_frac < 1:
            train_set, test_set, y_train, y_test = train_test_split(
                df_tot, labels_array, test_size=self.test_frac, random_state=self.seed_split
            )
        else:
            print("ERROR: test_fraction must belong to ]0,1[")
            sys.exit(0)

        train_test_data = [train_set, y_train, test_set, y_test]
        del df_tot # release memory

        # safety
        if len(np.unique(train_test_data[3])) != len(self.labels):
            print('\033[91mERROR: The number of labels defined does not match' \
                  'the number of classes! \nCheck the CONFIG file\033[0m')
            sys.exit()

        # plots
        df_list = [df_bkg, df_prompt] if self.binary else [df_bkg, df_prompt, df_nonprompt]

        #_____________________________________________
        plot_utils.plot_distr(df_list, self.vars_to_draw, 100, self.labels, figsize=(12, 7),
                            alpha=0.3, log=True, grid=False, density=True)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        for ext in self.extension:
            plt.savefig(f'{out_dir}/DistributionsAll_pT_{pt_bin[0]}_{pt_bin[1]}.{ext}')
        plt.close('all')
        #_____________________________________________
        corr_matrix_fig = plot_utils.plot_corr(df_list, self.vars_to_draw, self.labels)
        for fig, lab in zip(corr_matrix_fig, self.labels):
            plt.figure(fig.number)
            plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
            for ext in self.extension:
                fig.savefig(f'{out_dir}/CorrMatrix{lab}_pT_{pt_bin[0]}_{pt_bin[1]}.{ext}')

        return train_test_data

    #pylint: disable=too-many-statements, too-many-branches
    def __train_test(self, train_test_data, hyper_pars, pt_bin, out_dir):
        '''
        Helper method for model training and testing
        '''

        n_classes = len(np.unique(train_test_data[3]))
        model_clf = xgb.XGBClassifier(use_label_encoder=False)
        model_hdl = ModelHandler(model_clf, self.training_vars, hyper_pars)

        # hyperparams optimization
        if self.hyper_pars_opt['activate']:
            print('Performing optuna hyper-parameters optimisation: ...', end='\r')

            with open(os.path.join(out_dir, self.log_file), 'a',
                    encoding="utf-8") as file:
                file.write('\nOptuna hyper-parameters optimisation:')
                sys.stdout = file
                model_hdl.optimize_params_optuna(
                    train_test_data,
                    self.hyper_pars_opt['hyper_par_ranges'],
                    cross_val_scoring=self.score_metric,
                    timeout=self.hyper_pars_opt['timeout'],
                    n_jobs=self.hyper_pars_opt['njobs'],
                    n_trials=self.hyper_pars_opt['ntrials'],
                    direction='maximize'
                )
            sys.stdout = sys.__stdout__
            print('Performing optuna hyper-parameters optimisation: Done!')
            print(f'Optuna hyper-parameters:\n{model_hdl.get_model_params()}')
        else:
            model_hdl.set_model_params(hyper_pars)

        # store final hyperparameters in info file
        with open(os.path.join(out_dir, self.log_file), "a", encoding="utf-8") as file:
            file.write(f"\nModel hyperparameters:\n {model_hdl.get_model_params()}")

        # train and test the model with the updated hyper-parameters
        y_pred_test = model_hdl.train_test_model(
            train_test_data,
            True,
            output_margin=self.raw_output,
            average=self.roc_auc_average,
            multi_class_opt=self.roc_auc_approach
        )

        y_pred_train = model_hdl.predict(train_test_data[0], self.raw_output)

        # Save applied model to test set
        test_set_df = train_test_data[2]
        test_set_df = test_set_df.loc[:, self.column_to_save_list]
        test_set_df['Labels'] = train_test_data[3]

        for pred, lab in enumerate(self.labels):
            if self.binary:
                test_set_df['ML_output'] = y_pred_test
            else:
                test_set_df[f'ML_output_{lab}'] = y_pred_test[:, pred]

        test_set_df.to_parquet(f'{out_dir}/{self.channel}_ModelApplied' \
                               f'_pT_{pt_bin[0]}_{pt_bin[1]}.parquet.gzip')

        # save model
        if os.path.isfile(f"{out_dir}/ModelHandler_{self.channel}.pickle"):
            os.remove(f"{out_dir}/ModelHandler_{self.channel}.pickle")
        if os.path.isfile(f"{out_dir}/ModelHandler_onnx_{self.channel}.onnx"):
            os.remove(f"{out_dir}/ModelHandler_onnx_{self.channel}.onnx")

        model_hdl.dump_model_handler(f'{out_dir}/ModelHandler_{self.channel}' \
                                     f'_pT_{pt_bin[0]}_{pt_bin[1]}.pickle')
        model_conv = H4MLConverter(model_hdl)
        model_conv.convert_model_onnx(1)
        model_conv.dump_model_onnx(f'{out_dir}/ModelHandler_onnx_{self.channel}' \
                                   f'_pT_{pt_bin[0]}_{pt_bin[1]}.onnx')

        #plots
        #_____________________________________________
        plt.rcParams["figure.figsize"] = (10, 7)
        fig_ml_output = plot_utils.plot_output_train_test(
            model_hdl,
            train_test_data,
            80,
            self.raw_output,
            self.labels,
            True,
            density=True
        )
        if n_classes > 2:
            for fig, lab in zip(fig_ml_output, self.labels):
                for ext in self.extension:
                    fig.savefig(f'{out_dir}/MLOutputDistr{lab}_pT_{pt_bin[0]}_{pt_bin[1]}.{ext}')
        else:
            for ext in self.extension:
                fig_ml_output.savefig(f'{out_dir}/MLOutputDistr_pT_{pt_bin[0]}_{pt_bin[1]}.{ext}')
        #_____________________________________________
        plt.rcParams["figure.figsize"] = (10, 9)
        fig_roc_curve = plot_utils.plot_roc(
            train_test_data[3],
            y_pred_test,
            None,
            self.labels,
            self.roc_auc_average,
            self.roc_auc_approach
        )
        for ext in self.extension:
            fig_roc_curve.savefig(f'{out_dir}/ROCCurveAll_pT_{pt_bin[0]}_{pt_bin[1]}.{ext}')
        pickle.dump(fig_roc_curve, open(f'{out_dir}/ROCCurveAll_pT_{pt_bin[0]}_{pt_bin[1]}.pkl',
                                        'wb')) 
        #_____________________________________________
        plt.rcParams["figure.figsize"] = (10, 9)
        fig_roc_curve_tt = plot_utils.plot_roc_train_test(
            train_test_data[3],
            y_pred_test,
            train_test_data[1],
            y_pred_train,
            None,
            self.labels,
            self.roc_auc_average,
            self.roc_auc_approach
        )

        fig_roc_curve_tt.savefig(f'{out_dir}/ROCCurveTrainTest_pT_{pt_bin[0]}_{pt_bin[1]}.pdf')
        #_____________________________________________
        precision_recall_fig = plot_utils.plot_precision_recall(train_test_data[3],
                                                                y_pred_test, self.labels)
        precision_recall_fig.savefig(f'{out_dir}/PrecisionRecallAll_pT_{pt_bin[0]}_{pt_bin[1]}.pdf')
        #_____________________________________________
        plt.rcParams["figure.figsize"] = (12, 7)
        fig_feat_importance = plot_utils.plot_feature_imp(
                train_test_data[2][train_test_data[0].columns],
                train_test_data[3],
                model_hdl,
                self.labels
            )
        n_plot = n_classes if n_classes > 2 else 1
        for i_fig, fig in enumerate(fig_feat_importance):
            if i_fig < n_plot:
                lab = self.labels[i_fig] if n_classes > 2 else ''
                for ext in self.extension:
                    fig.savefig(f'{out_dir}/FeatureImportance_{lab}_{self.channel}.{ext}')
            else:
                for ext in self.extension:
                    fig.savefig(f'{out_dir}/FeatureImportanceAll_{self.channel}.{ext}')

    def process(self):
        """
        Process function of the class, performing data preparation,
        training, testing, saving the model and important plots
        """

        self.__check_input_consistency()
        df_bkg, df_prompt, df_nonprompt = self.__get_sliced_dfs()

        for i_pt, pt_bin in enumerate(self.pt_bins):
            print(f'\n\033[94mStarting ML analysis --- {pt_bin[0]} < pT < {pt_bin[1]} GeV/c\033[0m')

            out_dir_pt = os.path.join(os.path.expanduser(self.outdir),
                                    f'pt{pt_bin[0]}_{pt_bin[1]}')
            if os.path.isdir(out_dir_pt):
                print((f'\033[93mWARNING: Output directory \'{out_dir_pt}\' already exists,'
                    ' overwrites possibly ongoing!\033[0m'))
            else:
                os.makedirs(out_dir_pt)

            df_pt_nonprompt = pd.DataFrame() if self.binary else df_nonprompt.get_slice(i_pt)
            if self.share == 'all_signal':
                bkg_factor = self.bkg_factor[i_pt]
            else:
                bkg_factor = None

            train_test_data = self.__data_prep(df_bkg.get_slice(i_pt), df_prompt.get_slice(i_pt),
                                               df_pt_nonprompt, pt_bin, out_dir_pt,
                                               bkg_factor)
            self.__train_test(train_test_data, self.hyper_pars[i_pt], pt_bin, out_dir_pt)

def main(config):
    """
    Main function

    Parameters
    -----------------
    - config: dictionary with config read from a yaml file
    """
    MlTrainer(config).process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('config', metavar='text', default='config.yml',
                        help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.config, "r", encoding="utf-8") as yml_cfg:
        cfg = yaml.load(yml_cfg, yaml.FullLoader)
    print('Loading analysis configuration: Done!')

    main(cfg)
