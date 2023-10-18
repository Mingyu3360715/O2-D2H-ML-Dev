#!/usr/bin/env python3

# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.

# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".

# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

"""
file: optimisation.py
brief: script for the optimisation of the ML model working point
note: adapted from Run2 macros
usage: python3 optimisation.py CONFIG
author: Alexandre Bigot <alexandre.bigot@cern.ch>, Strasbourg University
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import yaml  # pylint: disable=import-error
from alive_progress import alive_bar  # pylint: disable=import-error

# pylint: disable=import-error, no-name-in-module
from ROOT import (
    TF1,
    TH1F,
    TH2F,
    TCanvas,
    TDirectoryFile,
    TFile,
    TLatex,
    TNtuple,
    gROOT,
    kBlack,
    kFullCircle,
    kRainBow,
)

from style_formatter import set_global_style, set_object_style
from utils.fit_utils import single_gaus
from utils.optimisation_utils import (
    get_acc_eff,
    get_cross_sections,
    get_expected_bkg_from_mc,
    get_expected_bkg_from_side_bands,
    get_expected_signal,
    get_fractions_fc,
    get_ml_efficiency,
    load_df_from_parquet,
)

LABEL_BKG = 0
LABEL_PROMPT = 1
LABEL_NONPROMPT = 2

NO_EXTRA_CUT = "Integral"  # default name when no cut applied (other than on ML output)

DUMMY_PAR_CUTS_MIN = -1.0e10
DUMMY_PAR_CUTS_MAX = 1.0e10

# dictionary of quantities estimated for each cut set
EST_NAMES = {
    "Signif": "expected significance",
    "SoverB": "S/B",
    "S": "expected signal",
    "B": "expected background",
    "EffAccPrompt": "(Acc#times#font[152]{e})_{prompt}",
    "EffAccNonprompt": "(Acc#times#font[152]{e})_{nonprompt}",
    "fPrompt": "#it{f}_{ prompt}^{ fc}",
    "fNonprompt": "#it{f}_{ nonprompt}^{ fc}",
}


def enforce_list(x):
    """
    Helper method to enforce list type

    Parameters
    ----------
    - x: a string or a list of string

    Returns
    ----------
    - x_list if x was not a list (and not None), x itself otherwise
    """

    if not isinstance(x, list):
        # handle possible spaces in config file entry
        x = x.split(",")
        for i, element in enumerate(x):
            x[i] = element.strip()

    return x


def enforce_trailing_slash(path):
    """
    Helper method to enforce '/' at the and of directory name

    Parameters
    ----------
    - path: some path

    Returns
    ----------
    - path with a trailing slash at the end if it was not there yet
    """

    if path[-1] != "/":
        path += "/"

    return path


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class MlOutputScanner:
    """
    Class for ML model working point optimisation
    """

    def __init__(self, config, batch):
        self.batch = batch

        # load constant terms
        self.n_ev = config["input"]["n_events"]
        self.sigma_mb = config["sigma_mb"]
        if config["integrated_luminosity"]:
            self.n_exp_ev = self.sigma_mb * config["integrated_luminosity"]
        else:
            self.n_exp_ev = config["n_expected_events"]

        # input files
        self.infiles = config["input"]["filename"]
        self.config_secpeak = config["input"]["secpeak"]

        # pt bins
        pt_bins_limits = enforce_list(config["pt_bins_limits"])
        self.pt_mins = pt_bins_limits[:-1]
        self.pt_maxs = pt_bins_limits[1:]

        # preselection acceptance times efficiency
        self.file_acc_eff_presel = TFile()
        self.config_presel = config["input"]["presel_acc_eff"]
        self.type_acc_eff_presel = self.config_presel["type"]

        # cross sections predictions
        self.file_cross_sec = TFile()
        self.config_cross_sec = config["predictions"]["crosssec"]

        # output files
        self.outdir = enforce_trailing_slash(config["output"]["outdir"])
        self.outfile = self.outdir + config["output"]["filename"]
        self.extension = config["output"]["extension"]
        self.watermark = config["output"]["watermark"]

        # expected signal promptness (prompt or nonprompt)
        self.expected_signal_promptness = config["expected_signal_promptness"]

        # background configuration
        self.bkg_frac_used_for_ml = config["input"]["background"]["fraction_used_for_ml"]
        self.func_fit_bkg = config["input"]["background"]["fit"]["func"]
        self.nsigma_fit_bkg = config["input"]["background"]["fit"]["nsigma"]
        self.mass_min = config["input"]["background"]["fit"]["mass_min"]
        self.mass_max = config["input"]["background"]["fit"]["mass_max"]
        self.bkg_is_mc = config["input"]["background"]["fit"]["is_mc"]
        self.bkg_corr_factor_option = config["input"]["background"]["corrfactor"]
        self.h_bkg_mc_corr_factor = None

        # load possible cuts on df
        self.name_par_cuts = config["dfparametercuts"]["name"]
        self.enable_par_cuts = config["dfparametercuts"]["enable"]
        if self.enable_par_cuts:
            self.par_cuts_min = enforce_list(config["dfparametercuts"]["min"])
            self.par_cuts_max = enforce_list(config["dfparametercuts"]["max"])
        else:
            self.name_par_cuts = NO_EXTRA_CUT
            self.par_cuts_min, self.par_cuts_max = [], []
            self.par_cuts_min.append(DUMMY_PAR_CUTS_MIN)
            self.par_cuts_max.append(DUMMY_PAR_CUTS_MAX)

        # ML output variables for scan
        self.cut_vars = config["cutvars"]
        self.n_scan_ml_vars = len(self.cut_vars)

    def __check_input_consistency(self):
        """
        Helper method to check self consistency of inputs
        """

        if os.path.isdir(self.outdir):
            print(
                (
                    f"\033[93mWARNING: Output directory '{self.outdir}' already exists,"
                    " overwrites possibly ongoing!\033[0m"
                )
            )
        else:
            os.makedirs(self.outdir)

        if self.config_presel["type"] not in ["TH1", "TEfficiency"]:
            print(
                "\033[91mERROR: option set for preselection acceptance times efficiency"
                " object type is not available: choose between TH1 and TEfficiency.\033[0m"
            )
            sys.exit()

        if self.n_scan_ml_vars not in [1, 2]:
            print("\033[91mERROR: can only scan one or two ML output score(s).\033[0m")
            sys.exit()

        if not self.bkg_is_mc and self.bkg_corr_factor_option["filename"]:
            print("\033[91mERROR: set filename of corrfactor to null if bkg is not from MC.\033[0m")
            sys.exit()

        if self.enable_par_cuts and not self.name_par_cuts:
            print("\033[91mERROR: dfparametercuts enabled but no name provided.\033[0m")
            sys.exit()
        del self.enable_par_cuts

        if self.expected_signal_promptness not in ["prompt", "nonprompt"]:
            print("\033[91mERROR: expected signal promptness option not available!\033[0m")
            sys.exit()

    def __complete_bkg_config(self):
        """
        Helper method to complete background configuration
        """

        file = self.bkg_corr_factor_option["filename"]

        if file:
            infile = TFile.Open(file)
            self.h_bkg_mc_corr_factor = infile.Get(self.bkg_corr_factor_option["histoname"])

        del self.bkg_corr_factor_option

    def __configure_output(self, out_file):
        """
        Helper method to configure output (files, plots, ...)

        Parameters
        ----------
        - out_file: output file

        Returns
        ----------
        - out_dir_fit_sidebands: TDirectory for future invariant-mass sidebands fits to be stored
        - out_dir_plot: TDirectory for plots of estimated quantities to be stored
        - t_signif: TNtuple for quantities of interested to be stored
        """

        out_dir_fit_sidebands = TDirectoryFile("SBfits", "SBfits")
        out_dir_fit_sidebands.Write()

        out_file.cd()
        out_dir_plots = TDirectoryFile("plots", "plots")
        out_dir_plots.Write()

        var_names_4_tuple = (
            ":".join(self.cut_vars) + ":PtMin:PtMax:ParCutMin:ParCutMax:"
            "EffAccPromptError:EffAccNonpromptError:SError:BError"
            ":SignifError:SoverBError:" + ":".join(EST_NAMES.keys())
        )
        t_signif = TNtuple("tSignif", "tSignif", var_names_4_tuple)

        return out_dir_fit_sidebands, out_dir_plots, t_signif

    def __configure_scan(self):
        """
        Helper method to configure scan of ML output scores

        Returns
        ----------
        - cut_ranges: ranges of the cuts on ML output score
        - cut_directions: direction of the cuts on ML output score
        - var_names: variable names of ML output score
        """

        cut_ranges, cut_directions, var_names = [], [], []

        for i_pt, _ in enumerate(self.pt_maxs):
            cut_ranges.append([])
            for var in self.cut_vars:
                cut_ranges[i_pt].append(
                    np.arange(
                        self.cut_vars[var]["min"][i_pt],
                        self.cut_vars[var]["max"][i_pt] + self.cut_vars[var]["step"][i_pt] / 10,
                        self.cut_vars[var]["step"][i_pt],
                    ).tolist()
                )
        for var in self.cut_vars:
            cut_directions.append(self.cut_vars[var]["cut_direction"])
            var_names.append(var)

        return cut_ranges, cut_directions, var_names

    def __get_labeled_dfs(self):
        """
        Helper method to get labeled dataframes from input files

        Returns
        ----------
        - df_bkg: background-labeled dataframe
        - df_prompt: prompt-labeled dataframe
        - df_nonprompt: nonprompt-labeled dataframe
        """

        # load dataframes from input files
        df_tot = load_df_from_parquet(self.infiles)
        df_bkg = df_tot.query(f"Labels == {LABEL_BKG}")
        df_prompt = df_tot.query(f"Labels == {LABEL_PROMPT}")
        df_nonprompt = df_tot.query(f"Labels == {LABEL_NONPROMPT}")

        return df_bkg, df_prompt, df_nonprompt

    def __get_labeled_dfs_secpeak(self):
        """
        Helper method to get labeled dataframes from input files (for Ds)

        Returns
        ----------
        - df_prompt_secpeak: prompt-labeled dataframe for second invariant-mass peak
        - df_nonprompt_secpeak: nonprompt-labeled dataframe for second invariant-mass peak
        """

        # load dataframes from input files
        if not self.config_secpeak["activate"]:
            return pd.DataFrame(), pd.DataFrame()

        df_prompt_secpeak = load_df_from_parquet(self.config_secpeak["filename_prompt"])
        df_nonprompt_secpeak = load_df_from_parquet(self.config_secpeak["filename_nonprompt"])

        return df_prompt_secpeak, df_nonprompt_secpeak

    def __get_var_min_max(self, var_names, i_pt):
        """
        Helper method to get minimum and maximum of ML output score in pT bin

        Parameters
        ----------
        - var_names: variable names of ML output score
        - i_pt: pt bin index

        Returns
        ----------
        - var_min: minimum ML output score value in pt bin
        - var_max: minimum ML output score value  in pt bin
        """

        var_min, var_max = [], []

        if self.n_scan_ml_vars == 1:
            var_min.append(self.cut_vars[var_names[0]]["min"][i_pt] - self.cut_vars[var_names[0]]["step"][i_pt] / 2)
            var_max.append(self.cut_vars[var_names[0]]["max"][i_pt] + self.cut_vars[var_names[0]]["step"][i_pt] / 2)

        elif self.n_scan_ml_vars == 2:
            var_min.append(self.cut_vars[var_names[0]]["min"][i_pt] - self.cut_vars[var_names[0]]["step"][i_pt] / 2)
            var_min.append(self.cut_vars[var_names[1]]["min"][i_pt] - self.cut_vars[var_names[1]]["step"][i_pt] / 2)
            var_max.append(self.cut_vars[var_names[0]]["max"][i_pt] + self.cut_vars[var_names[0]]["step"][i_pt] / 2)
            var_max.append(self.cut_vars[var_names[1]]["max"][i_pt] + self.cut_vars[var_names[1]]["step"][i_pt] / 2)

        return var_min, var_max

    def __get_hist_acc_eff_presel(self):
        """
        Helper method to load preselection acceptance times efficiency

        Returns
        ----------
        - h_acc_eff_presel_prompt: object (TH1 or TEfficiency)
            containing preselection acceptance times efficiency for prompt
        - h_acc_eff_presel_nonprompt: object (TH1 or TEfficiency)
            containing preselection acceptance times efficiency for nonprompt
        """

        h_acc_eff_presel_prompt = self.file_acc_eff_presel.Get(self.config_presel["names"]["prompt"])
        h_acc_eff_presel_nonprompt = self.file_acc_eff_presel.Get(self.config_presel["names"]["nonprompt"])

        # safety
        if not h_acc_eff_presel_prompt.InheritsFrom(self.type_acc_eff_presel):
            print(
                "\033[91mERROR: preselection acceptance times efficiency prompt input"
                f" does not match the type {self.type_acc_eff_presel} defined.\033[0m"
            )
            if not h_acc_eff_presel_nonprompt.InheritsFrom(self.type_acc_eff_presel):
                print(
                    "\033[91mERROR: preselection acceptance times efficiency nonprompt input"
                    f" does not match the type {self.type_acc_eff_presel} defined.\033[0m"
                )
            sys.exit()

        return h_acc_eff_presel_prompt, h_acc_eff_presel_nonprompt

    def __get_hist_cross_sections(self):
        """
        Helper method to load production cross section from FONLL predictions

        Returns
        ----------
        - h_cross_sec_prompt: histogram containing cross section prediction for prompt
        - h_cross_sec_nonprompt:  histogram containing cross section prediction for nonprompt
        """

        h_cross_sec_prompt = self.file_cross_sec.Get(self.config_cross_sec["histonames"]["prompt"])
        h_cross_sec_nonprompt = self.file_cross_sec.Get(self.config_cross_sec["histonames"]["nonprompt"])

        del self.config_cross_sec

        return h_cross_sec_prompt, h_cross_sec_nonprompt

    # pylint: disable=too-many-arguments
    def __configure_h_estim_vs_cut(self, h_estim_vs_cut, var_names, var_min, var_max, i_pt, pt_min, pt_max, suffix):
        """
        Helper method to configure scan plots (estimated quantities versus ML output scores)

        Parameters
        ----------
        - h_estim_vs_cut: histogram containing scan of ML output score for quantities of interest
            in pt bin
        - var_names: variable names of ML output score
        - var_min: minimum ML output score value in pt bin
        - var_max: maximum ML output score value in pt bin
        - i_pt: pt bin index
        - pt_min: pt bin lower limit
        - pt_max: pt bin upper limit
        - suffix: suffix of output histograms names
        """

        if self.n_scan_ml_vars == 1:
            var_min, var_max = var_min[0], var_max[0]
            n_bins_var = int((var_max - var_min) / self.cut_vars[var_names[0]]["step"][i_pt])

            for est, name in EST_NAMES.items():
                h_estim_vs_cut[est] = TH1F(
                    f"h{est}VsCut_pT{pt_min}-{pt_max}" f"_{suffix}",
                    f";{var_names[0]};{name}",
                    n_bins_var,
                    var_min,
                    var_max,
                )

                set_object_style(h_estim_vs_cut[i_pt][est], color=kBlack, marker=kFullCircle, linewidth=1)

        elif self.n_scan_ml_vars == 2:
            n_bins_var_0 = int((var_max[0] - var_min[0]) / self.cut_vars[var_names[0]]["step"][i_pt])
            n_bins_var_1 = int((var_max[1] - var_min[1]) / self.cut_vars[var_names[1]]["step"][i_pt])
            for est, name in EST_NAMES.items():
                h_estim_vs_cut[est] = TH2F(
                    f"h{est}VsCut_pT{pt_min}-{pt_max}" f"_{suffix}",
                    f";{var_names[0]};{var_names[1]};{name}",
                    n_bins_var_0,
                    var_min[0],
                    var_max[0],
                    n_bins_var_1,
                    var_min[1],
                    var_max[1],
                )

    def __fill_h_estim_vs_cut(self, h_estim_vs_cut, cut_set, est_values, est_values_err):
        """
        Helper method to fill histograms for scan plots
        Parameters
        ----------
        - h_estim_vs_cut: histogram containing scan of ML output score for quantities of interest
            in pt bin
        - cut_set: pt bin lower limit
        - est_values: estimated values of quantities of interest
        - est_values_err: error on estimated values of quantities of interest
        """

        if self.n_scan_ml_vars == 1:
            bin_var = h_estim_vs_cut["Signif"].GetXaxis().FindBin(cut_set[0])
            for est, value in est_values.items():
                h_estim_vs_cut[est].SetBinContent(bin_var, value)
                if f"{est}Error" in est_values_err:
                    h_estim_vs_cut[est].SetBinError(bin_var, est_values_err[f"{est}Error"])
        elif self.n_scan_ml_vars == 2:
            bin_var_0 = h_estim_vs_cut["Signif"].GetXaxis().FindBin(cut_set[0])
            bin_var_1 = h_estim_vs_cut["Signif"].GetYaxis().FindBin(cut_set[1])
            for est, value in est_values.items():
                h_estim_vs_cut[est].SetBinContent(bin_var_0, bin_var_1, value)

    # pylint: disable=too-many-statements, too-many-branches, too-many-locals
    def process(self):
        """
        Process method of the class
        """
        # set batch mode if enabled
        if self.batch:
            gROOT.SetBatch(True)
            gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;")

        set_global_style(
            padleftmargin=0.12,
            padrightmargin=0.21,
            padbottommargin=0.15,
            padtopmargin=0.075,
            titleoffset=1.1,
            palette=kRainBow,
            titlesize=0.06,
            labelsize=0.055,
            maxdigits=4,
        )

        self.__check_input_consistency()
        self.__complete_bkg_config()

        # configure output
        out_file = TFile(self.outfile, "recreate")
        out_dir_fit_sidebands, out_dir_plots, t_signif = self.__configure_output(out_file)
        out_dir_fit_sidebands_pt, out_dir_plots_pt = [], []
        c_signif_vs_rest, c_estim_vs_cut = [], []
        h_signif_vs_rest, h_estim_vs_cut = [], []

        watermark = TLatex(0.13, 0.95, self.watermark)
        watermark.SetNDC()
        watermark.SetTextSize(0.055)
        watermark.SetTextFont(42)

        # configure scan of ML output
        cut_ranges, cut_directions, var_names = self.__configure_scan()

        # load labeled dataframes
        df_bkg, df_prompt, df_nonprompt = self.__get_labeled_dfs()
        df_prompt_secpeak, df_nonprompt_secpeak = self.__get_labeled_dfs_secpeak()

        secpeak = False
        if not (df_prompt_secpeak.empty and df_nonprompt_secpeak.empty):
            secpeak = True

        # load preselection acceptance-times-efficiency
        self.file_acc_eff_presel = TFile.Open(self.config_presel["filename"])
        h_acc_eff_presel_prompt, h_acc_eff_presel_nonprompt = self.__get_hist_acc_eff_presel()

        # load theory-predicted cross sections
        self.file_cross_sec = TFile.Open(self.config_cross_sec["filename"])
        h_cross_sec_prompt, h_cross_sec_nonprompt = self.__get_hist_cross_sections()

        # compute total number of cut sets
        tot_sets = [1 for _ in enumerate(self.pt_maxs)]
        for i_pt, _ in enumerate(self.pt_maxs):
            for cut_range in cut_ranges[i_pt]:
                tot_sets[i_pt] *= len(cut_range)
        print(f"Total number of sets per pT bin: {tot_sets}")

        # loop over pT bins (i.e. over models)
        for i_pt, (pt_min, pt_max) in enumerate(zip(self.pt_mins, self.pt_maxs)):
            pt_cent = (pt_min + pt_max) / 2.0
            var_min, var_max = self.__get_var_min_max(var_names, i_pt)

            # configure pT-dependent output
            out_dir_fit_sidebands.cd()
            out_dir_fit_sidebands_pt.append(TDirectoryFile(f"pT{pt_min}-{pt_max}", f"pT{pt_min}-{pt_max}"))
            out_dir_fit_sidebands_pt[i_pt].Write()

            out_dir_plots.cd()
            out_dir_plots_pt.append(TDirectoryFile(f"pT{pt_min}-{pt_max}", f"pT{pt_min}-{pt_max}"))
            out_dir_plots_pt[i_pt].Write()

            h_signif_vs_rest.append({})
            h_estim_vs_cut.append({})

            # ??? no need to reshuffle bkg as we take the whole input of ModelApplied
            df_bkg_pt = df_bkg.query(f"{pt_min} < fPt < {pt_max}")
            df_prompt_pt = df_prompt.query(f"{pt_min} < fPt < {pt_max}")
            df_nonprompt_pt = df_nonprompt.query(f"{pt_min} < fPt < {pt_max}")

            # denominator for ML efficiencies
            n_tot_bkg = len(df_bkg_pt)
            n_tot_prompt = len(df_prompt_pt)
            n_tot_nonprompt = len(df_nonprompt_pt)

            if self.config_presel["filename"]:
                if self.type_acc_eff_presel == "TH1":
                    pt_bin = h_acc_eff_presel_prompt.GetXaxis().FindBin(pt_min * 1.0001)
                    acc_eff_presel_prompt = h_acc_eff_presel_prompt.GetBinContent(pt_bin)
                    acc_eff_presel_nonprompt = h_acc_eff_presel_nonprompt.GetBinContent(pt_bin)
                    unc_acc_eff_presel_prompt = h_acc_eff_presel_prompt.GetBinError(pt_bin)
                    unc_acc_eff_presel_nonprompt = h_acc_eff_presel_nonprompt.GetBinError(pt_bin)
                else:
                    pt_bin = h_acc_eff_presel_prompt.FindFixBin(pt_min * 1.0001)
                    acc_eff_presel_prompt = h_acc_eff_presel_prompt.GetEfficiency(pt_bin)
                    acc_eff_presel_nonprompt = h_acc_eff_presel_nonprompt.GetEfficiency(pt_bin)
                    unc_low_acc_eff_presel_prompt = h_acc_eff_presel_prompt.GetEfficiencyErrorLow(pt_bin)
                    unc_up_acc_eff_presel_prompt = h_acc_eff_presel_prompt.GetEfficiencyErrorUp(pt_bin)
                    unc_low_acc_eff_presel_nonprompt = h_acc_eff_presel_nonprompt.GetEfficiencyErrorLow(pt_bin)
                    unc_up_acc_eff_presel_nonprompt = h_acc_eff_presel_nonprompt.GetEfficiencyErrorUp(pt_bin)
                    # take the max, to be consistent with efficiency macro
                    unc_acc_eff_presel_prompt = max(unc_low_acc_eff_presel_prompt, unc_up_acc_eff_presel_prompt)
                    unc_acc_eff_presel_nonprompt = max(
                        unc_low_acc_eff_presel_nonprompt, unc_up_acc_eff_presel_nonprompt
                    )
                print(f"pt_bin: {pt_bin}")
            else:  # default dummy values
                acc_eff_presel_prompt = 1.0
                acc_eff_presel_nonprompt = 1.0
                unc_acc_eff_presel_prompt = 0.01
                unc_acc_eff_presel_nonprompt = 0.01

            # cross sections
            cross_sec_prompt, cross_sec_nonprompt = get_cross_sections(
                h_cross_sec_prompt, h_cross_sec_nonprompt, pt_min, pt_max
            )

            # signal histograms
            h_mass_signal = TH1F(
                f"hMassSignal_pT{pt_min}-{pt_max}",
                ";#it{M} (GeV/#it{c});Counts",
                400,
                min(df_prompt_pt["fM"]),
                max(df_prompt_pt["fM"]),
            )
            for mass in np.concatenate((df_prompt_pt["fM"].to_numpy(), df_nonprompt_pt["fM"].to_numpy())):
                h_mass_signal.Fill(mass)
            func_signal = TF1("funcSignal", single_gaus, 1.6, 2.2, 3)
            func_signal.SetParameters(h_mass_signal.Integral("width"), h_mass_signal.GetMean(), h_mass_signal.GetRMS())
            h_mass_signal.Fit("funcSignal", "Q0")
            mean = func_signal.GetParameter(1)
            sigma = func_signal.GetParameter(2)

            # second peak
            mean_secpeak, sigma_secpeak = 0.0, 0.0
            if secpeak:
                h_mass_secpeak = TH1F(
                    f"hMassSignal_pT{pt_min}-{pt_max}",
                    ";#it{M} (GeV/#it{c});Counts",
                    400,
                    min(df_prompt_secpeak["inv_mass"]),
                    max(df_prompt_secpeak["inv_mass"]),
                )
                for mass in np.concatenate((df_prompt_secpeak["fM"].to_numpy(), df_nonprompt_secpeak["fM"].to_numpy())):
                    h_mass_secpeak.Fill(mass)
                func_signal.SetParameters(
                    h_mass_secpeak.Integral("width"), h_mass_secpeak.GetMean(), h_mass_secpeak.GetRMS()
                )
                h_mass_secpeak.Fit("funcSignal", "Q0")
                mean_secpeak = func_signal.GetParameter(1)
                sigma_secpeak = func_signal.GetParameter(2)

            # cuts over df column query over each df and relative scan histos
            for i_par_cut, (par_cut_min, par_cut_max) in enumerate(zip(self.par_cuts_min, self.par_cuts_max)):
                # plots
                suffix = self.name_par_cuts
                if self.name_par_cuts != NO_EXTRA_CUT:
                    suffix = self.name_par_cuts[i_par_cut] + f"{par_cut_min}-{par_cut_max}"
                self.__configure_h_estim_vs_cut(
                    h_estim_vs_cut[i_pt], var_names, var_min, var_max, i_pt, pt_min, pt_max, suffix
                )

                start_time = time.time()
                with alive_bar(tot_sets[i_pt]) as bar:
                    for i_set, cut_set in enumerate(itertools.product(*cut_ranges[i_pt])):
                        selection = str()
                        for i_cut, (cut, cut_direction, name_var) in enumerate(zip(cut_set, cut_directions, var_names)):
                            if i_cut == 0:
                                selection = f"{name_var}{cut_direction}{cut}"
                            else:
                                selection += f" & {name_var}{cut_direction}{cut}"
                        if self.name_par_cuts != NO_EXTRA_CUT:
                            selection += f" & {par_cut_min} < {self.name_par_cuts} < {par_cut_max}"

                        bar()

                        # selected candidates
                        df_bkg_pt_sel = df_bkg_pt.query(selection)
                        df_prompt_pt_sel = df_prompt_pt.query(selection)
                        df_nonprompt_pt_sel = df_nonprompt_pt.query(selection)

                        # model efficiencies
                        eff_ml_bkg, _ = get_ml_efficiency(len(df_bkg_pt_sel), n_tot_bkg)
                        eff_ml_prompt, unc_eff_ml_prompt = get_ml_efficiency(len(df_prompt_pt_sel), n_tot_prompt)
                        eff_ml_nonprompt, unc_eff_ml_nonprompt = get_ml_efficiency(
                            len(df_nonprompt_pt_sel), n_tot_nonprompt
                        )

                        # acceptance-times-efficiency factors
                        acc_eff_prompt, unc_acc_eff_prompt = get_acc_eff(
                            acc_eff_presel_prompt, eff_ml_prompt, unc_acc_eff_presel_prompt, unc_eff_ml_prompt
                        )
                        acc_eff_nonprompt, unc_acc_eff_nonprompt = get_acc_eff(
                            acc_eff_presel_nonprompt,
                            eff_ml_nonprompt,
                            unc_acc_eff_presel_nonprompt,
                            unc_eff_ml_nonprompt,
                        )

                        # yield fractions
                        f_prompt, f_nonprompt = get_fractions_fc(
                            acc_eff_prompt, acc_eff_nonprompt, cross_sec_prompt, cross_sec_nonprompt
                        )

                        # expected signal (BR already included in cross section)
                        exp_signal = 0.0
                        if self.expected_signal_promptness == "prompt":
                            exp_signal = get_expected_signal(
                                cross_sec_prompt,
                                pt_max - pt_min,
                                1.0,
                                acc_eff_prompt,
                                1.0,
                                self.n_exp_ev,
                                f_prompt[0],
                                self.sigma_mb,
                            )
                        elif self.expected_signal_promptness == "nonprompt":
                            exp_signal = get_expected_signal(
                                cross_sec_nonprompt,
                                pt_max - pt_min,
                                1.0,
                                acc_eff_nonprompt,
                                1.0,
                                self.n_exp_ev,
                                f_nonprompt[0],
                                self.sigma_mb,
                            )

                        # expected background
                        out_dir_fit_sidebands_pt[i_pt].cd()
                        exp_bkg = 0.0
                        err_exp_bkg = 0.0

                        h_mass_bkg = TH1F(
                            f"hMassBkg_pT{pt_min}-{pt_max}_cutSet{i_set}",
                            ";#it{M} (GeV/#it{c});Counts",
                            200,
                            min(df_bkg_pt_sel["fM"]),
                            max(df_bkg_pt_sel["fM"]),
                        )

                        for mass in df_bkg_pt_sel["fM"].to_list():
                            h_mass_bkg.Fill(mass)
                        if self.bkg_is_mc:
                            exp_bkg, err_exp_bkg, h_mass_bkg = get_expected_bkg_from_mc(h_mass_bkg, mean, sigma)
                        else:
                            exp_bkg, err_exp_bkg, h_mass_bkg = get_expected_bkg_from_side_bands(
                                h_mass_bkg,
                                self.func_fit_bkg,
                                self.nsigma_fit_bkg,
                                mean,
                                sigma,
                                mean_secpeak,
                                sigma_secpeak,
                                self.mass_min,
                                self.mass_max,
                            )
                        h_mass_bkg.Write()

                        bkg_corr_factor = self.n_exp_ev / self.n_ev / self.bkg_frac_used_for_ml[i_pt] * eff_ml_bkg
                        exp_bkg *= bkg_corr_factor
                        err_exp_bkg *= bkg_corr_factor

                        if self.h_bkg_mc_corr_factor:
                            exp_bkg *= self.h_bkg_mc_corr_factor.GetBinContent(
                                self.h_bkg_mc_corr_factor.FindBin(pt_cent)
                            )
                            err_exp_bkg *= self.h_bkg_mc_corr_factor.GetBinContent(
                                self.h_bkg_mc_corr_factor.FindBin(pt_cent)
                            )

                        # S/B and significance
                        exp_s_over_b = 0.0
                        exp_signif = 0.0
                        # TODO: think how to define a meaningful error on the estimated signal
                        # and propagate it
                        err_s = 0.0
                        err_s_over_b = 0.0
                        err_signif = 0.0
                        if exp_bkg > 0:
                            exp_s_over_b = exp_signal / exp_bkg
                            exp_signif = exp_signal / np.sqrt(exp_signal + exp_bkg)
                            err_s_over_b = exp_s_over_b * err_exp_bkg / exp_bkg
                            err_signif = exp_signif * 0.5 * err_exp_bkg / (exp_signal + exp_bkg)
                        else:
                            print("\033[91mERROR: expected background is not positive!\033[0m")
                            sys.exit()

                        tuple_for_ntuple = cut_set + (
                            pt_min,
                            pt_max,
                            par_cut_min,
                            par_cut_max,
                            unc_acc_eff_prompt,
                            unc_acc_eff_nonprompt,
                            err_s,
                            err_exp_bkg,
                            err_signif,
                            err_s_over_b,
                            exp_signif,
                            exp_s_over_b,
                            exp_signal,
                            exp_bkg,
                            acc_eff_prompt,
                            acc_eff_nonprompt,
                            f_prompt[0],
                            f_nonprompt[0],
                        )
                        t_signif.Fill(np.array(tuple_for_ntuple, "f"))

                        # estimated values and errors
                        est_values = {
                            "Signif": exp_signif,
                            "SoverB": exp_s_over_b,
                            "S": exp_signal,
                            "B": exp_bkg,
                            "EffAccPrompt": acc_eff_prompt,
                            "EffAccNonprompt": acc_eff_nonprompt,
                            "fPrompt": f_prompt[0],
                            "fNonprompt": f_nonprompt[0],
                        }
                        est_values_err = {
                            "SignifError": err_signif,
                            "SoverBError": err_s_over_b,
                            "SError": err_s,
                            "BError": err_exp_bkg,
                            "EffAccPromptError": unc_acc_eff_prompt,
                            "EffAccNonpromptError": unc_acc_eff_nonprompt,
                        }

                        self.__fill_h_estim_vs_cut(h_estim_vs_cut[i_pt], cut_set, est_values, est_values_err)

                if self.name_par_cuts != NO_EXTRA_CUT:
                    print(
                        f"Time elapsed to test cut sets for pT bin {pt_min}-{pt_max} "
                        f"and {self.name_par_cuts} bin {par_cut_min}-{par_cut_max}: "
                        f"{time.time()-start_time:.2f}s          "
                    )
                else:
                    print(
                        f"Time elapsed to test cut sets for pT bin {pt_min}-{pt_max}: "
                        f"{time.time()-start_time:.2f}s            "
                    )

                out_dir_plots_pt[i_pt].mkdir(suffix)
                c_signif_vs_rest.append(
                    TCanvas(
                        f"cSignifVsRest_pT{pt_min}-{pt_max}_{suffix}",
                        f"cSignifVsRest_pT{pt_min}-{pt_max}_{suffix}",
                        800,
                        1000,
                    )
                )
                c_signif_vs_rest[i_pt].Divide(2, 4)

                # plots with significance vs other estimated quantities
                draw_watermark = True
                for i_pad, est in enumerate(EST_NAMES):
                    if est != "Signif":
                        h_frame = (
                            c_signif_vs_rest[i_pt]
                            .cd(i_pad)
                            .DrawFrame(
                                t_signif.GetMinimum(est) * 0.8,
                                t_signif.GetMinimum("Signif") * 0.8,
                                t_signif.GetMaximum(est) * 1.2,
                                t_signif.GetMaximum("Signif") * 1.2,
                                f";{EST_NAMES[est]};{EST_NAMES['Signif']}",
                            )
                        )
                        h_frame.GetXaxis().SetDecimals()
                        h_frame.GetYaxis().SetDecimals()

                        if draw_watermark:
                            watermark.Draw()
                            draw_watermark = False

                        h_signif_vs_rest[i_pt][est] = TH2F(
                            (f"hSignifVs{est}_pT{pt_min}-{pt_max}_{suffix}"),
                            f";{EST_NAMES[est]};{EST_NAMES['Signif']}",
                            50,
                            t_signif.GetMinimum(est) * 0.8,
                            t_signif.GetMaximum(est) * 1.2,
                            50,
                            t_signif.GetMinimum("Signif") * 0.8,
                            t_signif.GetMaximum("Signif") * 1.0,
                        )
                        t_signif.Draw(
                            f"Signif:{est}>>hSignifVs{est}_pT{pt_min}-{pt_max}_{suffix}",
                            f"PtMin == {pt_min} && PtMax == {pt_max}",
                            "colz same",
                        )
                        c_signif_vs_rest[i_pt].Update()
                        c_signif_vs_rest[i_pt].Modified()
                        out_dir_plots_pt[i_pt].cd(suffix)
                        h_signif_vs_rest[i_pt][est].Write()
                out_dir_plots_pt[i_pt].cd(suffix)
                c_signif_vs_rest[i_pt].Write()

                # plots: scan of estimated quantities
                draw_watermark = True
                if self.n_scan_ml_vars == 1:
                    c_estim_vs_cut.append(
                        TCanvas(
                            f"cEstimVsCut_pT{pt_min}-{pt_max}_{suffix}",
                            f"cEstimVsCut_pT{pt_min}-{pt_max}_{suffix}",
                            800,
                            1000,
                        )
                    )
                    c_estim_vs_cut[i_pt].Divide(2, 4)
                    for i_pad, est in enumerate(h_estim_vs_cut[i_pt]):
                        h_frame = (
                            c_estim_vs_cut[i_pt]
                            .cd(i_pad + 1)
                            .DrawFrame(
                                var_min[0],
                                t_signif.GetMinimum(est) * 0.8,
                                var_max[0],
                                t_signif.GetMaximum(est) * 1.2,
                                f";{var_names[0]};{EST_NAMES[est]}",
                            )
                        )
                        if draw_watermark:
                            watermark.Draw()
                            draw_watermark = False
                        if "Eff" in est:
                            min_bin = h_estim_vs_cut[i_pt][est].GetMinimumBin()
                            max_bin = h_estim_vs_cut[i_pt][est].GetMaximumBin()
                            min_value = h_estim_vs_cut[i_pt][est].GetBinContent(min_bin)
                            max_value = h_estim_vs_cut[i_pt][est].GetBinContent(max_bin)
                            # safety, otherwise y axis graduation is not readable
                            if (max_value - min_value) >= 11:
                                c_estim_vs_cut[i_pt].cd(i_pad + 1).SetLogy()
                                h_frame.GetYaxis().SetMoreLogLabels()
                        h_frame.GetXaxis().SetNdivisions(505)
                        h_frame.GetXaxis().SetDecimals()
                        h_frame.GetYaxis().SetDecimals()
                        h_estim_vs_cut[i_pt][est].DrawCopy("psame")
                        out_dir_plots_pt[i_pt].cd(suffix)
                        h_estim_vs_cut[i_pt][est].Write()

                elif self.n_scan_ml_vars == 2:
                    c_estim_vs_cut.append(
                        TCanvas(
                            f"cEstimVsCut_pT{pt_min}-{pt_max}_{suffix}",
                            f"cEstimVsCut_pT{pt_min}-{pt_max}_{suffix}",
                            800,
                            1000,
                        )
                    )
                    c_estim_vs_cut[i_pt].Divide(2, 4)
                    for i_pad, est in enumerate(h_estim_vs_cut[i_pt]):
                        h_frame = (
                            c_estim_vs_cut[i_pt]
                            .cd(i_pad + 1)
                            .DrawFrame(
                                var_min[0],
                                var_min[1],
                                var_max[0],
                                var_max[1],
                                f";{var_names[0]};{var_names[1]};{EST_NAMES[est]}",
                            )
                        )
                        if draw_watermark:
                            watermark.Draw()
                            draw_watermark = False
                        if "Eff" in est:
                            min_bin = h_estim_vs_cut[i_pt][est].GetMinimumBin()
                            max_bin = h_estim_vs_cut[i_pt][est].GetMaximumBin()
                            min_value = h_estim_vs_cut[i_pt][est].GetBinContent(min_bin)
                            max_value = h_estim_vs_cut[i_pt][est].GetBinContent(max_bin)
                            # safety, otherwise z axis graduation is not readable
                            if (max_value - min_value) >= 11:
                                c_estim_vs_cut[i_pt].cd(i_pad + 1).SetLogz()
                                h_frame.GetZaxis().SetMoreLogLabels()
                        h_frame.GetXaxis().SetNdivisions(505)
                        h_frame.GetYaxis().SetNdivisions(505)
                        h_frame.GetXaxis().SetDecimals()
                        h_frame.GetYaxis().SetDecimals()
                        h_estim_vs_cut[i_pt][est].DrawCopy("colzsame")
                        out_dir_plots_pt[i_pt].cd(suffix)
                        h_estim_vs_cut[i_pt][est].Write()
                c_estim_vs_cut[i_pt].Update()
                c_estim_vs_cut[i_pt].Modified()
                out_dir_plots_pt[i_pt].cd(suffix)
                c_estim_vs_cut[i_pt].Write()

                for ext in self.extension:
                    c_estim_vs_cut[i_pt].Print(
                        f"{self.outdir}{self.expected_signal_promptness}" f"_scan_pT{pt_min}_{pt_max}.{ext}", f"{ext}"
                    )

        out_file.cd()
        t_signif.Write()
        self.file_cross_sec.Close()
        out_file.Close()

        if not self.batch:
            input("Press enter to exit")


def main(cfg, batch):
    """
    Main function

    Parameters
    -----------------
    - config: dictionary with config read from a yaml file
    - batch: bool to suppress video output
    """
    MlOutputScanner(cfg, batch).process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument(
        "config", metavar="text", default="config.yml", help="config file name for working point optimisation"
    )
    parser.add_argument("--batch", help="suppress video output", action="store_true")
    args = parser.parse_args()

    print("Loading analysis configuration: ...", end="\r")
    with open(args.config, "r", encoding="utf-8") as yml_cfg:
        configuration = yaml.load(yml_cfg, yaml.FullLoader)
    print("Loading analysis configuration: Done!")

    main(configuration, args.batch)
