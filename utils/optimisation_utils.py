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
file: optimisation_utils.py
brief: module with utils methods for ML model working point optimisation
note: adapted from Run2 macros
author: Alexandre Bigot <alexandre.bigot@cern.ch>, Strasbourg University
"""

import ctypes

import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
from ROOT import TH1F  # pylint: disable=import-error

from utils.fit_utils import BkgFitFuncCreator


def load_df_from_parquet(file_names):
    """
    Helper method to load a pandas dataframe from either root or parquet files

    Arguments
    ----------
    - input file name of list of input file names

    Returns
    ----------
    - loaded pandas dataframe
    """

    if not isinstance(file_names, list):
        file_names = [file_names]

    df_list = []
    for file_name in file_names:
        if ".parquet" in file_name:
            df_list.append(pd.read_parquet(file_name))
        else:
            print("\033[91mERROR: only parquet files are supported! " "Returning empty dataframe\033[0m")
            return pd.DataFrame()

    df_out = pd.concat(df_list, ignore_index=True)

    return df_out


def get_cross_sections(h_cross_sec_prompt, h_cross_sec_nonprompt, pt_min, pt_max):
    """
    Method to get cross sections predictions from FONLL

    Parameters
    ----------
    - h_cross_sec_prompt: histogram containing prompt cross sections
    - h_cross_sec_nonprompt: histogram containing nonprompt cross sections
    - pt_min: lower limit of pT interval
    - pt_max: upper limit of pT interval

    Returns
    ----------
    - prompt and non prompt cross sections in pt_min < pT < pt_max
    """

    pt_bin_cross_sec_min = h_cross_sec_prompt.GetXaxis().FindBin(pt_min * 1.0001)
    pt_bin_cross_sec_max = h_cross_sec_prompt.GetXaxis().FindBin(pt_max * 0.9999)
    cross_sec_prompt = h_cross_sec_prompt.Integral(pt_bin_cross_sec_min, pt_bin_cross_sec_max, "width") / (
        pt_max - pt_min
    )
    cross_sec_nonprompt = h_cross_sec_nonprompt.Integral(pt_bin_cross_sec_min, pt_bin_cross_sec_max, "width") / (
        pt_max - pt_min
    )

    return cross_sec_prompt, cross_sec_nonprompt


def get_ml_efficiency(n_selected_ml, n_tot_ml):
    """
    Method to compute ML efficiency

    Parameters
    ----------
    - n_selected_ml: number of candidates that passed ML selections
    - n_tot_ml: total number of candidates

    Returns
    ----------
    - ML efficiency, error on ML efficiency
    """
    h_num = TH1F("h_num", "", 1, 0, 1)
    h_denom = TH1F("h_denom", "", 1, 0, 1)
    h_num.SetBinContent(1, n_selected_ml)
    h_denom.SetBinContent(1, n_tot_ml)
    h_num.SetBinError(1, np.sqrt(n_selected_ml))
    h_denom.SetBinError(1, np.sqrt(n_tot_ml))
    h_num.Divide(h_num, h_denom, 1.0, 1, "B")

    return h_num.GetBinContent(1), h_num.GetBinError(1)

    # TODO: check if use TEfficiency would not be better


def get_acc_eff(acc_eff_presel, eff_ml, unc_acc_eff_presel, unc_eff_ml):
    """
    Method to compute ML efficiency

    Parameters
    ----------
    - acc_eff_presel: preselection acceptance times efficiency
    - eff_ml: model efficiency

    Returns
    ----------
    - acceptance times efficiency, error acceptance times efficiency
    """

    acc_eff = acc_eff_presel * eff_ml
    unc_acc_eff = np.sqrt((eff_ml * unc_acc_eff_presel) ** 2 + (acc_eff_presel * unc_eff_ml) ** 2)

    return acc_eff, unc_acc_eff


# pylint: disable=too-many-branches
def get_fractions_fc(acc_eff_prompt, acc_eff_nonprompt, cross_sec_prompt, cross_sec_nonprompt):
    """
    Method to get fraction of prompt / FD fraction with fc method

    Parameters
    ----------
    - acc_eff_prompt: efficiency times acceptance of prompt D
    - acc_eff_nonprompt: efficiency times acceptance of feed-down D
    - cross_sec_prompt: list of production cross sections (cent, min, max)
                        of prompt D in pp collisions from theory
    - cross_sec_nonprompt: list of production cross sections (cent, min, max)
                           of non-prompt D in pp collisions from theory

    Returns
    ----------
    - frac_prompt: list of fraction of prompt D (cent, min, max)
    - frac_nonprompt: list of fraction of feed-down D (cent, min, max)
    """
    if not isinstance(cross_sec_prompt, list) and isinstance(cross_sec_prompt, float):
        cross_sec_prompt = [cross_sec_prompt]
    if not isinstance(cross_sec_nonprompt, list) and isinstance(cross_sec_nonprompt, float):
        cross_sec_nonprompt = [cross_sec_nonprompt]

    frac_prompt, frac_nonprompt = [], []
    frac_prompt_cent, frac_nonprompt_cent = 0.0, 1.0
    if acc_eff_prompt == 0:
        frac_prompt = [frac_prompt_cent, frac_prompt_cent, frac_prompt_cent]
        frac_nonprompt = [frac_nonprompt_cent, frac_nonprompt_cent, frac_nonprompt_cent]
        return frac_prompt, frac_nonprompt
    if acc_eff_nonprompt == 0:
        frac_nonprompt_cent = 0.0
        frac_prompt_cent = 1.0
        frac_prompt = [frac_prompt_cent, frac_prompt_cent, frac_prompt_cent]
        frac_nonprompt = [frac_nonprompt_cent, frac_nonprompt_cent, frac_nonprompt_cent]
        return frac_prompt, frac_nonprompt

    for i_sigma, (sigma_p, sigma_np) in enumerate(zip(cross_sec_prompt, cross_sec_nonprompt)):
        if i_sigma == 0:
            frac_prompt_cent = 1.0 / (1 + acc_eff_nonprompt / acc_eff_prompt * sigma_np / sigma_p)
            frac_nonprompt_cent = 1.0 / (1 + acc_eff_prompt / acc_eff_nonprompt * sigma_p / sigma_np)
        else:
            frac_prompt.append(1.0 / (1 + acc_eff_nonprompt / acc_eff_prompt * sigma_np / sigma_p))
            frac_nonprompt.append(1.0 / (1 + acc_eff_prompt / acc_eff_nonprompt * sigma_p / sigma_np))

    if frac_prompt and frac_nonprompt:
        frac_prompt.sort()
        frac_nonprompt.sort()
        frac_prompt = [frac_prompt_cent, frac_prompt[0], frac_prompt[-1]]
        frac_nonprompt = [frac_nonprompt_cent, frac_nonprompt[0], frac_nonprompt[-1]]
    else:
        frac_prompt = [frac_prompt_cent, frac_prompt_cent, frac_prompt_cent]
        frac_nonprompt = [frac_nonprompt_cent, frac_nonprompt_cent, frac_nonprompt_cent]

    return frac_prompt, frac_nonprompt


# pylint: disable= too-many-arguments
def get_expected_signal(cross_sec_br, delta_pt, delta_y, acc_eff, frag_frac, n_ev, yield_frac, sigma_mb):
    """
    Helper method to get expected signal from MC and predictions

    Parameters
    ----------
    - cross_sec_br: prediction for differential (cross section x BR) in pp
    - delta_pt: pT interval
    - delta_y: Y interval
    - acc_eff: efficiency times acceptance for prompt or feed-down
    - yield_frac: either prompt or feed-down fraction
    - BR: branching ratio of the decay channel
    - frag_frac: fragmentation fraction
    - nEv: number of expected events
    - sigmaMB: hadronic cross section for MB
    - TAA: average overlap nuclear function
    - RAA: expected nuclear modification factor

    Returns
    ----------
    - expected signal
    """
    return 2 * cross_sec_br * delta_pt * delta_y * acc_eff * frag_frac * n_ev / yield_frac / sigma_mb


def get_expected_bkg_from_side_bands(
    h_mass_bkg,
    func_fit_bkg="pol2",
    nsigma_side_bands=4,
    mean=0.0,
    sigma=0.0,
    meanSecPeak=0.0,
    sigmaSecPeak=0.0,
    mass_min=None,
    mass_max=None,
):
    """
    Helper method to get the expected bkg from side-bands, using maximum-likelihood and
    background functions defined only on the sidebands

    Parameters
    ----------
    - h_mass_bkg: invariant-mass histogram from which extract the estimated bkg
    - func_fit_bkg: expression for bkg fit function
    - nsigma_side_bands: number of sigmas away from the invariant-mass peak to define SB windows
    - mean: mean of invariant-mass peak of the signal
    - sigma: width of invariant-mass peak of the signal
    - meanSecPeak: mean of invariant-mass peak of the second peak (only Ds)
    - sigmaSecPeak: width of invariant-mass peak of the second peak (only Ds)

    Returns
    ----------
    - exp_bkg_3sigma: expected background within 3 sigma from signal peak mean
    - err_exp_bkg_3sigma: error on the expected background
    - h_mass_bkg: SB histogram with fit function (if fit occurred)
    """
    n_entries_side_bands = h_mass_bkg.Integral(1, h_mass_bkg.FindBin(mean - nsigma_side_bands * sigma))
    n_entries_side_bands += h_mass_bkg.Integral(
        h_mass_bkg.FindBin(mean + nsigma_side_bands * sigma), h_mass_bkg.GetNbinsX()
    )
    if meanSecPeak > 0 and sigmaSecPeak > 0:
        n_entries_side_bands -= h_mass_bkg.Integral(
            h_mass_bkg.FindBin(meanSecPeak - nsigma_side_bands * sigmaSecPeak),
            h_mass_bkg.FindBin(meanSecPeak + nsigma_side_bands * sigmaSecPeak),
        )

    if n_entries_side_bands <= 5:  # check to have some entries in the histogram before fitting
        return 0.0, 0.0, h_mass_bkg
    if mass_min is None:
        mass_min = h_mass_bkg.GetBinLowEdge(1)

    if mass_max is None:
        mass_max = h_mass_bkg.GetBinLowEdge(h_mass_bkg.GetNbinsX()) + h_mass_bkg.GetBinWidth(1)

    fitted_bkg = BkgFitFuncCreator(
        func_fit_bkg, mass_min, mass_max, nsigma_side_bands, mean, sigma, meanSecPeak, sigmaSecPeak
    )
    func_bkg_side_bands = fitted_bkg.get_func_side_bands(h_mass_bkg.Integral("width"))
    fit = h_mass_bkg.Fit(func_bkg_side_bands, "LRQ+")
    exp_bkg_3sigma, err_exp_bkg_3sigma = 0.0, 0.0
    if int(fit) == 0:
        func_bkg = fitted_bkg.get_func_full_range(func_bkg_side_bands)
        exp_bkg_3sigma = func_bkg.Integral(mean - 3 * sigma, mean + 3 * sigma) / h_mass_bkg.GetBinWidth(1)
        err_exp_bkg_3sigma = func_bkg.IntegralError(mean - 3 * sigma, mean + 3 * sigma) / h_mass_bkg.GetBinWidth(1)
    return exp_bkg_3sigma, err_exp_bkg_3sigma, h_mass_bkg


def get_expected_bkg_from_mc(h_mass_bkg, mean=0.0, sigma=0.0, do_fit=True, func_fit_bkg="pol3"):
    """
    Helper method to get the expected bkg from MC

    Parameters
    ----------
    - h_mass_bkg: invariant-mass histogram of background
    - mean: mean of invariant-mass peak of the signal
    - sigma: width of invariant-mass peak of the signal
    - do_fit: flag to enable fit of bkg distribution (useful with low stat)
    - func_fit_bkg: expression for bkg fit function (if fit enabled)

    Returns
    ----------
    - exp_bkg_3sigma: expected background within 3 sigma from signal peak mean
    - err_exp_bkg_3sigma: error on the expected background
    - h_mass_bkg: bkg histogram with fit function (if fit occurred)
    """
    exp_bkg_3sigma, err_exp_bkg_3sigma = 0.0, 0.0
    if do_fit:
        if h_mass_bkg.Integral() <= 5:  # check to have some entries in the histogram before fitting
            return 0.0, 0.0, h_mass_bkg
        mass_min = h_mass_bkg.GetBinLowEdge(1)
        mass_max = h_mass_bkg.GetBinLowEdge(h_mass_bkg.GetNbinsX()) + h_mass_bkg.GetBinWidth(1)
        fitted_bkg = BkgFitFuncCreator(func_fit_bkg, mass_min, mass_max)
        func_bkg = fitted_bkg.get_func_side_bands(h_mass_bkg.Integral("width"))
        fit = h_mass_bkg.Fit(func_bkg, "LRQ+")
        if int(fit) == 0:
            exp_bkg_3sigma = func_bkg.Integral(mean - 3 * sigma, mean + 3 * sigma) / h_mass_bkg.GetBinWidth(1)
            err_exp_bkg_3sigma = func_bkg.IntegralError(mean - 3 * sigma, mean + 3 * sigma) / h_mass_bkg.GetBinWidth(1)
    else:
        bin_min_mass = h_mass_bkg.GetXaxis().FindBin(mean - 3 * sigma)
        bin_max_mass = h_mass_bkg.GetXaxis().FindBin(mean + 3 * sigma)
        err_exp_bkg_3sigma = ctypes.c_double()
        exp_bkg_3sigma = h_mass_bkg.IntegralAndError(bin_min_mass, bin_max_mass, err_exp_bkg_3sigma)

    return exp_bkg_3sigma, err_exp_bkg_3sigma.value, h_mass_bkg  # type: ignore
