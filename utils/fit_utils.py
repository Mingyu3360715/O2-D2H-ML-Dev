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
file: fit_utils.py
brief: module with function definitions and fit utils
note: adapted from Run2 macros
author: Alexandre Bigot <alexandre.bigot@cern.ch>, Strasbourg University
"""

# pylint: disable=import-error, no-name-in-module
from ROOT import TF1, TDatabasePDG, TMath, kBlue, kGreen


def single_gaus(x, par):
    """
    Gaussian function

    Parameters
    ----------
    - x: function variable
    - par: function parameters
        par[0]: normalisation
        par[1]: mean
        par[2]: sigma
    """
    return par[0] * TMath.Gaus(x[0], par[1], par[2], True)


# pylint: disable=too-many-instance-attributes
class BkgFitFuncCreator:
    """
    Class to handle custom background functions as done by AliHFInvMassFitter. Mainly designed
    to provide functions for sidebands fitting

    Parameters
    -------------------------------------------------
    - name_func: function to use. Currently implemented: 'expo', 'pol0', 'pol1', 'pol2', 'pol3'
    - mass_min:  lower extreme of fitting interval
    - mass_max:  higher extreme of fitting interval
    - nsigma_side_bands: number of widths excluded around the peak
    - mean_mass_peak: peak mass
      (if not defined the signal region will not be excluded from the function)
    - sigma_mass_peak: peak width
    - mean_sec_mass_peak: second peak mass
      (if not defined the second-peak region will not be excluded from the function)
    - sigma_sec_mass_peak: second peak width
    """

    __impl_func = {
        "expo": "_expo_normalised",
        "pol0": "_pol0_normalised",
        "pol1": "_pol1_normalised",
        "pol2": "_pol2_normalised",
        "pol3": "_pol3_normalised",
        "expopow": "_expopow_normalised",
    }

    __n_pars = {"expo": 2, "pol0": 1, "pol1": 2, "pol2": 3, "pol3": 4, "expopow": 2}

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        name_func,
        mass_min,
        mass_max,
        nsigma_side_bands=0.0,
        mean_mass_peak=0.0,
        sigma_mass_peak=0.0,
        mean_sec_mass_peak=0.0,
        sigma_sec_mass_peak=0.0,
    ):
        if name_func not in self.__impl_func:
            raise ValueError(f"Function '{name_func}' not implemented")
        self.name_func = name_func
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.mean_mass_peak = mean_mass_peak
        self.delta_mass_peak = sigma_mass_peak * nsigma_side_bands
        self.mean_sec_mass_peak = mean_sec_mass_peak
        self.delta_sec_mass_peak = sigma_sec_mass_peak * nsigma_side_bands
        self.func_side_bands_callable = None
        self.func_full_callable = None

        self.remove_peak = False
        self.remove_sec_peak = False
        if self.mean_mass_peak > 0.0 and self.delta_mass_peak > 0.0:
            self.remove_peak = True
        if self.mean_sec_mass_peak > 0.0 and self.delta_sec_mass_peak > 0.0:
            self.remove_sec_peak = True

        self.mass_pi = TDatabasePDG.Instance().GetParticle(211).Mass()

    def _expo_normalised(self, x, par):
        """
        Exponential function normalised to its integral.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: expo slope
        """
        norm = par[0] * par[1] / (TMath.Exp(par[1] * self.mass_max) - TMath.Exp(par[1] * self.mass_min))
        return norm * TMath.Exp(par[1] * x[0])

    # pylint: disable=unused-argument
    def _pol0_normalised(self, x, par):
        """
        Constant function normalised to its integral.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
        """
        return par[0] / (self.mass_max - self.mass_min)

    def _pol1_normalised(self, x, par):
        """
        Linear function normalized to its integral.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: angular coefficient
        """
        return par[0] / (self.mass_max - self.mass_min) + par[1] * (x[0] - 0.5 * (self.mass_max + self.mass_min))

    def _pol2_normalised(self, x, par):
        """
        Second order polynomial function normalised to its integral.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: a
            par[2]: b
        """
        first_term = par[0] / (self.mass_max - self.mass_min)
        second_term = par[1] * (x[0] - 0.5 * (self.mass_max + self.mass_min))
        third_term = par[2] * (
            x[0] ** 2 - 1 / 3.0 * (self.mass_max**3 - self.mass_min**3) / (self.mass_max - self.mass_min)
        )
        return first_term + second_term + third_term

    def _pol3_normalised(self, x, par):
        """
        Third order polynomial function normalised to its integral.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: a
            par[2]: b
            par[3]: c
        """
        first_term = par[0] / (self.mass_max - self.mass_min)
        second_term = par[1] * (x[0] - 0.5 * (self.mass_max + self.mass_min))
        third_term = par[2] * (
            x[0] ** 2 - 1 / 3.0 * (self.mass_max**3 - self.mass_min**3) / (self.mass_max - self.mass_min)
        )
        fourth_term = par[3] * (
            x[0] ** 3 - 1 / 4.0 * (self.mass_max**4 - self.mass_min**4) / (self.mass_max - self.mass_min)
        )
        return first_term + second_term + third_term + fourth_term

    def _expopow_normalised(self, x, par):
        """
        Exponential times power law function normalized to its integral for D* background.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: expo slope
        """

        return par[0] * TMath.Sqrt(x[0] - self.mass_pi) * TMath.Exp(-1.0 * par[1] * (x[0] - self.mass_pi))

    def _func_side_bands(self, x, par):
        """
        Function where only sidebands are considered.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
        """
        if self.remove_peak and TMath.Abs(x[0] - self.mean_mass_peak) < self.delta_mass_peak:
            TF1.RejectPoint()
            return 0
        if self.remove_sec_peak and TMath.Abs(x[0] - self.mean_sec_mass_peak) < self.delta_sec_mass_peak:
            TF1.RejectPoint()
            return 0

        return getattr(self, self.__impl_func[self.name_func])(x, par)

    def get_func_side_bands(self, integral):
        """
        Return the ROOT.TF1 function defined on the sidebands

        Parameters
        --------------------------------------
        integral: integral of the histogram to fit, obtained with TH1.Integral('width')

        Returns
        ---------------------------------------
        func_bkg_side_bands: ROOT.TF1
            Background function
        """
        # trick to keep away the garbage collector
        self.func_side_bands_callable = self._func_side_bands
        func_bkg_side_bands = TF1(
            "bkgSBfunc", self.func_side_bands_callable, self.mass_min, self.mass_max, self.__n_pars[self.name_func]
        )
        func_bkg_side_bands.SetParName(0, "BkgInt")
        func_bkg_side_bands.SetParameter(0, integral)
        for i_par in range(1, self.__n_pars[self.name_func]):
            func_bkg_side_bands.SetParameter(i_par, 1.0)
        func_bkg_side_bands.SetLineColor(kBlue + 2)
        return func_bkg_side_bands

    def get_func_full_range(self, func):
        """
        Return the ROOT.TF1 function defined on the full range

        Parameters
        --------------------------------------
        func: function from get_func_side_bands() after the histogram fit

        Returns
        ---------------------------------------
        func_bkg: ROOT.TF1
            Background function
        """
        # trick to keep away the garbage collector
        self.func_full_callable = getattr(self, self.__impl_func[self.name_func])
        func_bkg = TF1("bkgFunc", self.func_full_callable, self.mass_min, self.mass_max, self.__n_pars[self.name_func])
        func_bkg.SetParName(0, "BkgInt")
        for i_par in range(0, self.__n_pars[self.name_func]):
            func_bkg.SetParameter(i_par, func.GetParameter(i_par))
        func_bkg.SetLineColor(kGreen + 2)
        return func_bkg
