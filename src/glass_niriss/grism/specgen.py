"""
Functions and classes for generating specific types of model spectra.
"""

import warnings
from copy import deepcopy

import bagpipes
import numpy as np
import spectres
from bagpipes import config, utils
from bagpipes.input.spectral_indices import measure_index
from bagpipes.models import model_galaxy as BagpipesModelGalaxy
from numpy.typing import ArrayLike

# from bagpipes.models.nebular_model import nebular

__all__ = ["ExtendedModelGalaxy"]


# T
def H(a: float, x: ArrayLike) -> ArrayLike:
    """
    The Voigt-Hjerting profile.

    Based on the numerical approximation by `Garcia+06
    <https://ui.adsabs.harvard.edu/abs/2006MNRAS.369.2025T/>`__.

    Parameters
    ----------
    a : float
        The damping parameter.
    x : ArrayLike
        The array of values over which the profile should be evaluated.

    Returns
    -------
    ArrayLike
        The numerical approximation to the Voigt-Hjerting profile.
    """
    P = x**2
    H0 = np.exp(-(x**2))
    Q = 1.5 * x ** (-2)
    return H0 - a / np.sqrt(np.pi) / P * (
        H0**2 * (4.0 * P**2 + 7.0 * P + 4.0 + Q) - Q - 1.0
    )


def addAbs(wl_mod: list, t: float, zabs: float) -> float:
    """
    A function that calculates the absorption from foreground source.

    Parameters
    ----------
    wl_mod : list
        The wavelength values, in units of :math:`\\mathring{A}`.
    t : float
        The hydrogen column density in units of :math:`\\text{cm}^{-2}`.
    zabs : float
        The redshift of the absorption source.

    Returns
    -------
    float
        The absorption fraction, :math:`e^{-\\tau}`.
    """

    # Constants
    m_e = 9.1095e-28
    e = 4.8032e-10
    c = 2.998e10
    lamb = 1215.67
    f = 0.416
    gamma = 6.265e8
    broad = 1

    C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1e-8 / m_e / c / broad
    a = lamb * 1.0e-8 * gamma / (4.0 * np.pi * broad)
    dl_D = broad / c * lamb
    x = (wl_mod / (zabs + 1.0) - lamb) / dl_D + 0.01

    # Optical depth
    tau = np.array([C_a * t * H(a, x)], dtype=np.float64)
    return np.exp(-tau)[0]


class ExtendedModelGalaxy(BagpipesModelGalaxy):
    """
    An extension of `bagpipes.models.model_galaxy`.

    This class allows a model spectrum to be generated, with one or more
    nebular emission lines excluded.
    """

    def update(
        self,
        model_components: dict,
        cont_only: bool = False,
        rm_line: list[str] | None = None,
    ):
        """
        Update the model outputs based on ``model_components``.

        Update the model outputs to reflect new parameter values in
        the model_components dictionary. Note that only the changing of
        numerical values is supported.

        Parameters
        ----------
        model_components : dict
            A dictionary containing information about the model to be
            generated.
        cont_only : bool, optional
            Generate only the continuum spectra, with no nebular emission
            at all. By default ``False``.
        rm_line : list[str] | None, optional
            The names of one or more lines to exclude from the model
            spectrum, based on the `Cloudy <https://www.nublado.org/>`__
            naming convention (see `here
            <https://bagpipes.readthedocs.io/en/latest/model_galaxies.html#getting-observables-line-fluxes>`__
            for more details). By default ``None``.
        """

        self.model_comp = model_components
        self.sfh.update(model_components)
        if self.dust_atten:
            self.dust_atten.update(model_components["dust"])

        # If the SFH is unphysical do not caclulate the full spectrum
        if self.sfh.unphysical:
            warnings.warn(
                "The requested model includes stars which formed "
                "before the Big Bang, no spectrum generated.",
                RuntimeWarning,
            )

            self.spectrum_full = np.zeros_like(self.wavelengths)
            self.uvj = np.zeros(3)

        else:
            self._calculate_full_spectrum(
                model_components, cont_only=cont_only, rm_line=rm_line
            )

        if self.spec_wavs is not None:
            self._calculate_spectrum(model_components)

        # Add any AGN component:
        if self.agn:
            self.agn.update(self.model_comp["agn"])
            agn_spec = self.agn.spectrum
            agn_spec *= self.igm.trans(self.model_comp["redshift"])

            self.spectrum_full += agn_spec / (1.0 + self.model_comp["redshift"])

            if self.spec_wavs is not None:
                zplus1 = self.model_comp["redshift"] + 1.0
                agn_interp = np.interp(
                    self.spec_wavs,
                    self.wavelengths * zplus1,
                    agn_spec / zplus1,
                    left=0,
                    right=0,
                )

                self.spectrum[:, 1] += agn_interp

        if self.filt_list is not None:
            self._calculate_photometry(model_components["redshift"])

        if not self.sfh.unphysical:
            self._calculate_uvj_mags()

        # Deal with any spectral index calculations.
        if self.index_list is not None:
            self.index_names = [ind["name"] for ind in self.index_list]

            self.indices = np.zeros(len(self.index_list))
            for i in range(self.indices.shape[0]):
                self.indices[i] = measure_index(
                    self.index_list[i], self.spectrum, model_components["redshift"]
                )

    def _calculate_full_spectrum(
        self,
        model_comp: dict,
        cont_only: bool = True,
        rm_line: list[str] | None = None,
    ):
        """
        Calculate a full model spectrum given a set of model components.

        This method combines the models for the various emission
        and absorption processes to generate the internal full galaxy
        spectrum held within the class. The `_calculate_photometry` and
        `_calculate_spectrum` methods generate observables using this
        internal full spectrum.

        Parameters
        ----------
        model_comp : dict
            A dictionary containing information about the model to be
            generated.
        cont_only : bool, optional
            Generate only the continuum spectra, with no nebular emission
            at all. By default ``False``.
        rm_line : list[str] | None, optional
            The names of one or more lines to exclude from the model
            spectrum, based on the `Cloudy <https://www.nublado.org/>`__
            naming convention (see `here
            <https://bagpipes.readthedocs.io/en/latest/model_galaxies.html#getting-observables-line-fluxes>`__
            for more details). By default ``None``.
        """

        t_bc = 0.01
        if "t_bc" in list(model_comp):
            t_bc = model_comp["t_bc"]

        spectrum_bc, spectrum = self.stellar.spectrum(self.sfh.ceh.grid, t_bc)
        em_lines = np.zeros(config.line_wavs.shape)

        if self.nebular and not cont_only:
            grid = np.copy(self.sfh.ceh.grid)

            if "metallicity" in list(model_comp["nebular"]):
                nebular_metallicity = model_comp["nebular"]["metallicity"]
                neb_comp = deepcopy(model_comp)
                for comp in list(neb_comp):
                    if isinstance(neb_comp[comp], dict):
                        neb_comp[comp]["metallicity"] = nebular_metallicity

                self.neb_sfh.update(neb_comp)
                grid = self.neb_sfh.ceh.grid

            em_lines += self.nebular.line_fluxes(
                grid, t_bc, model_comp["nebular"]["logU"]
            )

            # All stellar emission below 912A goes into nebular emission
            spectrum_bc[self.wavelengths < 912.0] = 0.0
            spectrum_bc += self.nebular.spectrum(
                grid, t_bc, model_comp["nebular"]["logU"]
            )

            if rm_line is not None:
                rm_line = np.atleast_1d(rm_line).ravel()
                for rm in rm_line:
                    rm_line_idx = np.argwhere(config.line_names == rm)

                    line_wav_shift = config.line_wavs[rm_line_idx] * (
                        1 + (model_comp["nebular"].get("velshift", 0) / (3 * 10**5))
                    )
                    ind = np.abs(self.wavelengths - line_wav_shift).argmin()
                    if ind != 0 and ind != self.wavelengths.shape[0] - 1:
                        width = (
                            self.wavelengths[ind + 1] - self.wavelengths[ind - 1]
                        ) / 2
                        spectrum_bc[ind] -= em_lines[rm_line_idx] / width

        # Add attenuation due to stellar birth clouds.
        if self.dust_atten:
            dust_flux = 0.0  # Total attenuated flux for energy balance.

            # Add extra attenuation to birth clouds.
            eta = 1.0
            if "eta" in list(model_comp["dust"]):
                eta = model_comp["dust"]["eta"]
                bc_Av_reduced = (eta - 1.0) * model_comp["dust"]["Av"]
                bc_trans_red = 10 ** (-bc_Av_reduced * self.dust_atten.A_cont / 2.5)
                spectrum_bc_dust = spectrum_bc * bc_trans_red
                dust_flux += np.trapz(
                    spectrum_bc - spectrum_bc_dust, x=self.wavelengths
                )

                spectrum_bc = spectrum_bc_dust

            # Attenuate emission line fluxes.
            bc_Av = eta * model_comp["dust"]["Av"]
            em_lines *= 10 ** (-bc_Av * self.dust_atten.A_line / 2.5)

        spectrum += spectrum_bc  # Add birth cloud spectrum to spectrum.

        # Add attenuation due to the diffuse ISM.
        if self.dust_atten:
            trans = 10 ** (-model_comp["dust"]["Av"] * self.dust_atten.A_cont / 2.5)
            dust_spectrum = spectrum * trans
            dust_flux += np.trapz(spectrum - dust_spectrum, x=self.wavelengths)

            spectrum = dust_spectrum
            self.spectrum_bc = spectrum_bc * trans

            # Add dust emission.
            qpah, umin, gamma = 2.0, 1.0, 0.01
            if "qpah" in list(model_comp["dust"]):
                qpah = model_comp["dust"]["qpah"]

            if "umin" in list(model_comp["dust"]):
                umin = model_comp["dust"]["umin"]

            if "gamma" in list(model_comp["dust"]):
                gamma = model_comp["dust"]["gamma"]

            spectrum += dust_flux * self.dust_emission.spectrum(qpah, umin, gamma)

        spectrum *= self.igm.trans(model_comp["redshift"])

        if "dla" in list(model_comp):
            spectrum *= addAbs(
                self.wavelengths * self.model_comp["redshift"],
                self.model_comp["dla"]["t"],
                self.model_comp["dla"]["zabs"],
            )

        if self.dust_atten:
            self.spectrum_bc *= self.igm.trans(model_comp["redshift"])

        # Convert from luminosity to observed flux at redshift z.
        self.lum_flux = 1.0
        if model_comp["redshift"] > 0.0:
            ldist_cm = (
                3.086
                * 10**24
                * np.interp(
                    model_comp["redshift"],
                    utils.z_array,
                    utils.ldist_at_z,
                    left=0,
                    right=0,
                )
            )

            self.lum_flux = 4 * np.pi * ldist_cm**2

        spectrum /= self.lum_flux * (1.0 + model_comp["redshift"])

        if self.dust_atten:
            self.spectrum_bc /= self.lum_flux * (1.0 + model_comp["redshift"])

        em_lines /= self.lum_flux

        # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
        spectrum *= 3.826 * 10**33

        if self.dust_atten:
            self.spectrum_bc *= 3.826 * 10**33

        em_lines *= 3.826 * 10**33

        self.line_fluxes = dict(zip(config.line_names, em_lines))

        self.spectrum_full = spectrum
