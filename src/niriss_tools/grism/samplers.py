"""Classes for sampling model seeds and spectral templates."""

import multiprocessing
from pathlib import Path

import h5py
import numpy as np
import ast

from bagpipes import config
from niriss_tools.grism.specgen import BagpipesSpecGenerator, air_to_vac
from functools import partial
from itertools import repeat
from grizli.utils_numba.interp import interp_conserve_c


class TemplateSampler:
    """
    The base class for spectral template sampling.

    Parameters
    ----------
    seed : int
        The base seed for all sampling.
    **kwargs : dict
        Any additional keyword parameters.
    """

    def __init__(self, seed: int = 2744, **kwargs):

        self.seed = seed

    def gen_model_seeds_from_iter(
        self, iter_seed: int, n_samples: int, **kwargs
    ) -> list[int]:
        """
        Construct a list of model seeds for a given iteration.

        For a given `iter_seed`, this must return the exact same set of
        model seeds. This method must be implemented by subclasses.

        Parameters
        ----------
        iter_seed : int
            The seed for a given iteration, typically the number of
            iterations already performed.
        n_samples : int
            The number of model seeds to generate.
        **kwargs : dict
            Any additional keyword parameters.

        Returns
        -------
        list[int]
            The list of model seeds.
        """

        raise NotImplementedError("Subclasses should implement this method.")


def init_bagpipes_spec_gen(fit_instructions, veldisp, spec_wavs):

    global spec_generator
    spec_generator = BagpipesSpecGenerator(
        fit_instructions=fit_instructions, veldisp=veldisp, spec_wavs=spec_wavs
    )


class BagpipesTemplateSampler(TemplateSampler):
    """
    A subclass of TemplateSampler to be used with SED fits from `bagpipes`.

    Parameters
    ----------
    posterior_dir : Path
        The directory containing the posterior ``*.h5`` files.
    seed : int, optional
        The base seed for all sampling, by default ``2744``.
    cpu_count : int, optional
        The number of CPUs to use for multiprocessing. Defaults to the
        number returned by `~multiprocessing.cpu_count()`.
    """

    def __init__(
        self,
        posterior_dir: Path,
        seed: int = 2744,
        cpu_count: int = multiprocessing.cpu_count(),
        cache_spectra: bool = True,
        spec_wavs: np.ndarray = np.arange(10000.0, 23000.0, 22.5),
        veldisp: float = 250,
    ):

        super().__init__(seed)

        self.base_rng = np.random.Generator(np.random.PCG64(self.seed))

        self.posterior_ids = [f.stem for f in posterior_dir.glob("*.h5")]
        try:
            self.posterior_ids.sort(key=int)
        except:
            self.posterior_ids.sort()

        self.cpu_count = cpu_count

        self.cache_spectra = True

        self.fit_instructions = self.load_fit_instructions(
            posterior_dir / f"{self.posterior_ids[0]}.h5"
        )

        self.veldisp = veldisp
        self.spec_wavs = spec_wavs

        with multiprocessing.Pool(self.cpu_count) as pool:

            params_lists = pool.map(
                self._load_model_params,
                [posterior_dir / f"{i}.h5" for i in self.posterior_ids],
            )

        params_array = np.concatenate(params_lists, axis=0)

        u, inv = np.unique(params_array, return_inverse=True)

        self.all_models_params = u

        # print (ast.literal_eval(self.all_models_params[0]))

        # exit()

        self.posterior_params_map = inv.reshape(len(params_lists), -1)

        # exit()

        if self.cache_spectra:
            # from niriss_tools.grism.specgen import init_bagpipes_spec_gen

            with multiprocessing.Pool(
                processes=self.cpu_count,
                initializer=init_bagpipes_spec_gen,
                initargs=(self.fit_instructions, self.veldisp, self.spec_wavs),
            ) as pool:

                spec_lists, line_flux_dicts = zip(
                    *pool.map(
                        self.worker_gen_spec_and_fluxes,
                        self.all_models_params[:10],
                    )
                )

                self.model_spectra = np.array(spec_lists)
                print(self.model_spectra.shape)

                # # print (spec_lists)
                # import matplotlib.pyplot as plt

                # for s in spec_lists:
                #     plt.plot(self.spec_wavs, s)
                # plt.show()

                # print (line_flux_dicts[0])

                # self.line_names = np.array(list(line_flux_dicts[0].keys()))
                self.line_names = np.array(config.line_names)
                self.line_wavs_rf = np.array(config.line_wavs)

                merged_line_flux_dict = {
                    k: [d.get(k, np.nan) for d in line_flux_dicts]
                    for k in self.line_names
                }
                self.line_fluxes = np.array(list(merged_line_flux_dict.values()))

                # print (merged_line_flux_dict)

                # dt = np.dtype([(k, np.array([v]).dtype) for k, v in line_flux_dicts[0].items()])
                # values = [tuple(d[key] for key in dt.names) for d in line_flux_dicts]
                # line_fluxes = np.array(values, dtype=dt)

                # print (line_fluxes)

        dummy_spec_gen = BagpipesSpecGenerator(
            self.fit_instructions, self.veldisp, self.spec_wavs
        )
        dummy_spec_gen.sample(ast.literal_eval(self.all_models_params[0]))

        model_comp = dummy_spec_gen.model_components

        self.param_names = dummy_spec_gen.params
        model_wavs_rf = dummy_spec_gen.model_gal.wavelengths

        # print (self.param_names)

        model_idxs = np.arange(10)

        if "redshift" in self.param_names:
            z_idx = (np.array(self.param_names) == "redshift").argmax()
            model_redshifts = np.array(
                [ast.literal_eval(m)[z_idx] for m in self.all_models_params[model_idxs]]
            )

        print(model_redshifts)

        emline = "H  1  6562.80A"
        # emline = ["H  1  6562.80A"]
        emline = ["H  1  6562.80A", "N  2  6583.45A", "N  2  6548.05A"]
        emline = [
            "N  2  6583.45A",
            "H  1  6562.80A",
            "N  2  6548.05A",
            "H  1  4861.32A",
            "O  3  5006.84A",
            "O  3  4958.91A",
            "O  2  3726.03A",
            "O  2  3728.81A",
            "S  2  6730.82A",
            "S  2  6716.44A",
        ]

        # Ensure that emission lines will always be an array
        emline = np.atleast_1d(emline)

        # Find the exact index of each emission line name
        # (order must be preserved)
        sorter = np.argsort(self.line_names)
        emline_idxs = sorter[np.searchsorted(self.line_names, emline, sorter=sorter)]

        print(emline_idxs)
        print(self.line_fluxes.shape)

        emline_wavs_rf = self.line_wavs_rf[emline_idxs] * (
            1 + (model_comp["nebular"].get("velshift", 0) / (3 * 10**5))
        )

        wav_idxs = np.abs(model_wavs_rf[:, np.newaxis] - emline_wavs_rf).argmin(axis=0)

        line_templates = np.zeros((len(model_idxs), len(model_wavs_rf)))

        for wav_idx, line_idx in zip(wav_idxs, emline_idxs):
            width = (model_wavs_rf[wav_idx + 1] - model_wavs_rf[wav_idx - 1]) / 2

            print(wav_idx, width)
            line_templates[:, wav_idx] = self.line_fluxes[line_idx, model_idxs] / width

        # # zplusone = model_comp["redshift"] + 1.0
        # print(line_templates)

        print(line_templates.__sizeof__())
        print(line_templates.shape)

        # Replicate the same sampling used within bagpipes
        if "veldisp" in list(model_comp):
            vres = 3 * 10**5 / config.R_spec / 2.0
            sigma_pix = model_comp["veldisp"] / vres
            k_size = 4 * int(sigma_pix + 1)
            x_kernel_pix = np.arange(-k_size, k_size + 1)

            kernel = np.exp(-(x_kernel_pix**2) / (2 * sigma_pix**2))
            kernel /= np.trapezoid(kernel)  # Explicitly normalise kernel

            model_wavs_rf = model_wavs_rf[k_size:-k_size]

            convolved_line_templates = np.apply_along_axis(
                np.convolve, -1, line_templates, kernel, mode="valid"
            )

        else:
            convolved_line_templates = line_templates

        redshifted_wavs = (1 + model_redshifts)[:, np.newaxis] * model_wavs_rf

        # if "R_curve" in list(model_comp):
        #     oversample = 4  # Number of samples per FWHM at resolution R
        #     new_wavs = dummy_spec_gen.model_gal._get_R_curve_wav_sampling(
        #         oversample=oversample
        #     )

        #     # with multiprocessing.Pool(
        #     #     processes=self.cpu_count,
        #     # ) as pool:
        #     #     resampled_spectra = np.array(
        #     #         pool.starmap(
        #     #             interp_conserve_c,
        #     #             zip(
        #     #                 repeat(new_wavs), redshifted_wavs, convolved_line_templates
        #     #             ),
        #     #         )
        #     #     )
        #     with multiprocessing.Pool(
        #         processes=self.cpu_count,
        #     ) as pool:
        #         resampled_spectra = np.array(
        #             pool.starmap(
        #                 interp_conserve_c,
        #                 zip(
        #                     repeat(new_wavs),
        #                     redshifted_wavs,
        #                     convolved_line_templates,
        #                 ),
        #             )
        #         )
        #     redshifted_wavs = new_wavs

        #     sigma_pix = oversample / 2.35  # sigma width of kernel in pixels
        #     k_size = 4 * int(sigma_pix + 1)
        #     x_kernel_pix = np.arange(-k_size, k_size + 1)

        #     kernel = np.exp(-(x_kernel_pix**2) / (2 * sigma_pix**2))
        #     kernel /= np.trapezoid(kernel)  # Explicitly normalise kernel

        #     # Disperse non-uniformly sampled spectrum
        #     spectrum = np.convolve(spectrum, kernel, mode="valid")
        #     redshifted_wavs = redshifted_wavs[k_size:-k_size]

        vac_redshifted_wavs = air_to_vac(redshifted_wavs)

        with multiprocessing.Pool(
            processes=self.cpu_count,
        ) as pool:
            model_line_fluxes = np.array(
                pool.starmap(
                    interp_conserve_c,
                    zip(
                        repeat(self.spec_wavs),
                        vac_redshifted_wavs,
                        convolved_line_templates,
                    ),
                )
            )

        model_line_fluxes = np.zeros(
            (convolved_line_templates.shape[0], len(self.spec_wavs))
        )

        for i, (w, c) in enumerate(zip(vac_redshifted_wavs, convolved_line_templates)):
            model_line_fluxes[i] = interp_conserve_c(self.spec_wavs, w, c)

        import matplotlib.pyplot as plt

        # for i, (w, c) in enumerate(zip(vac_redshifted_wavs, convolved_line_templates)):
        #     plt.plot(w, c)

        model_line_fluxes /= (1 + model_redshifts)[:, np.newaxis]

        # for l in model_line_fluxes:
        #     plt.plot(self.spec_wavs, l)
        #     # plt.plot(self.spec_wavs, l / 10**-29 * 2.9979 * 10**18 / self.spec_wavs**2)

        # for m in model_idxs:
        #     plt.plot(self.spec_wavs, self.model_spectra[m])

        for m, l in zip(model_idxs, model_line_fluxes):
            plt.plot(self.spec_wavs, self.model_spectra[m] - l)

        plt.xlim(xmin=1e4, xmax=3e4)
        plt.show()

        exit()

        if self.spec_units == "mujy":
            fluxes /= 10**-29 * 2.9979 * 10**18 / self.spec_wavs**2

        # self.spectrum = np.c_[self.spec_wavs, fluxes]

        # print(len(model_wavs_rf))

        # print(emline_wavs_rf)
        # print(ind)
        # print(model_wavs_rf[ind])

    @staticmethod
    def worker_gen_spec_and_fluxes(param_vector: str):

        return spec_generator.sample(
            ast.literal_eval(param_vector), return_line_fluxes=True
        )

        # print (params_lists)

        # array 1:
        # 1 spectrum per row (M x spec wavs)

        # array 2: self.all_models_params
        # model id, aka string of model parameters (M,)

        # array 3: self.posterior_ids
        # N posterior ids

        # array 4: self.posterior_params_map
        # rows in array 1 corresponding to model id in array 2 (N x 500(?))

    @staticmethod
    def _load_model_params(posterior_path: Path) -> np.ndarray:
        """
        Convert a bagpipes posterior object to a 1D array of strings.

        Each element in the array is a string-formatted list of the input
        model parameters.

        Parameters
        ----------
        posterior_path : Path
            The path of the posterior object.

        Returns
        -------
        np.ndarray
            A 1D array of model parameters.
        """

        with h5py.File(posterior_path, "r") as post_file:
            samples2d = np.array(post_file["samples2d"])

        return np.array([str(r.tolist()) for r in samples2d])

    @staticmethod
    def load_fit_instructions(posterior_path: Path) -> dict:
        """
        Load the bagpipes fit instructions from a posterior object.

        Parameters
        ----------
        posterior_path : Path
            The path of the posterior object.

        Returns
        -------
        dict
            The fit instructions as a key/value dictionary.
        """

        with h5py.File(
            posterior_path,
            "r",
        ) as test_post:

            fit_info_str = test_post.attrs["fit_instructions"]
            fit_info_str = fit_info_str.replace("array", "np.array")
            fit_info_str = fit_info_str.replace("float", "np.float")
            fit_info_str = fit_info_str.replace("np.np.", "np.")
            fit_instructions = eval(fit_info_str)

        return fit_instructions

    # Cache the model seeds in self
    # e.g. calling `self.sample_spec_from_iter(iter_seed, posterior_id)`
    # checks if self.model_seeds[iter_seed] already exists

    def gen_model_seeds_from_iter(
        self, iter_seed: int, n_samples: int, **kwargs
    ) -> list[int]:
        """
        Construct a list of model seeds for a given iteration.

        Parameters
        ----------
        iter_seed : int
            The seed for a given iteration, typically the number of
            iterations already performed.
        n_samples : int
            The number of model seeds to generate.
        **kwargs : dict
            Any additional keyword parameters.

        Returns
        -------
        list[int]
            The list of model seeds.
        """

        iter_rng = np.random.Generator(np.random.PCG64(self.seed + iter_seed))

        init_samples = iter_rng.random(size=n_samples)

        return init_samples


if __name__ == "__main__":

    posterior_dir = Path(
        "/media/sharedData/data/2025_12_06_glass-a2744/glass_niriss_bcgs/"
        "reduction_v9-TEST/sed_fitting/pipes/posterior/"
        "3070_colour_3_10_jwst-nircam-f150w"
    )

    template_sampler = BagpipesTemplateSampler(posterior_dir=posterior_dir)

    print(template_sampler.posterior_ids)
    print(template_sampler.fit_instructions)
    # print
