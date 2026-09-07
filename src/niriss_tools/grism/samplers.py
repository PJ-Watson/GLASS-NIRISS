"""Classes for sampling model seeds and spectral templates."""

import multiprocessing
from pathlib import Path

import h5py
import numpy as np


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
    ):

        super().__init__(seed)

        self.base_rng = np.random.Generator(np.random.PCG64(self.seed))

        self.posterior_ids = [f.stem for f in posterior_dir.glob("*.h5")]
        try:
            self.posterior_ids.sort(key=int)
        except:
            self.posterior_ids.sort()

        self.cpu_count = cpu_count

        self.fit_instructions = self.load_fit_instructions(
            posterior_dir / f"{self.posterior_ids[0]}.h5"
        )

        # # Testing
        # params_array = np.array([
        #     "a", "b", "a", "a", "c", "b"
        # ])

        # u, inv = np.unique(params_array, return_inverse=True)
        # print (u)
        # print (inv)
        # new = inv.reshape(3, -1)
        # # new = inv[2, :]
        # print (new[0])

        with multiprocessing.Pool(self.cpu_count) as pool:

            params_lists = pool.map(
                self._load_model_params,
                [posterior_dir / f"{i}.h5" for i in self.posterior_ids],
            )

        params_array = np.concatenate(params_lists, axis=0)

        u, inv = np.unique(params_array, return_inverse=True)

        self.all_models_params = u

        self.posterior_params_map = inv.reshape(len(params_lists), -1)

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
