"""Classes for sampling model seeds and spectral templates."""

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
    seed : int
        The base seed for all sampling.
    """

    def __init__(
        self,
        seed: int = 2744,
    ):

        super().__init__(self, seed)

        from numpy.random import PCG64, Generator

        rng = np.random.Generator(np.random.PCG64())
