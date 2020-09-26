import astropy.units as u


class SimulatedEventsInfo:
    """
    Information about all simulated events,
    needed for calculating event weights.

    Attributes
    ----------

    n_showers: int
        Total number of simulated showers. If reuse was used, this
        should already include the reuse.
    energy_min: u.Quantity[energy]
        Lower limit of the simulated energy range
    energy_max: u.Quantity[energy]
        Upper limit of the simulated energy range
    max_impact: u.Quantity[length]
        Maximum simulated impact parameter
    spectral_index: float
        Spectral Index of the simulated power law with sign included.
    """

    __slots__ = (
        "n_showers",
        "energy_min",
        "energy_max",
        "max_impact",
        "spectral_index",
        "viewcone",
    )

    @u.quantity_input(
        energy_min=u.TeV, energy_max=u.TeV, max_impact=u.m, viewcone=u.deg
    )
    def __init__(
        self, n_showers, energy_min, energy_max, max_impact, spectral_index, viewcone
    ):
        self.n_showers = n_showers
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.max_impact = max_impact
        self.spectral_index = spectral_index
        self.viewcone = viewcone

        if spectral_index > -1:
            raise ValueError("spectral index must be <= -1")

    @u.quantity_input(energy_bins=u.TeV)
    def calculate_n_showers(self, energy_bins):
        """
        Calculate number of showers that were simulated in the given interval

        Parameters
        ----------
        energy_bins: ``~astropy.units.Quantity``[energy]
            The interval edges for which to calculate the number of simulated showers

        Returns
        -------
        n_showers: ``~numpy.ndarray``
            The expected number of events inside each of the ``energy_bins``.
            This is a floating point number.
            The actual numbers will follow a poissionian distribution around this
            expected value.
        """
        bins = energy_bins.to_value(u.TeV)
        e_low = bins[:-1]
        e_high = bins[1:]

        int_index = self.spectral_index + 1
        e_min = self.energy_min.to_value(u.TeV)
        e_max = self.energy_max.to_value(u.TeV)

        e_term = e_low ** int_index - e_high ** int_index
        normalization = int_index / (e_max ** int_index - e_min ** int_index)

        return self.n_showers * normalization * e_term

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_showers={self.n_showers}, "
            f"energy_min={self.energy_min:.3f}, "
            f"energy_max={self.energy_max:.2f}, "
            f"spectral_index={self.spectral_index:.1f}, "
            f"max_impact={self.max_impact:.2f}, "
            f"viewcone={self.viewcone}"
            ")"
        )
