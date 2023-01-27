from .base_interpolators import BinnedInterpolator


class QuantileInterpolator(BinnedInterpolator):
    def __init__(
        self, grid_points, bin_edges, bin_contents, axis, quantile_resolution=1e-3
    ):
        """BinnedInterpolator constructor

        Parameters
        ----------
            grid_points: np.ndarray
                Grid points at which interpolation templates exist
            bin_edges: np.ndarray
                Edges of the data binning
            bin_content: np.ndarray
                Content of each bin in bin_edges for
                each point in grid_points. First dimesion has to correspond to number
                of grid_points, second dimension has to correspond to number of bins for
                the quantity that should be interpolated (e.g. the Migra axis for EDisp)
            axis:

            quantile_resolution:

        Raises
        ------
            TypeError:
                When bin_edges is not a np.ndarray
            TypeError:
                When bin_content is not a np.ndarray
            ValueError:
                When number of bins in bin_edges and contents bin_contents is
                not matching
            ValueError:
                When number of histograms in bin_contents and points in grid_points
                is not matching

        Note
        ----
            Also calls pyirf.interpolators.BaseInterpolators.__call__
        """
        super().__init__(grid_points, bin_edges, bin_contents)

        self.axis = axis
        self.quantile_resolution = quantile_resolution

    def _interpolate(self, target_point, **kwargs):
        return None
