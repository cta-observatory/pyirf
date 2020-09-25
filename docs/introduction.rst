.. _introduction:

Introduction to ``pyirf``
=========================


``pyirf`` aims to provide functions to calculate the Instrument Response Functions (IRFs)
and sensitivity for Imaging Air Cherenkov Telescopes.

To support a wide range of use cases, ``pyirf`` opts for a library approach of
composable building blocks with well-defined inputs and outputs.

For more information on IRFs, have a look at the `Specification of the Data Formats for Gamma-Ray Astronomy`_
or the `ctools documentation on IRFs <http://cta.irap.omp.eu/ctools/users/user_manual/irf_cta.html>`_.


Currently, ``pyirf`` allows calculation of the usual factorization of the IRFs into:

* Effective area
* Energy migration
* Point spread function

Additionally, functions for calculating point-source flux sensitivity are provided.
Flux sensitivity is defined as the smallest flux an IACT can detect with a certain significance,
usually 5 σ according to the Li&Ma likelihood ratio test, in a specified amount of time.

``pyirf`` also provides functions to calculate event weights, that are needed
to translate a set of simulations to a physical flux for calculating sensitivity
and expected event counts.

Event selection with energy dependent cuts is also supported,
but at the moment, only rudimentary functions to find optimal cuts are provided.


Input formats
-------------

``pyirf`` does not rely on specific input file formats.
All functions take ``numpy`` arrays, astropy quantities or astropy tables for the
required data and also return the results as these objects.

``~pyirf.io`` provides functions to export the internal IRF representation
to FITS files following the `Specification of the Data Formats for Gamma-Ray Astronomy`_


DL2 event lists
^^^^^^^^^^^^^^^

Most functions for calculating IRFs need DL2 event lists as input.
We use ``~astropy.table.QTable`` instances for this.
``QTable`` are very similar to the standard ``~astropy.table.Table``,
but offer better interoperability with ``astropy.units.Quantity``.

We expect certain columns to be present in the tables with the appropriate units.
To learn which functions need which columns to be present, have a look at the :ref:`_pyirf_api_docs`

Most functions only need a small subgroup of these columns.

.. table:: Column definitions for DL2 event lists

    +-------------------+--------+----------------------------------------------------+
    | Column            | Unit   | Explanation                                        |
    +===================+========+====================================================+
    | true_energy       | TeV    | True energy of the simulated shower                |
    +-------------------+--------+----------------------------------------------------+
    | weight            |        | Event weight                                       |
    +-------------------+--------+----------------------------------------------------+
    | source_fov_offset | deg    | Distance of the true origin to the FOV center      |
    +-------------------+--------+----------------------------------------------------+
    | true_alt          | deg    | True altitude of the shower origin                 |
    | true_alt          | deg    | True altitude of the shower origin                 |
    +-------------------+--------+----------------------------------------------------+
    | true_az           | deg    | True azimuth of the shower origin                  |
    +-------------------+--------+----------------------------------------------------+
    | pointing_alt      | deg    | Altitude of the field of view center               |
    +-------------------+--------+----------------------------------------------------+
    | pointing_az       | deg    | Azimuth of the field of view center                |
    +-------------------+--------+----------------------------------------------------+
    | reco_energy       | TeV    | Reconstructed energy of the simulated shower       |
    +-------------------+--------+----------------------------------------------------+
    | reco_alt          | deg    | Reconstructed altitude of shower origin            |
    +-------------------+--------+----------------------------------------------------+
    | reco_az           | deg    | Reconstructed azimuth of shower origin             |
    +-------------------+--------+----------------------------------------------------+
    | gh_score          |        | Gamma/Hadron classification output                 |
    +-------------------+--------+----------------------------------------------------+
    | multiplicity      |        | Number of telescopes used in the reconstruction    |
    +-------------------+--------+----------------------------------------------------+


.. _Specification of the Data Formats for Gamma-Ray Astronomy: https://gamma-astro-data-formats.readthedocs.io
