.. _io:

Input / Output
==============

Introduction
------------

This module contains a set of classes and functions for

- configuration input,
- DL2 input,
- DL3 output.

The precise structure is currently under development, but we expect that:

- there should be a reader for each data format (we expect to support only FITS and HDF5 for the moment),
- each reader should read a ``pipeline`` argument, depending on different DL2 file format and internal structure,
- every reader calls a mapper that reads user-defined DL2 column names from the configuration file (see :ref:`resources` for an example) into an internal data format,
- the output format for the IRFs is based on the latest version of the GADF plus any integration we think is necessary.

Most of the required column names for the interanl data format are defined in the configuration file under the
section ``column_definition``.

The current output is composed by:

- IRFs in FITS format,
- a table also in FITS format containing information about the cuts applied to generate the IRFs,
- an HDF5 table for each particle type containing the DL2 events selected with those cuts
- a diagnostic folder containing intermediate plots in form of PDF and pickle files

Reference/API
-------------

.. automodapi:: pyirf.io
   :no-inheritance-diagram:
