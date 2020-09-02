.. _io:

Input / Output
==============

Introduction
------------

This module contains a set of classes and functions for

- configuration input,
- DL2 input,
- DL3 output.

The precise structure is currently under development, but in general there is:

- a reader for each data format / pipeline depending on different DL2 file format and internal structure,
- a mapper that reads user-defined DL2 column names from the configuration file (see :ref:`resources` for an example) into GADF format,
- the only output format is defined by the latest version of the GADF.

Most of the GADF column names are defined in the configuration file under the
section 'column_definition'.

Reference/API
-------------

.. automodapi:: pyirf.io
   :no-inheritance-diagram:
