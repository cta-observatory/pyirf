.. _examples:

Examples
========

Calculating Sensitivity and IRFs for EventDisplay DL2 data
----------------------------------------------------------

The ``examples/calculate_eventdisplay_irfs.py`` file is
using ``pyirf`` to optimize cuts, calculate sensitivity and IRFs
and then store these to FITS files for DL2 event lists from EventDisplay.

The ROOT files were provided by Gernot Maier and converted to FITS format
using `the EventDisplay DL2 converter script <https://github.com/EventDisplay/Converters>`_.
The resulting FITS files are the input to the example and can be downloaded using:

.. code:: bash

    ./download_private_data.sh

This requires ``curl`` and ``unzip`` to be installed.
The download is password protected, please ask one of the maintainers for the
password.

A detailed explanation of the contents of such DL2 files can be found
`here (internal) <https://cta.cloud.xwiki.com/xwiki/wiki/aswg/view/Main/Eventdisplay%20Prod3b%20DL2%20Lists/>`_.

The example can then be run from the root of the repository after installing pyirf
by running:

.. code:: bash

    python examples/calculate_eventdisplay_irfs.py


A jupyter notebook plotting the results and comparing them to the EventDisplay output
is available in ``examples/comparison_with_EventDisplay.ipynb``


Visualization of the included Flux Models
-----------------------------------------

The ``examples/plot_spectra.py`` visualizes the Flux models included
in ``pyirf`` for Crab Nebula, cosmic ray and electron flux.
