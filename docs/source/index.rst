.. JaVAD documentation master file, created by
   sphinx-quickstart on Thu Dec 19 11:05:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

JaVAD: Just Another Voice Activity Detector
===========================================

Installation
============

You can install the package using pip:

.. code-block:: bash

    pip install javad

To install JaVAD with additional features: audio loading, quick processing, use following command:

.. code-block:: bash

    pip install javad[extras]

Requirements
============

- Python 3.8+
- PyTorch 2.0.0+
- NumPy 1.20.0+
- Optional, if using with [extras] (sound loading, Trainer): 
    - soundfile 0.12.0+,
    - torchvision 0.10.0+,
    - spgdataset 0.0.2+,
    - tqdm

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   main
   extras

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`