emuPK
======

*3D Matter Power Spectrum Emulator for Weak Lensing Analysis*

.. image:: https://img.shields.io/badge/astro--ph-arXiv%3A2005.06551-red?style=flat
    :target: https://arxiv.org/abs/2005.06551

.. image:: https://readthedocs.org/projects/blendz/badge/
    :target: https://emupk.readthedocs.io/en/latest

.. image:: https://img.shields.io/github/license/Harry45/emuPK
    :target: https://github.com/Harry45/emuPK



You can read the full documentation `here <https://emupk.readthedocs.io/en/latest/>`_.

Citation
--------

If you use this code in your research, please cite `this paper <https://arxiv.org/abs/2005.06551>`_:

.. code-block:: tex

	@ARTICLE{2020MNRAS.497.2213M,
	       author = {{Mootoovaloo}, Arrykrishna and {Heavens}, Alan F. and
	         {Jaffe}, Andrew H. and {Leclercq}, Florent},
	        title = {Parameter inference for weak lensing using Gaussian Processes and MOPED},
	      journal = {\mnras},
	     keywords = {gravitational lensing: weak, methods: data analysis, methods: statistical, cosmological parameters, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics},
	         year = 2020,
	        month = jul,
	       volume = {497},
	       number = {2},
	        pages = {2213-2226},
	          doi = {10.1093/mnras/staa2102},
	archivePrefix = {arXiv},
	       eprint = {2005.06551},
	 primaryClass = {astro-ph.CO},
	       adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.2213M},
	      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
	}

Directory Structure
-------------------

.. code:: bash

	.
	├── docs
	│   ├── build
	│   ├── make.bat
	│   ├── Makefile
	│   └── source
	├── emulator
	│   ├── class_settings.ipynb
	│   ├── cosmoclass
	│   ├── cosmogp.py
	│   ├── gp_moped.py
	│   ├── gps
	│   ├── ml
	│   ├── optimisation.py
	│   ├── priors.py
	│   ├── processing
	│   ├── process_outputs.py
	│   ├── scale_inputs.py
	│   ├── setemu.py
	│   ├── testing.ipynb
	│   ├── test_ps.py
	│   ├── tests
	│   ├── train_gps.py
	│   ├── training_points.py
	│   └── utils
	├── LICENSE
	├── README.rst
	├── requirements.txt
	├── testing.txt
	└── test_semigp.ipynb
