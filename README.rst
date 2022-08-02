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

If you use this code in your research, please cite these papers `2005.06551
<https://arxiv.org/abs/2005.06551>`_ and `2105.02256
<https://arxiv.org/abs/2105.02256>`_:

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

.. code-block:: tex

	@ARTICLE{2022A&C....3800508M,
		author    = {{Mootoovaloo}, A. and {Jaffe}, A.~H. and {Heavens}, A.~F. and {Leclercq}, F.},
			title = {Kernel-based emulator for the 3D matter power spectrum from CLASS},
		journal   = {Astronomy and Computing},
		keywords  = {Kernel, Gaussian Process, Emulation, Large scale structures, Astrophysics - Cosmology and Nongalactic Astrophysics},
			year  = 2022,
			month = jan,
		volume    = {38},
			eid   = {100508},
			pages = {100508},
			doi   = {10.1016/j.ascom.2021.100508},
	archivePrefix = {arXiv},
		eprint 	  = {2105.02256},
	primaryClass  = {astro-ph.CO},
		adsurl 	  = {https://ui.adsabs.harvard.edu/abs/2022A&C....3800508M},
		adsnote   = {Provided by the SAO/NASA Astrophysics Data System}
	}

Directory Structure
-------------------

.. code:: bash

	.
	├── docs
	│   ├── build
	│   ├── logs
	│   ├── make.bat
	│   ├── Makefile
	│   └── source
	├── emulator
	│   ├── cosmology
	│   ├── lhs
	│   ├── lhs.R
	│   ├── ml
	│   ├── playground.ipynb
	│   ├── plots
	│   ├── predictions
	│   ├── predictions.py
	│   ├── priors.py
	│   ├── semigps
	│   ├── settings.py
	│   ├── trainingpoints.py
	│   ├── training.py
	│   ├── trainingset
	│   └── utils
	├── LICENSE
	├── README.rst
	└── requirements.txt
