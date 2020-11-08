.. emuPK documentation master file, created by
   sphinx-quickstart on Thu Nov  5 21:09:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

emuPK
=====

*Emulator for 3D matter power spectrum*

``emuPK`` is an emulator for generating the 3D matter power spectrum which can be used in conjunction with a weak lensing likelihood code to derive constraints on cosmological parameters. It is built based on the following parameters and prior range:

.. list-table:: Definition of the parameter inputs to the emulator
	:header-rows: 1
	:widths: 1 2 1

	* - Parameters
	  - Description
	  - Prior

	* - :math:`\Omega_{\text{cdm}}h^{2}`
	  - CDM density 
	  - :math:`\mathcal{U}[0.01, 0.40]`

	* - :math:`\Omega_{\text{b}}h^{2}`
	  - Baryon density 
	  - :math:`\mathcal{U}[0.019, 0.026]`

	* - :math:`\text{ln}(10^{10}A_{s})`
	  - Scalar spectrum amplitude
	  - :math:`\mathcal{U}[1.7, 5.0]`

	* - :math:`n_{s}`
	  - Scalar spectral index 
	  - :math:`\mathcal{U}[0.7, 1.3]`

	* - :math:`h`
	  - Hubble parameter
	  - :math:`\mathcal{U}[0.64, 0.82]`

	* - :math:`\Sigma m_{\nu}`
	  - Neutrino mass (eV)
	  - :math:`\mathcal{U}[0.06, 1.0]`

	* - :math:`A_{\text{bary}}`
	  - Free amplitude baryon feedback parameter
	  - :math:`\mathcal{U}[0.0, 2.0]`

**Citation**

If you use this code in your research, please cite the following papers: `Paper 1 <https://arxiv.org/abs/2005.06551>`_ and Paper 2 (in preparation):

.. code-block:: tex

	@ARTICLE{2020MNRAS.497.2213M,
	       author = {{Mootoovaloo}, Arrykrishna and {Heavens}, Alan F. and
	         {Jaffe}, Andrew H. and {Leclercq}, Florent},
	        title = "{Parameter inference for weak lensing using Gaussian Processes and MOPED}",
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


Background
==========
In this section, we explain briefly how the emulator can be used in a weak lensing analysis and we also provide an example to illustrate the performance of the emulator. 

.. toctree::
   :maxdepth: 2
   :caption: Usage

   wl
   gettingStarted
   
API Documentation
=================
Here we provide a brief overview of the code structure. Please see `Github <https://github.com/Harry45/emuPK>`_ for full code structure. We do not cover each code in full detail here.

.. toctree::
   :maxdepth: 2
   :caption: Codes

   priors
   utilities


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
