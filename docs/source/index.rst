.. emuPK documentation master file, created by
   sphinx-quickstart on Thu Nov  5 21:09:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

emuPK
=====

*Emulator for 3D matter power spectrum*

``emuPK`` is an emulator for generating the 3D matter power spectrum which can be used in conjunction with a weak lensing likelihood code to derive constraints on cosmological parameters. It is built based on the following parameters and prior range:

.. list-table:: Title
   :widths: 25 25 50
   :header-rows: 1

   * - Heading row 1, column 1
     - Heading row 1, column 2
     - Heading row 1, column 3
   * - Row 1, column 1
     -
     - Row 1, column 3
   * - Row 2, column 1
     - Row 2, column 2
     - Row 2, column 3

     
.. ==========   							===========   								===========   
.. Parameters   							Description   								Prior Range
.. ==========  							===========   								===========
.. :math: \Omega_{\textrm{cdm}}h^{2}  		CDM density   								Prior Range
.. :math: \Omega_{\textrm{b}h^{2}    		Baryon density   							Prior Range
.. :math: \textrm{ln}10^{10}A_{s}   		Scalar spectrum amplitude   				Prior Range
.. :math: n_{s}  							Scalar spectral index   					Prior Range
.. :math: h   								Hubble parameter   							Prior Range
.. :math: \Sigma m_{\nu}   				Neutrino mass (eV)   						Prior Range
.. :math: A_{\textrm{bary}}   				Free amplitude baryon feedback parameter   	Prior Range
.. ==========  							===========   								===========


Citation
--------

If you use this code in your research, please attribute `this paper <https://arxiv.org/abs/2005.06551>`_:

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


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   utilities


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
