Weak Lensing Power Spectrum
===========================

One can use the 3D matter power spectrum to calculate the EE, GI and II power spectra for weak lensing. For this particular application, :math:`P_{\delta}(k,z)\rightarrow P_{\text{emu}}(k,z)`. The EE, GI and II power spectra are given respectively by:

.. math::

	\mathcal{C}_{\ell,\,ij}^{\text{EE}}=\int_{0}^{\chi_{H}}\text{d}\chi\,\dfrac{w_{i}(\chi)w_{j}(\chi)}{\chi^{2}}\,P_{\delta}(k,\chi)

.. math::

	\mathcal{C}_{\ell,\,ij}^{\text{GI}}=\int_{0}^{\chi_{H}}\text{d}\chi\,\dfrac{w_{i}(\chi)n_{j}(\chi)+w_{j}(\chi)n_{i}(\chi)}{\chi^{2}}P_{\delta}(k,\chi)\,F(\chi)

.. math::

	\mathcal{C}_{\ell,\,ij}^{\text{II}}=\int_{0}^{\chi_{H}}\text{d}\chi\,\dfrac{w_{i}(\chi)w_{j}(\chi)}{\chi^{2}}\,P_{\delta}(k,\chi)\,F^{2}(\chi)

where :math:`\chi` is the comoving radial distance and :math:`n(\chi)` is the (tomographic) redshift distribution. Please see paper for full definition of :math:`F(\chi)` and :math:`w(\chi)`.