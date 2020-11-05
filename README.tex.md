# GP Emulator For KiDS-450 Analysis 

We use the publicly available KiDS-450 data (see <a href="http://kids.strw.leidenuniv.nl/sciencedata.php">here</a>). 

In particular, we emulate three quantities
- $\sigma_{8}$ (<a href="https://github.com/Harry45/gp_emulator/tree/master/sigma_eight">here</a>)
- band powers (<a href="https://github.com/Harry45/gp_emulator/tree/master/bandpowers">here</a>)
- MOPED coefficients (<a href="https://github.com/Harry45/gp_emulator/tree/master/moped">here</a>)

### Computing $\sigma_{8}$

<p align="justify">Since our emulator is not a function of $\sigma_{8}$, we record the values of the latter as we generate our training set and build a GP emulator on top it. We also make this package publicly available and is found <a href="https://github.com/Harry45/gp_emulator/tree/master/sigma_eight">here</a>. In particular, the GP for $\sigma_{8}$ is a function of the following parameters:</p>

$$
\left[\Omega_{\textrm{cdm}}h^{2},\,\Omega_{\textrm{b}}h^{2},\,\textrm{ln}\left(10^{10}A_{s}\right),\,n_{s},\,h,\,\Sigma m_{\nu}\right]
$$

### Note 

<p align="justify">We do not provide all the GPs (MOPED and band powers) in the folder gps because the GP models require $\mathcal{O}(N_{\textrm{train}}^{2})$ memory storage. However, we also share the scripts to train the Gaussian Processes for MOPED and band powers respectively. We provide only one GP model, for computing $\sigma_{8}$.</p>