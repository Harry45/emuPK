# GP Emulator For KiDS-450 Analysis 

We use the publicly available KiDS-450 data (see <a href="http://kids.strw.leidenuniv.nl/sciencedata.php">here</a>). 

In particular, we emulate three quantities
- <img src="/tex/69bd68f5246fed6ce37aca9dff83028c.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94565279999999pt height=14.15524440000002pt/> (<a href="https://github.com/Harry45/gp_emulator/tree/master/sigma_eight">here</a>)
- band powers (<a href="https://github.com/Harry45/gp_emulator/tree/master/bandpowers">here</a>)
- MOPED coefficients (<a href="https://github.com/Harry45/gp_emulator/tree/master/moped">here</a>)

### Computing <img src="/tex/69bd68f5246fed6ce37aca9dff83028c.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94565279999999pt height=14.15524440000002pt/>

<p align="justify">Since our emulator is not a function of <img src="/tex/69bd68f5246fed6ce37aca9dff83028c.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94565279999999pt height=14.15524440000002pt/>, we record the values of the latter as we generate our training set and build a GP emulator on top it. We also make this package publicly available and is found <a href="https://github.com/Harry45/gp_emulator/tree/master/sigma_eight">here</a>. In particular, the GP for <img src="/tex/69bd68f5246fed6ce37aca9dff83028c.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94565279999999pt height=14.15524440000002pt/> is a function of the following parameters:</p>

<p align="center"><img src="/tex/cc22b0f214d24ba3260d4ed3eb5de2a0.svg?invert_in_darkmode&sanitize=true" align=middle width=298.98615615pt height=19.9563243pt/></p>

### Note 

<p align="justify">We do not provide all the GPs (MOPED and band powers) in the folder gps because the GP models require <img src="/tex/6b6c021e426987f0a7fb2eabcf24461f.svg?invert_in_darkmode&sanitize=true" align=middle width=68.16691199999998pt height=26.76175259999998pt/> memory storage. However, we also share the scripts to train the Gaussian Processes for MOPED and band powers respectively. We provide only one GP model, for computing <img src="/tex/69bd68f5246fed6ce37aca9dff83028c.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94565279999999pt height=14.15524440000002pt/>.</p>