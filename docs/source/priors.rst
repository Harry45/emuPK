Priors
======

We can setup any prior (uniform, normal, beta and others) for the inputs to the emulator. This can be achieved by using the definitions of each probability distribution in the ``scipy.stats`` `documentation <https://docs.scipy.org/doc/scipy/reference/stats.html>`_. For example, for a specifit parameter, one just needs to specify the following:

.. code:: python

	p1 = {'distribution': 'uniform', 'parameter': 'omega_cdm', 'specs': [0.01, 0.39]}

that is, we define

- the distribution type (for example, uniform),

- the parameter name (a string) and 

- the arguments to ``scipy.stats.uniform``.

Functions
---------

.. automodule:: priors
   :members:
   :undoc-members:
   :show-inheritance:
