Emulator Build-Up
-----------------

Building the emulator requires different stages:

- generating the training points
- training the Gaussian Processes

The inputs are generated using Latin Hypercube sampling and these are scaled using the :code:`priors` module below. One can specify a prior using the dictionary format as follows:

.. code:: python

	{'distribution': 'uniform', 'specs': [0.06, 0.34]}

where :code:`specs` are the specificifications for the distribution. In the example above, :math:`0.06` is the minimum and the maximum is :math:`0.06+0.34=0.40`. Note that we are using :code:`scipy.stats` and hence the above convention. The GP models are trained in parallel (number of cores available on your computer) and are stored.

.. toctree::
   :maxdepth: 4

   predictions
   priors
   training
   trainingpoints
