Gaussian Process
----------------
Here, we provide multiple code (although not all of them are integrated in the full pipeline) to learn a function between the inputs and the outputs. In the simple case, one can use a Gaussian Linear Model or a zero mean Gaussian Process. Note that throughout, we are using the Radial Basis Function (RBF) kernel. We can also opt for applying :math:`\text{log}_{10}` transformation to the output. For the inputs, the pre-whitening step is highly recommended. 

.. toctree::
   :maxdepth: 4

   algebra
   gaussianlinear
   kernel
   optimisation
   semigp
   transformation
   zerogp
