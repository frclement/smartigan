# smartigan
Code for the results in https://arxiv.org/abs/2604.21798
# Gaussians
submit.py contains sufficient code to copy the Gaussian experiments. The implementation does not aim to be optimal, we recommend simply adapting any existing implementation for efficient code.

The output files list seeds and kmeans values for the Gaussian experiments. Each block of 4 lines corresponds to a (n,k,seed) setting, with the dimension given in the file name. H or S indicates Smartigan or Hartigan and the first number is the seed. The first pair of lines in each block corresponds to clusters closer together with higher covariance for the Gaussians, the other two clusters further away with lower covariance.

The "Many" and "Manybest" files contain respectively the mean and best values for the tests in the files above.

# Lederman et al. experiments

We took exactly the implementation at https://github.com/Lederman-Group/Catastrophic_Failure_KMeans/tree/main.
The only modifications to the code necessary are all in _hartigan.py (for our current choice of the threshold function):
1) line 32 : Add a parameter n_iter
2) line 44 Modify distances[i] \*=scale_factor to distances[i] \*=scale_factor\*(1.5-0.5\*n_iter)
3) line 60: Add n_iters/max_iters as a last parameter for the function call _assign_label_hartigan_np()

# Mathematica files
We also provide Mathematica files used for the Fisher data sets and initial experiments. LLMs were used for parts of their generation.
