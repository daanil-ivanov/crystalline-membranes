# crystalline-membranes

This repository contains 2 of my works.

The first one is related to calculating the critical index η in crystalline membranes without noise. The final result and some illustrations are posted in the no_noise directory. Since this work has already been published, the working codes are posted without changes. C with OpenMP were used. This task is poorly parallelized on GPUs.

The second one is related to calculating critical indices in membranes with Gaussian noise. This task is ideal for CUDA C/C++, since it is necessary to average the obtained results between different noise realisations. Please note that this work is not yet finished: in particular, data is collected in order to plot a graph of the dependence of this critical index on noise in order to find the phase transition point. This work is carried out for numerical verification of the theoretical provisions described in Saykin's article: with infinitely small noise, the critical index practically does not change, but after passing the critical point, it sharply drops almost twofold. Since the work is not yet published, some parts of the code have been modified, which may make it difficult to reproduce the results I have posted.
