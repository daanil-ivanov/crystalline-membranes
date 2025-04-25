# crystalline-membranes

This repository contains 2 of my works.

# no_noise dir

The first one is related to calculating the critical index η in crystalline membranes without noise. The final result and some illustrations are posted in the no_noise directory. Since this work has already been published, the working codes are posted without changes. C with OpenMP were used. This task is poorly parallelized on GPUs.

This plot depicts main result of simulations: numeric results of Monte Carlo simulations for bending rigidity exponent (eta) with plotted theoretical approximations.

![1](https://github.com/user-attachments/assets/fd7e37c4-f90f-419b-b25a-65d78c9ffd87)


# noise dir

The second one is related to calculating critical indices in membranes with Gaussian noise. This task is ideal for CUDA C/C++, since it is necessary to average the obtained results between different noise realisations. Please note that this work is not yet finished: in particular, data is still collecting in order to plot a graph of the dependence of this critical index on noise in order to find the phase transition point. This work is carried out for numerical verification of the theoretical provisions described in Saykin's article: with infinitesimal noise amplitude, the critical index practically does not change, but after passing the critical point, it sharply drops almost twofold. Since the work is not yet published, some parts of the code have been modified, which may make it difficult to reproduce the results I have posted.

As you can see, the very first numeric results give a good confirmation of theory for different sizes of lattice. Dotted lines depict the initial distibution of lattice, dashed line means the numeric result of this simulation, blue line and green line are stand for theoretical results for no noise and huge noise amplitude. This is just one of several critical indices that are calculating in this work.

![scatter_N=120_dc=1_sigma=1e-07](https://github.com/user-attachments/assets/83ff58d3-bf3b-4ff7-9c60-28b4957571e7)

![scatter_N=120_dc=1_sigma=3 5](https://github.com/user-attachments/assets/bfa00c09-1206-43ef-bd73-70a8c4289236)

![scatter_N=170_dc=1_sigma=3 0](https://github.com/user-attachments/assets/35034237-6cca-4d63-8f22-fc968fb7b16d)
