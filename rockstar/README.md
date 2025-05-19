The C script in this folder runs the [ROCKSTAR phase-space temporal halo finder](https://iopscience.iop.org/article/10.1088/0004-637X/762/2/109). The script takes in simulation particles from TNGIllustrius as a single HDF5 file. The file contains particles identified by the Friends-of-Friends algorithm `subfind`. Using particles from host halos, we aim to find the subhalos/substructure within the TNGIllustrius halos/galaxy clusters. 

The `rockstar_interface.c` script must be placed inside of the [ROCSKTAR directory](https://bitbucket.org/gfcstanford/rockstar/src/main/) and compiled using the following command: 

<pre><code>``` gcc -std=c99 -o rockstar_interface rockstar_interface.c -L. -lrockstar -lm -lhdf5 ``` </code></pre>

This will produce an executable inside of the ROCKSTAR directory that can then be ran using the `run_rockstar.slurm` file. The results of the ROCKSTAR run will be stored in a separated directory called `rockstar_output/rockstar_results_<slurm ID>` that will contain a `halos.bin` file. This binary file contains information such as subhalo mass, subhalo velocity and the cutouts of the subhalos in the coordinates of the simulation. 