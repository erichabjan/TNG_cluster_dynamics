The code in this folder runs the [ROCKSTAR phase-space temporal halo finder](https://iopscience.iop.org/article/10.1088/0004-637X/762/2/109) for a single Friends-of-Friends halo identified by algorithm `subfind`. The script takes in simulation particles from TNGIllustrius as a single HDF5 file. Using particles from host halos, we aim to find the subhalos/substructure within the TNGIllustrius halos/galaxy clusters. 

To set up this code, download the [ROCKSTAR repository](https://bitbucket.org/gfcstanford/rockstar/src/main/) and add the following lines to the `Makefile` insde the rockstar repository: 

<pre><code> librockstar.a: $(CFILES:.c=.o)
	ar rcs librockstar.a $(CFILES:.c=.o) </code></pre>

This allows for a static library called `librockstar.a` to be created inside of the rockstar folder. Then, run the following command to compile the static library: 

<pre><code> make librockstar.a \
  CFLAGS="-I$HOME/.local/include -I$HOME/.local/include/tirpc -D_DEFAULT_SOURCE" \
  LDFLAGS="-lm -Wl,--whole-archive $HOME/.local/lib/libtirpc.a -Wl,--no-whole-archive"
 </code></pre>

The `CFLAGS` and `LDFLAGS` specifications may be unnecessary, however I needed to point the make file to my version of `tirpc`. A Python wrapper will be written to utilize this static library.

The results of the ROCKSTAR run will be stored in a separated directory called `rockstar_output/rockstar_results_<slurm ID>` that will contain a `halos.bin` file. This binary file contains information such as subhalo mass, subhalo velocity and the cutouts of the subhalos in the coordinates of the simulation. 