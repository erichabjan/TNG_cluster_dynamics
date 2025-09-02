#ifndef PYTHON_INTERFACE_H
#define PYTHON_INTERFACE_H

#include <stdint.h>
#include "fof.h"

int rockstar_analyze_fof_group(struct particle *particles, int64_t num_particles, 
                               int return_substructure, double dm_mass, 
                               const char *subhalo_fname, const char *member_fname, 
                               int min_halo_particle_size, double fof_fraction);

#endif