#include <stdint.h>
#include "fof.h"
#include "groupies.h"
#include "python_interface.h"

__attribute__((visibility("default")))
int rockstar_analyze_fof_group(struct particle *particles, int64_t num_particles, int return_substructure) {
    struct fof fake_fof;
    fake_fof.num_p = num_particles;
    fake_fof.particles = particles;

    find_subs(&fake_fof);

    return 0;
}