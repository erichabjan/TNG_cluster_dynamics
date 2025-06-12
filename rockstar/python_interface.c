#include <stdint.h>
#include <stdio.h>
#include "fof.h"
#include "groupies.h"
#include "python_interface.h"
#include "halo.h"  // for access to global halos[] and struct halo

extern struct halo *halos;       // global halo array
extern int64_t num_halos;        // total number of halos

__attribute__((visibility("default")))
int rockstar_analyze_fof_group(struct particle *particles, int64_t num_particles, int return_substructure) {
    struct fof fake_fof;
    fake_fof.num_p = num_particles;
    fake_fof.particles = particles;

    int64_t h_start = num_halos;  // track where new halos begin

    find_subs(&fake_fof);

    FILE *f = fopen("rockstar_subhalos_0.list", "w");
    if (!f) {
        fprintf(stderr, "Failed to open output file.\n");
        return -1;
    }

    fprintf(f, "# id    x       y       z       vx      vy      vz      mass    num_p\n");

    for (int64_t i = h_start; i < num_halos; i++) {
        struct halo *h = &halos[i];
        fprintf(f, "%ld %.6f %.6f %.6f %.6f %.6f %.6f %.6e %ld\n",
                i - h_start,
                h->pos[0], h->pos[1], h->pos[2],
                h->pos[3], h->pos[4], h->pos[5],
                h->m,
                h->num_p);
    }

    fclose(f);

    return 0;
}