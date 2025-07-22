#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include "fof.h"
#include "groupies.h"
#include "python_interface.h"
#include <inttypes.h>
#include "halo.h"

extern struct halo *halos;
extern struct particle *p;
extern int64_t    num_halos;
extern struct particle *copies;
extern int64_t        *particle_halos;

__attribute__((visibility("default")))
int rockstar_analyze_fof_group(struct particle *particles, int64_t num_particles, int return_substructure, double dm_mass, const char *subhalo_fname, const char *member_fname) {
    printf("OpenMP: using %d threads\n", omp_get_max_threads());

    int64_t *orig_ids = malloc(num_particles * sizeof(int64_t));
    if (!orig_ids) {
        fprintf(stderr, "Failed to allocate orig_ids.\n");
        return -1;
    }
    for (int64_t i = 0; i < num_particles; i++) {
        orig_ids[i] = particles[i].id;
    }

    struct fof fake_fof;
    fake_fof.num_p = num_particles;
    fake_fof.particles = particles;
    p = particles;
    int64_t h_start = num_halos;

    find_subs(&fake_fof);

    FILE *f = fopen(subhalo_fname, "w");
    if (!f) {
        fprintf(stderr, "Failed to open output file.\n");
        free(orig_ids);
        return -1;
    }

    fprintf(f,
        "# id    pos_0       pos_1       pos_2       pos_3       pos_4       pos_5       "
        "bulkvel_0      bulkvel_1      bulkvel_2      corevel_0      corevel_1      corevel_2      "
        "mgrav    p_start    num_p\n");

    for (int64_t i = h_start; i < num_halos; i++) {
        struct halo *h = &halos[i];
        double grav_mass = h->num_p * dm_mass;
        int64_t hid = i - h_start;
        fprintf(f, 
            "%" PRId64    " "    //  id
            "%.6f "  "%.6f "  "%.6f "  // pos[0..2]
            "%.6f "  "%.6f "  "%.6f "  // pos[3..5]
            "%.6f "  "%.6f "  "%.6f "  // bulkvel[0..2]
            "%.6f "  "%.6f "  "%.6f "  // corevel[0..2]
            "%.6e "           // mgrav
            "%" PRId64 " "
            "%" PRId64 "\n",           // num_p
            hid,
            h->pos[0], h->pos[1], h->pos[2],
            h->pos[3], h->pos[4], h->pos[5],
            h->bulkvel[0], h->bulkvel[1], h->bulkvel[2],
            h->corevel[0], h->corevel[1], h->corevel[2],
            grav_mass,
            h->p_start,
            h->num_p);
    }
    fclose(f);

    FILE *f2 = fopen(member_fname, "w");
    if (!f2) {
        fprintf(stderr, "Failed to open membership file.\n");
        free(orig_ids);
        return -1;
    }

    for (int64_t i = h_start; i < num_halos; i++) {
        int64_t hid = i - h_start;
        for (int64_t j = 0; j < num_particles; j++) {
            if (particle_halos[j] == i) {
                int64_t real_id = orig_ids[j];
                fprintf(f2, "%" PRId64 " %" PRId64 "\n", hid, real_id);
            }
        }
    }
    fclose(f2);
    free(orig_ids);
    return 0;
}