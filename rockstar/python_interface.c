#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include "fof.h"
#include "groupies.h"
#include "python_interface.h"
#include <inttypes.h>
#include "halo.h"
#include <math.h>

extern struct halo *halos;
extern struct particle *p;
extern int64_t    num_halos;
extern struct particle *copies;
extern int64_t        *particle_halos;

extern int    MIN_HALO_PARTICLES;
extern double FOF_FRACTION; 

__attribute__((visibility("default")))
int rockstar_analyze_fof_group(struct particle *particles, int64_t num_particles, 
                               int return_substructure, double dm_mass, 
                               const char *subhalo_fname, const char *member_fname,
                               int min_halo_particle_size, double fof_fraction) {
    
    /* Save current global config so we can restore after */
    int    old_min_halo_particle_size = MIN_HALO_PARTICLES;
    double old_fof_fraction         = FOF_FRACTION;

    if (min_halo_particle_size > 0) {
        MIN_HALO_PARTICLES = min_halo_particle_size;
    }
    if (fof_fraction > 0.0) {
        FOF_FRACTION = fof_fraction;
    }

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

    /* Run ROCKSTAR*/
    find_subs(&fake_fof);

    /* Restore global variables */
    MIN_HALO_PARTICLES = old_min_halo_particle_size;
    FOF_FRACTION         = old_fof_fraction;

    /* Compute vrms for subhalos */
    int64_t n_new = num_halos - h_start;
    if (n_new > 0) {
        double *sum_sq = calloc((size_t)n_new, sizeof(double));
        int64_t *counts = calloc((size_t)n_new, sizeof(int64_t));
        if (!sum_sq || !counts) {
            fprintf(stderr, "Failed to allocate vrms accumulators.\n");
            free(orig_ids);
            free(sum_sq); free(counts);
            return -1;
        }

        /* Single pass over all particles */
        for (int64_t j = 0; j < num_particles; j++) {
            int64_t hid = particle_halos[j];       /* global halo index */
            if (hid < h_start || hid >= num_halos) {
                continue; /* not one of the new halos we just built */
            }
            int64_t local = hid - h_start;

            /* particle velocity components (pos[3..5]) */
            double vx = particles[j].pos[3];
            double vy = particles[j].pos[4];
            double vz = particles[j].pos[5];

            /* subtract halo bulk velocity */
            double dvx = vx - halos[hid].bulkvel[0];
            double dvy = vy - halos[hid].bulkvel[1];
            double dvz = vz - halos[hid].bulkvel[2];

            /* accumulate squared speed residual */
            sum_sq[local] += dvx*dvx + dvy*dvy + dvz*dvz;
            counts[local] += 1;
        }

        /* finalize vrms per halo */
        for (int64_t i = h_start; i < num_halos; i++) {
            int64_t local = i - h_start;
            if (counts[local] > 0) {
                halos[i].vrms = sqrt(sum_sq[local] / (double)counts[local]);
            } else {
                halos[i].vrms = 0.0; /* defensive */
            }
        }

        free(sum_sq);
        free(counts);
    }

    /* Make a subhalo file*/
    FILE *f = fopen(subhalo_fname, "w");
    if (!f) {
        fprintf(stderr, "Failed to open output file.\n");
        free(orig_ids);
        return -1;
    }

    /* Subhalo file columns */
    fprintf(f,
        "# id    pos_0       pos_1       pos_2       pos_3       pos_4       pos_5       "
        "mgrav      vrms    p_start    num_p\n");

    for (int64_t i = h_start; i < num_halos; i++) {
        struct halo *h = &halos[i];
        double grav_mass = h->num_p * dm_mass;
        int64_t hid = i - h_start;
        fprintf(f, 
            "%" PRId64    " "    //  id
            "%.6f "  "%.6f "  "%.6f "  // pos[0..2]
            "%.6f "  "%.6f "  "%.6f "  // pos[3..5]
            "%.6e "           // mgrav
            "%.6e "           // vrms
            "%" PRId64 " "
            "%" PRId64 "\n",           // num_p
            hid,
            h->pos[0], h->pos[1], h->pos[2],
            h->pos[3], h->pos[4], h->pos[5],
            grav_mass,
            h->vrms,
            h->p_start,
            h->num_p);
    }
    fclose(f);

    /* Make membership file*/
    FILE *f2 = fopen(member_fname, "w");
    if (!f2) {
        fprintf(stderr, "Failed to open membership file.\n");
        free(orig_ids);
        return -1;
    }

    /* Add entries to membership file*/
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