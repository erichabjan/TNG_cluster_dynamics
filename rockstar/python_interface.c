#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

#include "fof.h"
#include "groupies.h"
#include "python_interface.h"
#include "halo.h"

extern struct halo *halos;
extern struct extra_halo_info *extra_info;
extern struct particle *p;
extern int64_t num_halos;
extern int64_t *particle_halos;

extern int    MIN_HALO_PARTICLES;
extern double FOF_FRACTION;

extern double PARTICLE_MASS;   /* Msun/h */
extern double FORCE_RES;       /* Mpc/h   */
extern double SCALE_NOW;       /* a = 1/(1+z) */
extern int    BOUND_PROPS;     /* 1 = use bound particles for props */

__attribute__((visibility("default")))
int rockstar_analyze_fof_group(struct particle *particles, int64_t num_particles,
                               int return_substructure, double dm_mass,
                               const char *subhalo_fname, const char *member_fname,
                               int min_halo_particle_size, double fof_fraction,
                               double dm_mass_in_Msun_over_h, double softening_in_Mpc_over_h,
                               double a_scale_factor)
{
    (void)return_substructure; // currently unused

    /* Save current global config */
    int    old_min_halo_particle_size = MIN_HALO_PARTICLES;
    double old_fof_fraction           = FOF_FRACTION;

    if (min_halo_particle_size > 0) MIN_HALO_PARTICLES = min_halo_particle_size;
    if (fof_fraction > 0.0)         FOF_FRACTION       = fof_fraction;

    struct fof fake_fof;
    fake_fof.num_p = num_particles;
    fake_fof.particles = particles;

    /* global p */
    p = particles;

    int64_t h_start = num_halos;

    /* Set physical scalings for unbinding/props */
    PARTICLE_MASS = dm_mass_in_Msun_over_h;
    FORCE_RES     = softening_in_Mpc_over_h;  /* comoving Mpc/h */
    SCALE_NOW     = a_scale_factor;
    BOUND_PROPS   = 1;

    /* Run ROCKSTAR */
    find_subs(&fake_fof);

    /* Restore globals */
    MIN_HALO_PARTICLES = old_min_halo_particle_size;
    FOF_FRACTION       = old_fof_fraction;

    /* Write subhalo catalog */
    FILE *f = fopen(subhalo_fname, "w");
    if (!f) {
        fprintf(stderr, "Failed to open subhalo output file.\n");
        return -1;
    }

    fprintf(f,
    "# id parent_id "
    "pos_0 pos_1 pos_2 pos_3 pos_4 pos_5 "
    "num_p mgrav_est mgrav_bound "
    "vrms vmax rvmax "
    "rs kin_to_pot Xoff\n");

    int64_t local_id = 0;

    for (int64_t i = h_start; i < num_halos; i++) {
        /* Only keep particles assigned to subhalos */
        if (extra_info[i].sub_of < 0) continue;

        struct halo *h = &halos[i];

        int64_t parent = extra_info[i].sub_of;
        int64_t parent_local = (parent >= h_start && parent < num_halos) ? (parent - h_start) : -1;

        double mgrav_est   = (double)h->num_p * dm_mass;
        double mgrav_bound = (double)h->mgrav;

        fprintf(f,
        "%" PRId64 " %" PRId64 " "
        "%.6f %.6f %.6f %.6f %.6f %.6f "
        "%" PRId64 " %.6e %.6e "
        "%.6e %.6e %.6e "
        "%.6e %.6e %.6e\n",
        local_id, parent_local,
        h->pos[0], h->pos[1], h->pos[2],
        h->pos[3], h->pos[4], h->pos[5],
        h->num_p, mgrav_est, mgrav_bound,
        h->vrms, h->vmax, h->rvmax,
        h->rs, h->kin_to_pot, h->Xoff
        );

        local_id++;
    }
    fclose(f);

    /* Subhalo Membership file */
    FILE *f2 = fopen(member_fname, "w");
    if (!f2) {
        fprintf(stderr, "Failed to open membership file.\n");
        return -1;
    }

    for (int64_t j = 0; j < num_particles; j++) {
        int64_t hid = particle_halos[j];
        if (hid < h_start || hid >= num_halos) continue;

        /* Remove inner fuzz */
        if (extra_info[hid].sub_of < 0) continue;

        int64_t local_hid = hid - h_start;

        fprintf(f2, "%" PRId64 " %" PRId64 "\n",
                local_hid, (int64_t)particles[j].id);
    }
    fclose(f2);

    return 0;
}