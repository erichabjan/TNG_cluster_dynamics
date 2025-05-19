#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <stdint.h>
#include <math.h>

// Include ROCKSTAR headers
#include "rockstar.h"
#include "particle.h"
#include "config_vars.h"
#include "check_syscalls.h"

// Function prototypes
void convert_hdf5_to_particles(const char* filename);

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <hdf5_file>\n", argv[0]);
        return 1;
    }

    const char* hdf5_file = argv[1];
    
    // Set configuration parameters
    PERIODIC = 0;                 // No periodic boundaries for single group
    GADGET_MASS_CONVERSION = 1.0; // No mass conversion needed
    UNBOUND_THRESHOLD = 0.5;      // Require 50% of particles to be bound
    BOUND_PROPS = 1;              // Calculate properties using bound particles
    FOF_FRACTION = 0.7;           // FOF fraction for refinement levels
    MIN_HALO_PARTICLES = 10;      // Minimum particles per halo
    
    // Load the particles from HDF5 file
    printf("Loading particles from %s\n", hdf5_file);
    convert_hdf5_to_particles(hdf5_file);
    
    if (!num_p) {
        printf("Error: No particles loaded\n");
        return 1;
    }
    
    printf("Loaded %"PRId64" particles\n", num_p);
    
    // Configure for single FOF group
    all_fofs = check_realloc(NULL, sizeof(struct fof), "allocation for FOF group");
    all_fofs[0].num_p = num_p;
    all_fofs[0].particles = p;
    num_all_fofs = 1;
    
    // Step 2-6: Process the particles
    printf("Finding halos...\n");
    rockstar(NULL, 0);  // With manual_subs=0, rockstar will automatically find substructure
    
    printf("Found %"PRId64" halos\n", num_halos);
    
    // Output halo properties
    for (int64_t i=0; i<num_halos; i++) {
        struct halo *h = &halos[i];
        printf("Halo %"PRId64":\n", i);
        printf("  Position:  [%.3f, %.3f, %.3f]\n", h->pos[0], h->pos[1], h->pos[2]);
        printf("  Velocity:  [%.3f, %.3f, %.3f]\n", h->pos[3], h->pos[4], h->pos[5]);
        printf("  Particles: %"PRId64"\n", h->num_p);
        printf("  Mass:      %.3e\n", h->m);
        printf("  Radius:    %.3f\n", h->r);
        printf("  Vmax:      %.3f\n", h->vmax);
        printf("\n");
    }
    
    // Output a simple binary format with halo information
    FILE *f = fopen("halos.bin", "wb");
    if (f) {
        fwrite(&num_halos, sizeof(int64_t), 1, f);
        fwrite(halos, sizeof(struct halo), num_halos, f);
        fclose(f);
        printf("Saved halo catalog to halos.bin\n");
    }
    
    // Clean up
    rockstar_cleanup();
    return 0;
}

void convert_hdf5_to_particles(const char* filename) {
    hid_t file_id, group_id, pos_id, vel_id, space_id;
    herr_t status;
    
    // Open the HDF5 file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Error: Could not open HDF5 file %s\n", filename);
        exit(1);
    }
    
    // Open the PartType1 group
    group_id = H5Gopen(file_id, "PartType1", H5P_DEFAULT);
    if (group_id < 0) {
        printf("Error: Could not open PartType1 group\n");
        H5Fclose(file_id);
        exit(1);
    }
    
    // Open the coordinates and velocities datasets
    pos_id = H5Dopen(group_id, "Coordinates", H5P_DEFAULT);
    vel_id = H5Dopen(group_id, "Velocities", H5P_DEFAULT);
    
    if (pos_id < 0 || vel_id < 0) {
        printf("Error: Could not open Coordinates or Velocities datasets\n");
        if (pos_id >= 0) H5Dclose(pos_id);
        if (vel_id >= 0) H5Dclose(vel_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        exit(1);
    }
    
    // Get the dimensions
    space_id = H5Dget_space(pos_id);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    
    // Allocate memory for particles
    num_p = dims[0];
    p = check_realloc(NULL, num_p * sizeof(struct particle), "particle allocation");
    
    // Read position data
    float (*pos_data)[3] = malloc(num_p * sizeof(float[3]));
    status = H5Dread(pos_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, pos_data);
    if (status < 0) {
        printf("Error: Could not read position data\n");
        free(pos_data);
        H5Dclose(pos_id);
        H5Dclose(vel_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        exit(1);
    }
    
    // Read velocity data
    float (*vel_data)[3] = malloc(num_p * sizeof(float[3]));
    status = H5Dread(vel_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vel_data);
    if (status < 0) {
        printf("Error: Could not read velocity data\n");
        free(pos_data);
        free(vel_data);
        H5Dclose(pos_id);
        H5Dclose(vel_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        exit(1);
    }
    
    // Convert to particle structure
    for (int64_t i = 0; i < num_p; i++) {
        p[i].id = i;  // Use index as ID
        
        for (int j = 0; j < 3; j++) {
            p[i].pos[j] = pos_data[i][j];     // Position
            p[i].pos[j+3] = vel_data[i][j];   // Velocity
        }
    }
    
    // Clean up
    free(pos_data);
    free(vel_data);
    H5Dclose(pos_id);
    H5Dclose(vel_id);
    H5Sclose(space_id);
    H5Gclose(group_id);
    H5Fclose(file_id);
    
    // Set some approximate simulation parameters based on the data
    BOX_SIZE = 0;
    for (int i = 0; i < 3; i++) {
        float min_pos = 1e10, max_pos = -1e10;
        for (int64_t j = 0; j < num_p; j++) {
            if (p[j].pos[i] < min_pos) min_pos = p[j].pos[i];
            if (p[j].pos[i] > max_pos) max_pos = p[j].pos[i];
        }
        float size = max_pos - min_pos;
        if (size > BOX_SIZE) BOX_SIZE = size;
    }
    
    // Add some padding
    BOX_SIZE *= 1.1;
    
    // Calculate approximate particle mass
    PARTICLE_MASS = 1.0;  // Default to unity if no mass info
    
    // Calculate the average particle spacing
    AVG_PARTICLE_SPACING = pow(BOX_SIZE * BOX_SIZE * BOX_SIZE / num_p, 1.0/3.0);
    
    // Set force resolution to 1/30 of average particle spacing
    FORCE_RES = AVG_PARTICLE_SPACING / 30.0;
    
    // Set cosmological parameters (not important for single group analysis)
    SCALE_NOW = 1.0;
    Om = 0.3;
    Ol = 0.7;
    h0 = 0.7;
    
    printf("Box size: %f\n", BOX_SIZE);
    printf("Avg particle spacing: %f\n", AVG_PARTICLE_SPACING);
    printf("Force resolution: %f\n", FORCE_RES);
}