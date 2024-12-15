#!/bin/bash
usage(){ cat << EOU

~/o/qudarap/tests/QSimTest_ALL.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))


tests=$(sed 's/#.*//' << EOT

rng_sequence

boundary_lookup_all
boundary_lookup_water
boundary_lookup_ls


wavelength_scintillation

#wavelength_cerenkov

scint_generate
cerenkov_generate

fill_state_0
fill_state_1

#rayleigh_scatter_align

propagate_to_boundary

hemisphere_s_polarized
hemisphere_p_polarized
hemisphere_x_polarized

propagate_at_boundary_s_polarized
propagate_at_boundary_p_polarized
propagate_at_boundary_x_polarized

#propagate_at_boundary
#propagate_at_boundary_normal_incidence

random_direction_marsaglia
lambertian_direction
reflect_diffuse
reflect_specular
propagate_at_surface
randgaussq_shoot

fake_propagate
gentorch

smear_normal_sigma_alpha

EOT
)

source ALL_TEST_runner.sh QSimTest.sh 


