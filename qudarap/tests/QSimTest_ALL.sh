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

count=0
pass=0
fail=0

flog=""
t0="$(printf "\n$(date)"$'\n')"
echo "$t0"
echo 

for t in $tests ; do 

    tt="TEST=$t $(realpath QSimTest.sh)"

    l0="$(printf " === %0.3d === [ $tt "$'\n' "$count")"
    echo "$l0"
    eval "$tt" > /dev/null 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then 
        msg="***FAIL***"
        fail=$(( $fail + 1 ))
    else
        msg=PASS
        pass=$(( $pass + 1 ))
    fi   
    l1="$(printf " === %0.3d === ] %s "$'\n' "$count" "$msg")"
    echo "$l1"
    echo
 
    if [ $rc -ne 0 ]; then
       flog+="$l0"$'\n'
       flog+="$l1"$'\n'
    fi 

   #[ $rc -ne 0 ] && echo non-zero RC && break
    count=$(( $count + 1 ))
done 

t1="$(printf "\n$(date)"$'\n')"

echo "$t0"
echo "$t1"
echo

printf " TOTAL : %d \n" $count
printf " PASS  : %d \n" $pass
printf " FAIL  : %d \n" $fail

echo "$flog"


