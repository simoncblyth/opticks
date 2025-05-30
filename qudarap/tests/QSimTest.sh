#!/bin/bash
usage(){ cat << EOU
QSimTest.sh
=============

::

    ~/o/qudarap/tests/QSimTest.sh 
        runs the executable and invoke the python script  

    PIDX=0 ~/o/qudarap/tests/QSimTest.sh
    PIDX=2 ~/o/qudarap/tests/QSimTest.sh
        assuming QUDARap was compiled with DEBUG_PIDX this
        provides debug output for the provided photon id 

    TEST=fill_state_cf ~/o/qudarap/tests/QSimTest.sh ana
        just invoke the analysis script for the named TEST 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=QSimTest 

bin=$name



source $HOME/.opticks/GEOM/GEOM.sh
msg="=== $BASH_SOURCE :"


#export QSim__photon_launch_mutate_DEBUG_NUM_PHOTON=1


#test=rng_sequence

#test=boundary_lookup_all
#test=boundary_lookup_water
#test=boundary_lookup_ls

#test=wavelength_scintillation
#test=wavelength_cerenkov         ### non-active moved to QSim_dbg.cu 

#test=scint_generate
#test=cerenkov_generate

#test=fill_state_0
#test=fill_state_1

#test=rayleigh_scatter_align

#test=propagate_to_boundary

#test=hemisphere_s_polarized
#test=hemisphere_p_polarized
#test=hemisphere_x_polarized

#test=propagate_at_boundary_s_polarized
#test=propagate_at_boundary_p_polarized
#test=propagate_at_boundary_x_polarized

#test=propagate_at_boundary
#test=propagate_at_boundary_normal_incidence

#test=random_direction_marsaglia
#test=lambertian_direction
#test=reflect_diffuse
#test=reflect_specular
#test=propagate_at_surface
#test=randgaussq_shoot

test=fake_propagate
#test=gentorch

#test=smear_normal_sigma_alpha

export TEST=${TEST:-$test}
export BASE=/tmp/QSimTest         ## BASE NEEDED BY QSimTest::photon_launch_mutate 
                                  ## AS U::Resolve DOES NOT YET SUPPORT "$FOLD/.." 
                                  ## TODO:bring it over, done it elsewhere?
       
export FOLD=/tmp/QSimTest/$TEST   ## CAUTION clean subcommand deletes this directory and contents 
mkdir -p $FOLD



defarg=run_ana
if [ "$(uname)" == "Darwin" ]; then
   defarg="run_ana"
fi 

if [ -n "$BP" ]; then 
   defarg="dbg"
fi 

if [ "$TEST" == "rng_sequence" ]; then
   defarg="run_ana_clean"  # delete FOLD after test as writes almost 1GB 
fi 


arg=${1:-$defarg}



M1=1000000
K2=100000

#num=8
num=$K2
#num=$M1

nrm=0,0,1
#nrm=0,0,-1

case $TEST in
    rng_sequence) num=$M1 ;; 
    random_direction_marsaglia) num=$M1 ;; 
    lambertian_direction) num=$M1 ;; 
    randgaussq_shoot) num=$M1 ;; 
     wavelength*) num=$M1 ;; 
     randgaussq*) num=$M1 ;; 
  scint_generate) num=$M1 ;;
  cerenkov_generate) num=$M1 ;;
   hemisphere_s_polarized|propagate_at_boundary_s_polarized) num=$M1 ;; 
   hemisphere_p_polarized|propagate_at_boundary_p_polarized) num=$M1 ;; 
   hemisphere_x_polarized|propagate_at_boundary_x_polarized) num=$M1 ;; 
   propagate_at_multifilm_s_polarized) num=$M1;;
   propagate_at_multifilm_p_polarized) num=$M1;; 
   propagate_at_multifilm_x_polarized) num=$M1;;   

esac


case $TEST in
          X_rng_sequence)   script=rng_sequence.py   ;;
          rng_sequence)   script=QSimTest.py   ;;
           X_rng_sequence_with_skipahead)   script=rng_sequence_with_skipahead.py   ;;
random_direction_marsaglia) script=random_direction_marsaglia.py ;; 
   boundary_lookup_all)    script=boundary_lookup_all.py ;;
   boundary_lookup_water)  script=boundary_lookup_line.py ;;
   boundary_lookup_ls)     script=boundary_lookup_line.py ;;
       scint_generate)     script=scint_generate.py  ;;
    cerenkov_generate)     script=cerenkov_generate.py  ;;

   fill_state_0)           script=fill_state.py ;;
   fill_state_1)           script=fill_state.py ;;
   fill_state_cf)          script=fill_state_cf.py ;;

   hemisphere_s_polarized) script=hemisphere_polarized.py ;;
   hemisphere_p_polarized) script=hemisphere_polarized.py ;;
   hemisphere_x_polarized) script=hemisphere_polarized.py ;;

   propagate_at_boundary*) script=propagate_at_boundary.py ;; 
   propagate_at_multifilm*) script=propagate_at_multifilm.py ;;

    lambertian_direction)  script=lambertian_direction.py ;; 
         fake_propagate*)  script=fake_propagate.py ;; 
         randgaussq_shoot) script=randgaussq_shoot.py ;; 
            smear_normal*) script=smear_normal.py ;; 
                        *) script=generic.py      ;;
esac


export NUM=${NUM:-$num}
export NRM=${NRM:-$nrm}

loglevels()
{
    #export SEvent=INFO
    export SEvt=INFO
    #export QBnd=INFO
    #export QSim=INFO
    #export QEvent=INFO
    export QNonExisting=INFO 
}
[ -n "$LOG" ] && loglevels


export SPRD_BND=$(cat << EOV
Water///Water
Water///Water
Water///Water
Water///Water
EOV
)


source fill_state.sh 
source ephoton.sh    # branching on TEST inside ephoton.sh 
source eprd.sh
source dbg__.sh 


if [ "$TEST" == "smear_normal_sigma_alpha" ]; then 
   export DBG_VALUE=0.1
elif [ "$TEST" == "rng_sequence_with_skipahead" ]; then

   eventID=0
   export QSimTest__rng_sequence_with_skipahead__eventID=${QSimTest__rng_sequence_with_skipahead__eventID:-$eventID}
   export OPTICKS_EVENT_MODE=Nothing 
fi 



TMP=${TMP:-/tmp/$USER/opticks}
export EBASE=$TMP/GEOM/$GEOM/QSimTest/ALL/p001

vars="BASH_SOURCE arg TEST script NUM NRM FOLD GEOM TMP EBASE"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/grab}" != "$arg" ]; then 
    echo $BASH_SOURCE EBASE $EBASE
    source ../../bin/rsync.sh $EBASE
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin 
   [ $? -ne 0 ] && echo $msg run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   dbg__ $bin
   [ $? -ne 0 ] && echo $msg dbg error && exit 2 
fi

if [ "${arg/pdb}" != "$arg" ]; then 
   echo $BASH_SOURCE pdb script $script
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $msg pdb error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then 
   echo $BASH_SOURCE ana script $script
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $msg ana error && exit 3
fi


if [ "${arg/clean}" != "$arg" ]; then 
   iwd=$PWD
   cd $(dirname $FOLD)

   if [ -d "$TEST" ]; then
       if [ ${#TEST} -gt 3 ]; then
           echo $msg : delete TEST [$TEST] folder from $(dirname $FOLD) namelength ${#TEST} 
           rm -rf $TEST 
       else
            echo $msg : TEST [$TEST] name too short ${#TEST} : NOT DELETING 
       fi 
   else
       echo $msg : TEST [$TEST] folder not present within $(dirname $FOLD) 
   fi 
   cd $iwd
fi





relative_stem(){   ## THIS IS USING OBSOLETE GEOCACHE PATHS 
   local img=$1
   
   local geocache=${OPTICKS_GEOCACHE_PREFIX:-$HOME/.opticks}/geocache/
   local oktmp=/tmp/$USER/opticks/
   
   local rel 
   case $img in 
      ${geocache}*)  rel=${img/$geocache/} ;;
      ${oktmp}*)     rel=${img/$oktmp/} ;;
   esac 
   rel=${rel/\.jpg}
   rel=${rel/\.png}
   
   echo $rel 
}


if [ "${arg/OLDana}" != "$arg" ]; then 

    # PYVISTA_KILL_DISPLAY envvar is observed to speedup exiting from ipython after pyvista plotting 
    # see https://github.com/pyvista/pyvista/blob/main/pyvista/plotting/plotting.py
    export PYVISTA_KILL_DISPLAY=1

    if [ -f "$script" ]; then

        export FOLD="/tmp/QSimTest/$TEST"

        export EYE=-1,-1,1 
        export LOOK=0,0,0
        export UP=0,0,1 
        export PARA=1 

        echo $msg invoking analysis script $script
        ${IPYTHON:-ipython} --pdb -i $script
        [ $? -ne 0 ] && echo $msg ana error && exit 2


        if [ -n "$PUB" ]; then 

            png=$FOLD/figs/pvplt_polarized.png
            rel=$(relative_stem $png)

            if [ "$PUB" == "1" ]; then
                ext=""
            else
                ext="_${PUB}" 
            fi 

            s5p=/env/presentation/${rel}${ext}.png
            pub=$HOME/simoncblyth.bitbucket.io$s5p

            if [ -f "$png" ]; then 

                echo $msg PUB $PUB
                echo $msg png $png 
                echo $msg rel $rel
                echo $msg ext $ext
                echo $msg pub $pub 
                echo $msg s5p $s5p 
              
                if [ -f "$pub" ]; then 
                    echo $msg pub $pub exists already : not copying  
                elif [ "$ext" == "" ]; then 
                    echo $msg set PUB to short descriptive string 
                else
                    mkdir -p $(dirname $pub)
                    echo $msg copy to pub $pub 
                    cp $png $pub
                    echo 
                    echo $msg s5p $s5p 1280px_720px 
                fi 
            fi  
        fi 


    else
        echo $msg there is no analysis script $script
    fi  
fi

exit 0 

