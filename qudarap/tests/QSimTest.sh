#!/bin/bash -l 

usage(){ cat << EOU
QSimTest.sh
=============

::

    ./QSimTest.sh 
        run the executable and invoke the python script  

    TEST=fill_state_cf ./QSimTest.sh ana
        just invoke the analysis script for the named TEST 

EOU
}


arg=${1:-run_ana}

msg="=== $BASH_SOURCE :"

#export QBnd=INFO

#test=rng_sequence

#test=fill_state_0
#test=fill_state_1

#test=water
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

test=mock_propagate

M1=1000000
K2=100000

num=8
#num=$K2
#num=$M1

nrm=0,0,1
#nrm=0,0,-1


export NUM=${NUM:-$num}
export NRM=${NRM:-$nrm}
export TEST=${TEST:-$test}

source fill_state.sh 

case $TEST in 
   propagate_at_boundary_normal_incidence) source ephoton.sh ;;
                    propagate_at_boundary) source ephoton.sh ;; 
                          reflect_diffuse) source ephoton.sh ;;
                         reflect_specular) source ephoton.sh ;;
esac


if [ "${arg/run}" != "$arg" ]; then 
   if [ -n "$DEBUG" ]; then 
       lldb__ QSimTest
   else
       QSimTest
   fi 
   [ $? -ne 0 ] && echo $msg run error && exit 1 
fi



relative_stem(){
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


if [ "${arg/ana}" != "$arg" ]; then 

    # PYVISTA_KILL_DISPLAY envvar is observed to speedup exiting from ipython after pyvista plotting 
    # see https://github.com/pyvista/pyvista/blob/main/pyvista/plotting/plotting.py
    export PYVISTA_KILL_DISPLAY=1


    export FOLD=/tmp/$USER/opticks/QSimTest/$TEST 

    case $TEST in
       fill_state_0)           script=fill_state.py ;;
       fill_state_1)           script=fill_state.py ;;
       fill_state_cf)          script=fill_state_cf.py ;;

       hemisphere_s_polarized) script=hemisphere_polarized.py ;;
       hemisphere_p_polarized) script=hemisphere_polarized.py ;;
       hemisphere_x_polarized) script=hemisphere_polarized.py ;;

       propagate_at_boundary*) script=propagate_at_boundary.py ;; 
   random_direction_marsaglia) script=random_direction_marsaglia.py ;; 
        lambertian_direction)  script=lambertian_direction.py ;; 

                            *) script=$TEST.py      ;;
    esac

    if [ -f "$script" ]; then

        export FOLD="/tmp/$USER/opticks/QSimTest/$TEST"

        export EYE=-1,-1,1 
        #export LOOK=0,0,0.5 
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

