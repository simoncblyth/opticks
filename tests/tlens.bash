tlens-source(){   echo $(opticks-home)/tests/tlens.bash ; }
tlens-vi(){       vi $(tlens-source) ; }
tlens-usage(){ cat << \EOU

tlens- : Disc shaped beam of white light incident on convex lens  
====================================================================


`tlens-vi`
    edit the bash functions 

`tlens--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 


EXERCISE
-----------

* Change the lens material and interpret what you get, 
  see :doc:`overview` regarding materials.

* Try adding one or more lens, for example with line::

     shape=lens  parameters=641.2,641.2,-400,800 boundary=Vacuum///$material

* Try adding a different shape, examine the **GMaker** source code :oktip:`ggeo/GMaker.cc`
  to see what shapes are available

* Write a python analysis script **tlens.py** that 
  
    * loads an event from `tlens-`
    * prints the photon history table
    * select a subset of the photons (using **seqs** argument to Evt class)
    * plot distributions for the subset using **matplotlib**  
    * interpret the plot

   


EOU
}
tlens-env(){      olocal- ;  }
tlens-dir(){ echo $(opticks-home)/tests ; }
tlens-cd(){  cd $(tlens-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tlens-det(){ echo lens ; }
tlens-src(){ echo torch ; }

tlens-args() {        echo  --det $(tlens-det) --src $(tlens-src) ; }
tlens-py() {          tlens.py  $(tlens-args) $* ; } 

tlens--()
{
    type $FUNCNAME
    local pol=${1:-s}
    case $pol in  
        s) tag=1 ;;
        p) tag=2 ;;
    esac
    echo  pol $pol tag $tag

    local material=GlassSchottF2

    local torch_config=(
                 type=disc
                 photons=500000
                 mode=${pol}pol,wavelengthSource
                 frame=-1
                 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000
                 target=0,0,0
                 source=0,0,-600
                 radius=100
                 distance=500
                 zenithazimuth=0,1,0,1
                 material=Vacuum
               )

    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 shape=box   parameters=-1,1,0,700           boundary=Rock//perfectAbsorbSurface/Vacuum
                 shape=lens  parameters=641.2,641.2,-600,600 boundary=Vacuum///$material
               )

    op.sh  \
            $* \
            --animtimemax 7 \
            --timemax 7 \
            --geocenter \
            --eye 0,1,0 \
            --up  1,0,0 \
            --test --testconfig "$(join _ ${test_config[@]})" \
            --torch --torchconfig "$(join _ ${torch_config[@]})" \
            --torchdbg \
            --save --tag $tag --cat $(tlens-det)
}

tlens-t()
{
    tlens-- --compute
}


