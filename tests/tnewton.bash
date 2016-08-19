tnewton-source(){   echo $(opticks-home)/tests/tnewton.bash ; }
tnewton-vi(){       vi $(tnewton-source) ; }
tnewton-usage(){ cat << \EOU

tnewton- : Pencil beam incident on Glass prism
==================================================

Pencil beam of five wavelengths incident on GlassSchottF2 prism.


`tnewton-vi`
    edit the bash functions 

`tnewton--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`tnewton-py s/p`
    plots simulated prism deviation angle against analytic expectation

`tnewton-test`
    does both s and p simulation in compute mode and runs the py check 



Running with `--tcfg4` option to use geant4 simulation runs into 
missing prism implementation for geant4, causing missing solid warning
and segv. 



EOU
}
tnewton-env(){      olocal- ;  }
tnewton-dir(){ echo $(opticks-home)/tests ; }
tnewton-cd(){  cd $(tnewton-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tnewton-det(){ echo newton ; }
tnewton-src(){ echo torch ; }
tnewton-tag(){
    case ${1:-s} in  
        s) echo 1 ;;
        p) echo 2 ;;
    esac
}

tnewton--(){

    local msg="=== $FUNCNAME :"
    local cmdline=$*
    local det=$(tnewton-det)

    local pol
    if [ "${cmdline/--spol}" != "${cmdline}" ]; then
         pol=s
         cmdline=${cmdline/--spol}
    elif [ "${cmdline/--ppol}" != "${cmdline}" ]; then
         pol=p
         cmdline=${cmdline/--ppol}
    else
         pol=s
    fi  

    local tag=$(tnewton-tag $pol)
    echo  $msg pol $pol tag $tag

    local material=GlassSchottF2
    local surfaceNormal=0,1,0

    local torch_config=(
                 type=point
                 photons=500000
                 mode=${pol}pol,wavelengthComb
                 polarization=$surfaceNormal
                 frame=-1
                 transform=0.500,0.866,0.000,0.000,-0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,-86.603,0.000,0.000,1.000
                 source=-200,200,0
                 target=0,0,0
                 radius=50
                 distance=25
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=0 
               )
 
    local test_config=(
                 mode=BoxInBox
                 analytic=1
                 shape=box   parameters=-1,1,0,700       boundary=Rock//perfectAbsorbSurface/Vacuum
                 shape=prism parameters=60,300,300,200   boundary=Vacuum///$material
               )

    op.sh  \
            $* \
            --animtimemax 7 \
            --timemax 7 \
            --geocenter \
            --eye 0,0,1 \
            --test --testconfig "$(join _ ${test_config[@]})" \
            --torch --torchconfig "$(join _ ${torch_config[@]})" \
            --torchdbg \
            --save --tag $tag --cat $det



}
 
tnewton-args(){ echo --tag $(tnewton-tag ${1:-s}) --det newton --src torch ; } 
tnewton-py() { tprism.py $(tnewton-args $1) ;  }
tnewton-ipy(){ ipython -i $(which tprism.py) -- $(tnewton-args) ; }

tnewton-t()
{
    tnewton-- --spol --compute
    tnewton-- --ppol --compute

    tnewton-py s
    tnewton-py p
}



