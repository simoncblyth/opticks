tboolean-source(){   echo $(opticks-home)/tests/tboolean.bash ; }
tboolean-vi(){       vi $(tboolean-source) ; }
tboolean-usage(){ cat << \EOU

tboolean- 
======================================================
EOU
}

tboolean-env(){      olocal- ;  }
tboolean-dir(){ echo $(opticks-home)/tests ; }
tboolean-cd(){  cd $(tboolean-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tboolean-tag(){  echo 1 ; }
tboolean-det(){  echo boolean ; }
tboolean-src(){  echo torch ; }
tboolean-args(){ echo  --det $(tboolean-det) --src $(tboolean-src) ; }

tboolean--(){

    tboolean-

    local msg="=== $FUNCNAME :"
    local cmdline=$*

    op.sh  \
            $cmdline \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 1,0,0 \
            --dbganalytic \
            --test --testconfig "$(tboolean-testconfig)" \
            --torch --torchconfig "$(tboolean-torchconfig)" \
            --tag $(tboolean-tag) --cat $(tboolean-det) \
            --save
}



tboolean-torchconfig()
{
    local pol=${1:-s}
    local wavelength=500
    local identity=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000

    #local photons=1000000
    local photons=100000

    local torch_config=(
                 type=disc
                 photons=$photons
                 mode=fixpol
                 polarization=0,1,0
                 frame=-1
                 transform=$identity
                 source=0,0,599
                 target=0,0,0
                 time=0.1
                 radius=110
                 distance=25
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength 
               )
    echo "$(join _ ${torch_config[@]})" 
}


tboolean-testconfig()
{
    local material=GlassSchottF2
    #local material=MainH2OHale

    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 shape=box      parameters=0,0,0,1200               boundary=Rock//perfectAbsorbSurface/Vacuum
 
                 shape=union        parameters=0,0,0,100            boundary=Vacuum///$material
                 shape=box          parameters=0,0,0,100            boundary=Vacuum///$material
                 shape=box          parameters=0,0,0,100            boundary=Vacuum///$material


               )

    #             shape=intersection parameters=0,0,0,400            boundary=Vacuum///$material
    #             shape=sphere       parameters=0,0,-600,641.2          boundary=Vacuum///$material
    #             shape=sphere       parameters=0,0,600,641.2           boundary=Vacuum///$material





     echo "$(join _ ${test_config[@]})" 
}



tboolean-v-g4(){  tboolean-- $* --load --tcfg4 ; } 
tboolean-v() {    tboolean-- $* --load ; } 




