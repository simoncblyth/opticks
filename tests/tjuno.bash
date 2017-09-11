tjuno-source(){   echo $(opticks-home)/tests/tjuno.bash ; }
tjuno-vi(){       vi $(tjuno-source) ; }
tjuno-usage(){ cat << \EOU
tjuno- 
======================================================
 
FUNCTIONS
----------
EOU
}

tjuno-env(){      olocal- ;  }
tjuno-dir(){ echo $(opticks-home)/tests ; }
tjuno-cd(){  cd $(tjuno-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tjuno-tag(){  echo 1 ; }
tjuno-det(){  echo boolean ; }
tjuno-src(){  echo torch ; }
tjuno-args(){ echo  --det $(tjuno-det) --src $(tjuno-src) ; }

tjuno-ls-(){ grep TESTCONFIG= $BASH_SOURCE ; }
tjuno-ls(){ $FUNCNAME- | perl -ne 'm/(\S*)\(\)/ && print "$1\n" ' -   ; }

tjuno--(){

    tjuno-

    local msg="=== $FUNCNAME :"
    local cmdline=$*

    #local stack=4096
    local stack=2180  # default

    #        --rendermode "-axis" \

    op.sh  \
            $cmdline \
            --animtimemax 200 \
            --timemax 200 \
            --geocenter \
            --stack $stack \
            --eye 1,0,0 \
            --dbganalytic \
            --torch --torchconfig "$(tjuno-torchconfig)" \
            --torchdbg \
            --tag $(tjuno-tag) --cat $(tjuno-det) \
            --save 
}

tjuno-tracetest()
{
    tjuno-- --tracetest $*
}

tjuno-torchconfig()
{

    local pol=${1:-s}
    local wavelength=500
    local identity=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000

    #local photons=1000000
    local photons=100000
    #local photons=1

    local torch_config_disc=(
                 type=disc
                 photons=$photons
                 mode=fixpol
                 polarization=1,1,0
                 frame=-1
                 transform=$identity
                 source=0,0,599
                 target=0,0,0
                 time=0.1
                 radius=300
                 distance=200
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength 
               )


    local discaxial_target=0,0,0
    local torch_config_discaxial=(
                 type=discaxial
                 photons=$photons
                 frame=12
                 transform=$identity
                 source=$discaxial_target
                 target=0,0,0
                 time=0.1
                 radius=100
                 distance=400
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength 
               )



    local torch_config_sphere=(
                 type=sphere
                 photons=10000
                 frame=12
                 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1000.000,1.000
                 source=0,0,0
                 target=0,0,1
                 time=0.1
                 radius=100
                 distance=400
                 zenithazimuth=0,1,0,1
                 material=LS
                 wavelength=$wavelength 
               )



    #echo "$(join _ ${torch_config_discaxial[@]})" 
    #echo "$(join _ ${torch_config_disc[@]})" 
    echo "$(join _ ${torch_config_sphere[@]})" 
}


