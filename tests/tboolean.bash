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
    #local photons=1

    local torch_config_disc=(
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
                 distance=200
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength 
               )


    local discaxial_target=0,0,0
    local torch_config_discaxial=(
                 type=discaxial
                 photons=$photons
                 frame=-1
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

    echo "$(join _ ${torch_config_discaxial[@]})" 
}



tboolean-material(){ echo GlassSchottF2 ; }
#tboolean-material(){ echo MainH2OHale ; }



tboolean-box()
{
    local material=$(tboolean-material)
    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 node=box      parameters=0,0,0,1200               boundary=Rock//perfectAbsorbSurface/Vacuum
                 node=box      parameters=0,0,0,100                boundary=Vacuum///$material

                    )
     echo "$(join _ ${test_config[@]})" 
}



tboolean-box-small-offset-sphere()
{
    local operation=${1:-difference}
    local material=$(tboolean-material)
    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 node=sphere      parameters=0,0,0,1000          boundary=Rock//perfectAbsorbSurface/Vacuum
 
                 node=$operation   parameters=0,0,0,300           boundary=Vacuum///$material
                 node=box          parameters=0,0,0,200           boundary=Vacuum///$material
                 node=sphere       parameters=0,0,200,100         boundary=Vacuum///$material
               )

     echo "$(join _ ${test_config[@]})" 
}

tboolean-box-sphere()
{
    local operation=${1:-difference}
    local material=$(tboolean-material)
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 node=box          parameters=0,0,0,1000          boundary=Rock//perfectAbsorbSurface/Vacuum
 
                 node=$operation   parameters=0,0,0,300           boundary=Vacuum///$material
                 node=box          parameters=0,0,0,$inscribe     boundary=Vacuum///$material
                 node=sphere       parameters=0,0,0,200           boundary=Vacuum///$material
               )

     echo "$(join _ ${test_config[@]})" 
}



tboolean-csg-notes(){ cat << EON

* CSG tree is defined in breadth first order

* parameters of boolean operations currently define adhoc box 
  intended to contain the geometry, TODO: calculate from bounds of the contained tree 

* offsets arg identifies which nodes belong to which primitives by pointing 
  at the nodes that start each primitive

EON
}

tboolean-csg()
{
    local material=$(tboolean-material)
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local radius=200

    local test_config=(
                      mode=CsgInBox
                      analytic=1
                      offsets=0,1

                      node=box          parameters=0,0,0,1000          boundary=Rock//perfectAbsorbSurface/Vacuum

                      node=union        parameters=0,0,0,400           boundary=Vacuum///$material
                      node=difference   parameters=0,0,100,300         boundary=Vacuum///$material
                      node=difference   parameters=0,0,-100,300        boundary=Vacuum///$material
                      node=box          parameters=0,0,100,$inscribe   boundary=Vacuum///$material
                      node=sphere       parameters=0,0,100,$radius     boundary=Vacuum///$material
                      node=box          parameters=0,0,-100,$inscribe  boundary=Vacuum///$material
                      node=sphere       parameters=0,0,-100,$radius    boundary=Vacuum///$material
  
                      )

    echo "$(join _ ${test_config[@]})" 
}



tboolean-testconfig()
{
    #tboolean-box-sphere intersection    ## looks like a dice, sphere chopped by cube
    #tboolean-box-sphere union
    #tboolean-box-sphere difference
     tboolean-csg

    #tboolean-box
    #tboolean-box-small-offset-sphere difference
}



