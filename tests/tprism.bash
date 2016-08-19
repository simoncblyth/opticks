tprism-source(){   echo $(opticks-home)/tests/tprism.bash ; }
tprism-vi(){       vi $(tprism-source) ; }
tprism-usage(){ cat << \EOU

tprism- : Cylindrical light source focussed on prism face point 
==================================================================

White light incident on glass prism from all incidence angles.

A frame transform is supplied in the torch config
which is calculated from prism geometry/position (see :doc:`../ana/tprism`).
This allows source and target positions to be expressed in 
"natural" coordinates with the frame that has:

* origin within prism face
* +Y normal to the face
* +X towards the apex 
 

.. code-block:: sh

    local torch_config=(
                 type=invcylinder
                 frame=-1
                 transform=0.500,0.866,0.000,0.000,-0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,-86.603,0.000,0.000,1.000
                 target=0,-500,0
                 source=0,0,0 
               )


With **invcylinder** source type 

* *distance* corresponds to length of the cylinder, using 25mm is good for ripple tank effect 
* *polarization* is mis-used to carry in the surface normal in intersect frame




`tprism-vi`
    edit the bash functions 

`tprism--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`tprism-- --compute`
    create Opticks geometry, simulates photons in compute mode, saves evt file


`tprism-cf`
    compare Opticks and Geant4 material/flag sequence histories




EOU
}
tprism-env(){      olocal- ;  }
tprism-dir(){ echo $(opticks-home)/tests ; }
tprism-cd(){  cd $(tprism-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tprism-det(){ echo prism ; }
tprism-src(){ echo torch ; }



tprism-tag()
{
   case ${1:-s} in 
      s) echo 1 ;;
      p) echo 2 ;;
   esac
}

tprism--(){
    local msg="=== $FUNCNAME :"

    local cmdline=$*
    local pol
    if [ "${cmdline/--spol}" != "${cmdline}" ]; then
        pol=s
    elif [ "${cmdline/--ppol}" != "${cmdline}" ]; then
        pol=p
    else
        pol=s
    fi
    
    local tag=$(tprism-tag $pol)

    echo  $msg pol $pol tag $tag

    local phifrac0=0.1667  # 60/360  
    local phifrac1=0.4167  # 0.1667+0.250   from 1-2 is selects lower quadrant 
    local phifrac2=0.6667  # 0.1667+0.5
    local quadrant=$phifrac1,$phifrac2  

    # phifrac range around critical angle
    local critical=0.4854,0.4855       # 19k sneak out, grazing prism face  
    #local critical=0.485452,0.485453     # no "BT BT SA" getting TIR at 2nd interface


    #local material=Pyrex
    local material=GlassSchottF2

    local azimuth=$quadrant
    local surfaceNormal=0,1,0

    local torch_config=(
                 type=invcylinder
                 photons=500000
                 mode=${pol}pol,wavelengthComb
                 polarization=$surfaceNormal
                 frame=-1
                 transform=0.500,0.866,0.000,0.000,-0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,-86.603,0.000,0.000,1.000
                 target=0,-500,0
                 source=0,0,0 
                 radius=300
                 distance=25
                 zenithazimuth=0,1,$azimuth
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
            --save --tag $tag --cat $(tprism-det)

    local RC=$?
    echo $FUNCNAME RC $RC
    return $RC
}


tprism-args() {        echo  --det $(tprism-det) --src $(tprism-src) ; }

tprism-py() {         
    local pol=${1:-s}
    shift 
    local tag=$(tprism-tag $pol)
    echo  $msg pol $pol tag $tag
    tprism.py $(tprism-args) --tag $tag $*
} 

tprism-ipy(){
    local pol=${1:-s}
    shift 
    ipython -i $(which tprism.py) -- $(tprism-args) --tag $(tprism-tag $pol)
}

tprism-pol()
{
    local pol=${1:-s}
    shift 
    local tag=$(tprism-tag $pol)
    echo  $msg pol $pol tag $tag

    tprism-- --${pol}pol --compute   
}

tprism-t()
{
    tprism-pol s 
    tprism-pol p

    tprism-py s
    tprism-py p
}



