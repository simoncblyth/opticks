trainbow-source(){   echo $(opticks-home)/tests/trainbow.bash ; }
trainbow-vi(){       vi $(trainbow-source) $(ggv-source) ; }
trainbow-usage(){ cat << \EOU

trainbow- : Rainbow geometry with torch source
======================================================

`trainbow-pol s/p`
    Op and G4 simulations of rainbow geometry using S or P polarized torch source 
    followed by below -cf comparisons.

`trainbow-cf s/p`
    compare Opticks and Geant4 material/flag sequence histories and make 
    deviation angle plot.

    To see the plots use interactive ipython::

        In [1]: run trainbow.py --tag 6
        In [1]: run trainbow.py --tag 5


`trainbow-test`
    invokes `trainbow-pol` for both s and p  


EOU
}

trainbow-env(){      olocal- ;  }
trainbow-dir(){ echo $(opticks-home)/tests ; }
trainbow-cd(){  cd $(trainbow-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

trainbow-det(){  echo rainbow ; }
trainbow-src(){  echo torch ; }
trainbow-args(){ echo  --det $(trainbow-det) --src $(trainbow-src) ; }

trainbow-poltag()
{
   case $1 in 
     s) echo 5 ;;
     p) echo 6 ;;
   esac
}

trainbow--(){

    local msg="=== $FUNCNAME :"

    local cmdline=$*
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

    local tag=$(trainbow-poltag $pol)
    if [ "${cmdline/--tcfg4}" != "${cmdline}" ]; then
        tag=-$tag  
    fi 

    #local material=GlassSchottF2
    local wavelength=500
    local material=MainH2OHale
    local surfaceNormal=0,1,0
    #local azimuth=-0.25,0.25
    local azimuth=0,1
    local identity=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000

    local photons=1000000
    #local photons=10000

    local torch_config=(
                 type=discIntersectSphere
                 photons=$photons
                 mode=${pol}pol
                 polarization=$surfaceNormal
                 frame=-1
                 transform=$identity
                 source=0,0,600
                 target=0,0,0
                 radius=100
                 distance=25
                 zenithazimuth=0,1,$azimuth
                 material=Vacuum
                 wavelength=$wavelength 
               )
 
    local test_config=(
                 mode=BoxInBox
                 analytic=1
                 shape=box    parameters=0,0,0,1200       boundary=Rock//perfectAbsorbSurface/Vacuum
                 shape=sphere parameters=0,0,0,100         boundary=Vacuum///$material
               )
    echo $msg pol $pol wavelength $wavelength tag $tag tag_offset $tag_offset 

    op.sh  \
            $cmdline \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 0,0,1 \
            --test --testconfig "$(join _ ${test_config[@]})" \
            --torch --torchconfig "$(join _ ${torch_config[@]})" \
            --tag $tag --cat $(trainbow-det) \
            --save

        #    --torchdbg \
}



trainbow-cf() 
{       
   local pol=${1:-s}
   shift 
   local tag=$(trainbow-poltag ${pol})
   trainbow.py $(trainbow-args) --tag $tag $* 
} 

trainbow-pol()
{
   local pol=${1:-s}
   shift 

   trainbow-- --${pol}pol $*
   trainbow-- --${pol}pol --tcfg4

   trainbow-cf $pol
}


trainbow-test()
{
   trainbow-pol s
   trainbow-pol p
}

trainbow-load-g4(){  trainbow-- $* --load --tcfg4 ; } 
trainbow-load() {    trainbow-- $* --load ; } 




