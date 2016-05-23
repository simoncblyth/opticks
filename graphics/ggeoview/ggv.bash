# === func-gen- : graphics/ggeoview/ggv fgp graphics/ggeoview/ggv.bash fgn ggv fgh graphics/ggeoview
ggv-src(){      echo graphics/ggeoview/ggv.bash ; }
ggv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ggv-src)} ; }
ggv-vi(){       vi $(ggv-source) ; }
ggv-usage(){ cat << EOU

GGV : GGeoView invokations
=============================

See Also
---------

* ggeoview- building GGeoView



FUNCTIONS
-----------

ggv-jpmt-cd
      cd into jpmt cache directory 

ggv-pmt
      single uncorrected tesselated DYB PMT in tracer mode (no propagation) 

ggv-ppmt
      DYB geometry with view targetting particular PMT, center AD light source 
      note the OpenGL PMT is uncorrected, but the OptiX geometry is analytic

ggv-allpmt
      just the PMTs geometry in tracer mode, with analytic OptiX geometry

ggv-torpedo
      full geometry with a spherical light source.
      seems source config default has changed since this was developed its no longer torpedo like

ggv-bib
      tracer mode box in box geometry, now with one very fat lens in a box

ggv-pmt-test
      single PMT validation test, compare Opticks and Geant4::

         ggv-pmt-test --cfg4       # G4
         ggv-pmt-test              # Opticks interop
         ggv-pmt-test --compute    # Opticks compute

         ggv-pmt-test --cdetector    # 

ggv-dpib-test
      DYB Pmt In Box, note the OptiX geometry is tesselated::

         ggv-dpib-test --tracer

ggv-box-test
      Box in Box, disc beam 


many more undocumented




EOU
}
ggv-env(){      elocal- ; }
ggv-dir(){ echo $(env-home)/graphics/ggeoview ; }
ggv-cd(){  cd $(ggv-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

ggv-jpmt-cd(){   cd $(ggv-jpmt-idp) ; }
ggv-jpmt-idp(){  echo $(op.sh --jpmt --idp 2>/dev/null) ; }

ggv-pmt(){
   type $FUNCNAME
   local test_config=(
                 mode=PmtInBox
                 analytic=1

                 shape=sphere
                 boundary=Rock//perfectAbsorbSurface/MineralOil
                 parameters=-1,1,0,300
                   ) 

   op.sh --tracer \
          --test --testconfig "$(join _ ${test_config[@]})" \
          --eye 0.5,0.5,0.0 \
           $*  
}


ggv-ppmt(){   op.sh --analyticmesh 1 --target 3199 --torchconfig "radius=1500_zenithazimuth=0,1,0,1" $* ; }

ggv-allpmt(){ op.sh --tracer --restrictmesh 1 --analyticmesh 1 $* ; }

ggv-torpedo(){ op.sh --analyticmesh 1 --torchconfig "frame=3199_source=0,0,1000_target=0,0,0_radius=150_zenithazimuth=0,1,0,1" $* ; }

ggv-bib(){
   type $FUNCNAME
   local test_config=(
                 mode=BoxInBox
                 analytic=1

                 shape=box
                 boundary=Rock//perfectAbsorbSurface/MineralOil
                 parameters=-1,1,0,500

                 shape=lens
                 boundary=MineralOil///Pyrex
                 parameters=100,100,-50,50
                   ) 

   op.sh --tracer \
          --test --testconfig "$(join _ ${test_config[@]})" \
          --eye 0.5,0.5,0.0 \
           $*  
}

ggv-pmt-test(){
   type $FUNCNAME

    local msg="=== $FUNCNAME :"

    local cmdline=$*
    local tag=4
    local photons=500000
    if [ "${cmdline/--g4ui}" != "${cmdline}" ]; then
        photons=10000
    fi 

    local zenith 
    local note

    case $tag in
      1) zenith=0,0.97  ; note="ok"      ;;
      2) zenith=0.97,1        ;;
      3) zenith=0.9671,0.9709 ;;
      4) zenith=0.0001,1      ;;
      5) zenith=0.99999,1     ;;
    esac

    #local typ=disclin
    local typ=disc
    local src=0,0,300
    local tgt=0,0,0
    local radius=100
    local testverbosity=3 

    local mode=""
    local polarization=""
 
    if [ "$tag" == "5" ]; then
        typ=point
        src=99,0,300
        tgt=99,0,0
        mode=fixpol
        polarization=0,1,0

    elif [ "$tag" == "6" ]; then

        src=99,0,300
        tgt=99,0,0
        radius=0.5
        zenith=0,1
    fi  

    local torch_config=(
                 type=$typ
                 photons=$photons
                 wavelength=380 
                 frame=1
                 source=$src
                 target=$tgt
                 radius=$radius
                 zenithazimuth=$zenith,0,1
                 material=Vacuum

                 mode=$mode
                 polarization=$polarization
               )


    local test_config=(
                 mode=PmtInBox
                 pmtpath=$IDPATH_DPIB_PMT/GMergedMesh/0
                 control=$testverbosity,0,0,0
                 analytic=1
                 groupvel=1
                 shape=box
                 boundary=Rock//perfectAbsorbSurface/MineralOil
                 parameters=0,0,0,300
                   ) 

    if [ "${cmdline/--cfg4}" != "${cmdline}" ]; then
        tag=-$tag  
        g4-
        g4-export
        env | grep G4
    fi 

    ## hmm such pre-launch environment setup should happen inside op.sh 
    ## to avoid duplication

   op.sh \
       --test --testconfig "$(join _ ${test_config[@]})" \
       --torch --torchconfig "$(join _ ${torch_config[@]})" \
       --timemax 10 \
       --animtimemax 10 \
       --cat PmtInBox --tag $tag --save \
       --eye 0.0,-0.5,0.0 \
       --geocenter \
       $* 

}



ggv-phycache(){ echo /tmp/$FUNCNAME ; }

ggv-g4gun()
{
    type $FUNCNAME

    local msg="=== $FUNCNAME :"
    local tag=-1

    g4-
    g4-export
    env | grep G4

    local g4gun_config=(
                 comment=$FUNCNAME
                 particle=mu-
                 number=1
                 frame=3153
                 position=0,0,0
                 direction=0,0,1
              polarization=1,0,0
                      time=0.
                    energy=10.0
                   ) 
          # mm, ns, MeV


   local phycache=$(ggv-phycache)
   if [ ! -d "$phycache" ]; then 
      mkdir -p $phycache
   fi

   local inimac=/tmp/g4ini.mac
   cat << EOI > $inimac
/OpNovice/phys/verbose 0
/run/particle/verbose 1
/run/particle/retrievePhysicsTable $phycache
EOI


   local finmac
   if [ -f "$phycache/material.dat" ]; then 
       finmac="-"
   else
       finmac=/tmp/g4fin.mac
       cat << EOI > $finmac
/run/particle/storePhysicsTable $phycache
EOI
   fi


   op.sh \
       --cfg4 \
       --cat G4Gun --tag $tag --save \
       --g4inimac "$inimac" \
       --g4finmac "$finmac" \
       --g4gun --g4gundbg --g4gunconfig "$(join _ ${g4gun_config[@]})" \
       $* 

}

ggv-g4gun-notes(){ cat <<EON

Lots of G4 init noise from:

   G4VUserPhysicsList::BuildPhysicsTable

   find source -name 'G4VEmProcess.hh'

EON
}



ggv-dpib-test()
{
    type $FUNCNAME

    local msg="=== $FUNCNAME :"

    local cmdline=$*
    local photons=500000
    local tag=4
    local zenith 
    local note

    case $tag in
      1) zenith=0,0.97  ; note="ok"      ;;
      2) zenith=0.97,1        ;;
      3) zenith=0.9671,0.9709 ;;
      4) zenith=0.0001,1      ;;
      5) zenith=0.99999,1     ;;
    esac


    local typ=disc
    local src=0,0,300
    local tgt=0,0,0
    local radius=100

    local mode=""
    local polarization=""
 
    local torch_config=(
                 type=$typ
                 photons=$photons
                 wavelength=380 
                 frame=1
                 source=$src
                 target=$tgt
                 radius=$radius
                 zenithazimuth=$zenith,0,1
                 material=Vacuum

                 mode=$mode
                 polarization=$polarization
               )

   op.sh \
       --dpib \
       --torch --torchconfig "$(join _ ${torch_config[@]})" \
       --timemax 10 \
       --animtimemax 10 \
       --cat DPIB --tag $tag --save \
       --eye 0.0,-0.5,0.0 \
       --geocenter \
       $* 

}



ggv-box-test(){
    type $FUNCNAME


    local cmdline=$*
    local tag=1
    if [ "${cmdline/--cfg4}" != "${cmdline}" ]; then
        tag=-$tag  
    fi 

    local photons=500000
    if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
        photons=1
    fi

    local torch_config=(
                 type=disclin
                 photons=$photons
                 wavelength=380 
                 frame=1
                 source=0,0,300
                 target=0,0,0
                 radius=100
                 zenithazimuth=0,1,0,1
                 material=Vacuum
               )

    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 shape=box
                 boundary=Rock//perfectAbsorbSurface/MineralOil
                 parameters=0,0,0,300

                 shape=box
                 boundary=MineralOil///Pyrex
                 parameters=0,0,0,100
                   ) 

    op.sh \
       --test --testconfig "$(join _ ${test_config[@]})" \
       --torch --torchconfig "$(join _ ${torch_config[@]})" \
       --animtimemax 10 \
       --timemax 10 \
       --cat BoxInBox --tag $tag --save  \
       --eye 0.5,0.5,0.0 \
       $* 

}



    
# For invsphere and refltest "source" provides the
# center of the sphere and target is not used
# the photons start from a position on the sphere and go in 
# direction towards the center of the sphere 
# where the "source" position provides the center of the sphere 
# (so in this case the target is not used)
#
#
# no leaks with analytic box,  so no need to offset *source* to eg 10,0,300 to avoid cracks
# analytic sphere needs bbox expanded slightly to avoid leaks at touch points
#


ggv-jpmt-usage(){ cat << EOU

Propagating and viz in same invokation running out of GPU memory, so do compute and viz separately::

    op.sh --jpmt --cerenkov --compute --timemax 400 --save
    op.sh --jpmt --cerenkov --load
    op.sh --jpmt --cerenkov --load 

EOU
}


ggv-jpmt-propagate-cerenkov(){
    op.sh --jpmt --cerenkov --tag 1_check --compute --timemax 400 --animtimemax 200 --save
}
ggv-jpmt-propagate-scintillation(){
    op.sh --jpmt --scintillation --tag 1_mod1000 --compute --timemax 400 --animtimemax 200 --save
}


ggv-jpmt-viz(){
    op.sh --jpmt --cerenkov --animtimemax 200 --load  --size 1024,768,2 $*
      # --optixviz
}

jpmt(){ 
    #local spa=retina
    local spa=hd
    op.sh --jpmt --cerenkov --animtimemax 80 --load $(ggv-size-position $spa)  --state fly0 --ivperiod 250 --ovperiod 360 $*
}



ggv-vid-x(){ echo 100 ; }
ggv-vid-y(){ echo 100 ; }
ggv-vid-w(){ echo 1920 ; }
ggv-vid-h(){ echo 1080 ; }

ggv-size-position()
{
    local arg=${1:-hd}
    local pt2px=2

    local position=100,100

    local vga=640,480,$pt2px
    local projector=1024,768,$pt2px 
    local retina=2880,1704,$pt2px
    local retina_full=2880,1800,$pt2px
    local projector_wide=1920,1080,$pt2px 

    local size=$retina
 
    if [ "$arg" == "hd" ];  then
        size=$(ggv-vid-w),$(ggv-vid-h),$pt2px 
        position=$(ggv-vid-x),$(ggv-vid-y)
    elif [ "$arg" == "retina_full" ];  then
        size=$retina_full
    elif [ "$arg" == "retina" ];  then
        size=$retina
    elif [ "$arg" == "projector" ];  then
        size=$projector
    elif [ "$arg" == "projector_wide" ];  then
        size=$projector_wide
    elif [ "$arg" == "vga" ];  then
        size=$vga
    fi
    echo --size $size --position $position
}

ggv-hd-info(){ cat << EOU

See vids-vi

EOU
}

ggv-hd-test()
{
    op.sh --tracer $(ggv-size-position hd)
}

ggv-hd-capture()
{
    ggv-hd-info

    local x=$(ggv-vid-x)
    local y=$(ggv-vid-y)
    local w=$(( $(ggv-vid-w)/2 ))
    local h=$(( $(ggv-vid-h)/2 ))

    local cmd="caperture.swift -x $x -y $y -w $w -h $h "
    echo $cmd
    eval $cmd
}


ggv-dyb()
{
    op.sh --analyticmesh 1 --cerenkov --animtimemax 80 $(ggv-size-position hd) --load --optixviz --state fly0 --ivperiod 250 --ovperiod 360 --evperiod 400 $*
}



ggv-wavelength()
{
    local color=${1:-white}
    local X_red=600.0
    local Y_green=554.0
    local Z_blue=448.0   
    # huh blue coming out magenta 

    local tag=0
    local wavelength=0
    # wavelength=0 for Plankian D65 white light 

    case $color in  
        white) wavelength=0        ; tag=1 ;;
          red) wavelength=$X_red   ; tag=2 ;;
        green) wavelength=$Y_green ; tag=3 ;;
         blue) wavelength=$Z_blue  ; tag=4 ;;
    esac
    echo  color $color wavelength $wavelength tag $tag
}


ggv-rainbow-usage(){ cat << EOU

ggv-;ggv-rainbow 
ggv-;ggv-rainbow --ppol
ggv-;ggv-rainbow --spol
ggv-;ggv-rainbow --cfg4 --spol
ggv-;ggv-rainbow --cfg4 --ppol
ggv-;ggv-rainbow --cfg4 --ppol --dbg


ggv-;ggv-rainbow --load
    # load and visualized photons persisted from Opticks

ggv-;ggv-rainbow --cfg4 --load
    # load and visualize photons persisted from Geant4 (cfg4) simulation

EOU
}


ggv-catdir(){ echo $LOCAL_BASE/env/opticks/${1:-rainbow} ; }

ggv-times(){
   local udet=${1:-rainbow}
   find $(ggv-catdir $udet)  -name t_delta.ini -exec grep -H ^propagate {} \;
}

ggv-param(){
   local parm=${1:-photonData}
   local udet=${2:-rainbow}
   find $(ggv-catdir $udet)  -name parameters.json -exec grep -H $parm {} \;
}

ggv-rainbow()
{
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

    local tag

    case $pol in 
       s) tag=5 ;;   
       p) tag=6 ;;   
    esac

    if [ "${cmdline/--cfg4}" != "${cmdline}" ]; then
        tag=-$tag  
    fi 


    #local material=GlassSchottF2
    local material=MainH2OHale
    local surfaceNormal=0,1,0
    #local azimuth=-0.25,0.25
    local azimuth=0,1
    #local wavelength=0
    local wavelength=500
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

    #args.sh  \
    op.sh  \
            $cmdline \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 0,0,1 \
            --test --testconfig "$(join _ ${test_config[@]})" \
            --torch --torchconfig "$(join _ ${torch_config[@]})" \
            --tag $tag --cat rainbow \
            --save

        #    --torchdbg \
}

ggv-newton()
{
    type $FUNCNAME



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


    local tag=1 
    case $pol in  
        s) tag=1 ;;
        p) tag=2 ;;
    esac
    echo  pol $pol tag $tag


    local material=GlassSchottF2
    local surfaceNormal=0,1,0

    # i1_mindev arcsin( n*sin(a/2)) ) = arcsin(n/2)    for sin(60/2)=0.5 
    #In [10]: 1./np.tan(np.arcsin(1.613/2.))
    #Out[10]: 0.73308628728462222
    #In [11]: 500.*1./np.tan(np.arcsin(1.613/2.))
    #Out[11]: 366.54314364231112

    local torch_config=(
                 type=point
                 photons=500000
                 polz=${pol}pol
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
            --save --tag $tag --cat newton

}

ggv-reflect()
{
    type $FUNCNAME

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

    local tag=1 
    case $pol in  
        s) tag=1 ;;
        p) tag=2 ;;
    esac
    if [ "${cmdline/--cfg4}" != "${cmdline}" ]; then
        tag=-$tag  
    fi 
    echo  pol $pol tag $tag


    local photons=10000

    #local material=GlassSchottF2
    local material=MainH2OHale

    # target is ignored for refltest, source is the focus point 

    local torch_config=(
                 type=refltest
                 photons=$photons
                 mode=${pol}pol,flatTheta
                 polarization=0,0,-1
                 frame=-1
                 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000
                 source=0,0,-200
                 radius=100
                 distance=25
                 zenithazimuth=0.5,1,0,1
                 material=Vacuum
                 wavelength=550
               )

    local test_config=(
                 mode=BoxInBox
                 analytic=1
                 shape=box   parameters=0,0,0,1000       boundary=Rock//perfectAbsorbSurface/Vacuum
                 shape=box   parameters=0,0,0,200        boundary=Vacuum///$material
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
            --save --tag $tag --cat reflect
}




ggv-prism()
{
    type $FUNCNAME
    local pol=${1:-s}
    local tag=1 
    case $pol in  
        s) tag=1 ;;
        p) tag=2 ;;
    esac
    echo  pol $pol tag $tag


    # position from: source
    # direction from: target - source
    #
    # zenithazimuth .z:.w  provides azimuth phi fraction range
    # ie the generated random range within 0:1 (that is mapped to 0:2pi) 
    #
    # apex angle 60, half = 30
    # phi=0 corresponds to x axis
    # offset phi by 180-90-30=60 
    # hence phifrac0 = 60/360
    #
    # focus on critical angle 24.7632939454   (60.+90.+24.763)/360. = 0.4854527777777778
    #

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

    # below transform string obtained from prism.py s_i2w
    # is the 
    #
    #     origin within the face 
    #     +Y normal to face 
    #     +X towards the apex 
    #
    # with invcylinder
    #     *distance* corresponds to length of the cylinder 
    #     using 25mm is good for ripple tank effect 
    # 
    #     *polarization* used to carry in the surface normal in intersect frame
    # 

    local torch_config=(
                 type=invcylinder
                 photons=500000
                 polz=${pol}pol
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


    #echo >/dev/null op.sh  \
    #ggv.py \ 
    op.sh  \
            $* \
            --animtimemax 7 \
            --timemax 7 \
            --geocenter \
            --eye 0,0,1 \
            --test --testconfig "$(join _ ${test_config[@]})" \
            --torch --torchconfig "$(join _ ${torch_config[@]})" \
            --torchdbg \
            --save --tag $tag --cat prism

}



ggv-lens()
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
                 polz=${pol}pol
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
            --save --tag $tag --cat lens
}






