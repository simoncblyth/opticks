tlaser-source(){   echo $(opticks-home)/tests/tlaser.bash ; }
tlaser-asource(){  echo $(opticks-home)/ana/tlaser.py ; }
tlaser-vi(){       vi $(tlaser-source) $(tlaser-asource) ; }
tlaser-usage(){ cat << \EOU

tlaser- : Pencil beam 
==================================================

See :doc:`notes/issues/geant4_opticks_integration/reemission_review`


EOU
}
tlaser-env(){      olocal- ;  }
tlaser-dir(){ echo $(opticks-home)/tests ; }
tlaser-cd(){  cd $(tlaser-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tlaser-tag(){ echo 1 ; }

tlaser--(){

    local msg="=== $FUNCNAME :"

    local photons=1000000
    #local photons=100000
    #local photons=20000
    #local photons=100

    local g4ppe=10000  # default 10k photons per g4 evt (subevt splitting for G4 memory reasons)
    case $photons in
       1|10|100|1000|10000) g4ppe=$photons ;;
     esac

    local nm=430
    #local nm=480

    ## beam from center of GDML AD in +X direction, fixpol +Y direction
    local tag=$(tlaser-tag)
    local torch_config=(
                 type=point
                 mode=fixpol
                 polarization=0,1,0
                 frame=3153
                 source=0,0,0
                 target=1,0,0
                 photons=$photons
                 material=GdDopedLS
                 wavelength=$nm
                 weight=1.0
                 time=0.1
                 zenithazimuth=0,1,0,1
                 radius=0
               )

    op.sh  \
            $* \
            --g4ppe $g4ppe \
            --animtimemax 15 \
            --timemax 15 \
            --eye 0,1,0 \
            --torch --torchconfig "$(join _ ${torch_config[@]})" \
            --save --tag $tag --cat laser



}
 
tlaser-args(){ echo --tag $(tlaser-tag) --det laser --src torch ; } 
tlaser-i(){ ipython -i $(which tlaser.py) ; }


tlaser-t(){ tlaser-;tlaser-- --okg4 --compute $* ; } 
tlaser-d(){ tlaser-;tlaser-t --steppingdbg $* ; } 

tlaser-v(){   tlaser-;tlaser-- --okg4 --load $* ; }
tlaser-vg4(){ tlaser-;tlaser-- --okg4 --load --vizg4 $* ; }



tlaser-tfx()
{
    tlaser-t  --fxabconfig 10000 --fxab --fxscconfig 10000 --fxsc --fxreconfig 0.5 --fxre $*
}


