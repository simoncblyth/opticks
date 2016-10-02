tlaser-source(){   echo $(opticks-home)/tests/tlaser.bash ; }
tlaser-vi(){       vi $(tlaser-source) ; }
tlaser-usage(){ cat << \EOU

tlaser- : Pencil beam 
==================================================

EOU
}
tlaser-env(){      olocal- ;  }
tlaser-dir(){ echo $(opticks-home)/tests ; }
tlaser-cd(){  cd $(tlaser-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tlaser-tag(){ echo 1 ; }

tlaser--(){

    local msg="=== $FUNCNAME :"

    local tag=$(tlaser-tag)
    local torch_config=(
                 type=point
                 frame=3153
                 source=0,0,0
                 target=1,0,0
                 photons=10000
                 material=GdDopedLS
                 wavelength=430
                 weight=1.0
                 time=0.1
                 zenithazimuth=0,1,0,1
                 radius=0
               )

    op.sh  \
            $* \
            --animtimemax 15 \
            --timemax 15 \
            --eye 0,1,0 \
            --torch --torchconfig "$(join _ ${torch_config[@]})" \
            --torchdbg \
            --save --tag $tag --cat laser



}
 
tlaser-args(){ echo --tag $(tlaser-tag) --det laser --src torch ; } 
tlaser-i(){ ipython -i $(which tlaser.py) ; }


tlaser-okg4(){ tlaser-;tlaser-- --okg4 --compute ; } 

