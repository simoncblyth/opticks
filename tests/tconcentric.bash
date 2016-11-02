tconcentric-source(){   echo $(opticks-home)/tests/tconcentric.bash ; }
tconcentric-vi(){       vi $(tconcentric-source) ; }
tconcentric-usage(){ cat << \EOU

tconcentric- 
==================================================


EOU
}
tconcentric-env(){      olocal- ;  }
tconcentric-dir(){ echo $(opticks-home)/tests ; }
tconcentric-cd(){  cd $(tconcentric-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tconcentric-tag(){ echo 1 ; }
tconcentric-det(){ echo concentric ; }
tconcentric-src(){ echo torch ; }

tconcentric--(){
    local cmdline=$*

     local photons=1000000
    #local photons=100000
    #local photons=10000
    #local photons=100

    local g4ppe=10000  # default 10k photons per g4 evt (subevt splitting for G4 memory reasons)
    case $photons in
       1|10|100|1000|10000) g4ppe=$photons ;;
    esac

   local nm=430
   local m1=GdDopedLS

   local torch_config=(
                 type=point
                 mode=fixpol
                 polarization=0,1,0
                 source=0,0,0
                 target=1,0,0
                 photons=$photons
                 material=$m1
                 wavelength=$nm
                 weight=1.0
                 time=0.1
                 zenithazimuth=0,1,0,1
                 radius=0
               )   

    ## tconcentric typically uses a random point source from center point
    ## however handy to use laser for easy interpretation in tconcentric_distrib.py 

    op.sh \
       --g4ppe $g4ppe \
       --test --testconfig "$(tconcentric-testconfig)" \
       --torch \
       --torchconfig "$(join _ ${torch_config[@]})" \
       --animtimemax 30 \
       --timemax 30 \
       --cat $(tconcentric-det) --tag $(tconcentric-tag) --save  \
       --eye 0.5,0.5,0.0 \
       $* 

   ## 300 mm/ns

}
tconcentric-args(){  echo  --tag $(tconcentric-tag) --det $(tconcentric-det) --src $(tconcentric-src) ; }
tconcentric-py(){    tconcentric.py  $(tconcentric-args) $* ; } 
tconcentric-i(){     ipython --profile=g4opticks -i $(which tconcentric.py) --  $(tconcentric-args) $* ; } 
tconcentric-d(){     ipython --profile=g4opticks -i $(which tconcentric_distrib.py) --  $(tconcentric-args) $* ; } 
tconcentric-t()
{
    tconcentric-
    tconcentric-- --okg4 --compute $*
    #tconcentric-py 
}

tconcentric-tt()
{
    tconcentric-t --bouncemax 15 --recordmax 16 $*
}

tconcentric-v()
{
    tconcentric-
    tconcentric-- --okg4 --load $*
}

tconcentric-vg4()
{
    tconcentric-
    tconcentric-- --okg4 --vizg4 --load $*
}

tconcentric-testconfig()
{
    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 shape=sphere
                 boundary=StainlessSteel///Acrylic
                 parameters=0,0,0,$(( 5000 + 5 ))

                 shape=sphere
                 boundary=Acrylic//RSOilSurface/MineralOil
                 parameters=0,0,0,$(( 5000 - 5 ))


                 shape=sphere
                 boundary=MineralOil///Acrylic
                 parameters=0,0,0,$(( 4000 + 5 ))

                 shape=sphere
                 boundary=Acrylic///LiquidScintillator
                 parameters=0,0,0,$(( 4000 - 5 ))


                 shape=sphere
                 boundary=LiquidScintillator///Acrylic
                 parameters=0,0,0,$(( 3000 + 5 ))

                 shape=sphere
                 boundary=Acrylic///GdDopedLS
                 parameters=0,0,0,$(( 3000 - 5 ))

                   ) 

     echo "$(join _ ${test_config[@]})" 
}


tconcentric-testconfig-()
{
     CTestDetectorTest --testconfig "$(tconcentric-testconfig)"  --dbgtestgeo  $* 
}


