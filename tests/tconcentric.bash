tconcentric-source(){   echo $(opticks-home)/tests/tconcentric.bash ; }
tconcentric-vi(){       vi $(tconcentric-source) ; }
tconcentric-usage(){ cat << \EOU

tconcentric- 
==================================================

::

    tconcentric-d --noplot --rehist --sel 0:100
        ## rehistogramming and making chi2 distrib comparisons
        ## after this use "ip> run abstat.py" to examine the results 




EOU
}
tconcentric-env(){      olocal- ;  }
tconcentric-dir(){ echo $(opticks-home)/tests ; }
tconcentric-cd(){  cd $(tconcentric-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tconcentric-tag(){ echo 1 ; }
tconcentric-det(){ echo concentric ; }
tconcentric-src(){ echo torch ; }

tconcentric-photons(){
    local photons=1000000
   #local photons=100000
   #local photons=10000
   #local photons=100
   #local photons=30
   #local photons=10
   #local photons=2
   echo $photons
}
tconcentric-oindex(){
   local photons=$(tconcentric-photons)
   local num=${1:-10}
   echo $(join , $(seq $(( $photons - 1 )) -1 $(( $photons - $num ))))
}

tconcentric--(){
    local cmdline=$*
    local photons=$(tconcentric-photons)
    local g4ppe=10000  # default 10k photons per g4 evt (subevt splitting for G4 memory reasons)
    case $photons in
       1|2|3|4|5|6|7|8|9|10|20|30|40|50|60|70|80|90|100|1000|10000) g4ppe=$photons ;;
    esac

    local animtimemax=132
    local timemax=132

    op.sh \
       --g4ppe $g4ppe \
       --test --testconfig "$(tconcentric-testconfig)" \
       --torch \
       --torchconfig "$(tconcentric-torchconfig)" \
       --animtimemax $animtimemax \
       --timemax $timemax \
       --cat $(tconcentric-det) --tag $(tconcentric-tag) --save  \
       --eye 0,5,0 \
       $* 
}

tconcentric-t()
{
    tconcentric-
    tconcentric-- --okg4 --compute $*
}
tconcentric-tt()
{
   # TODO: make these options the defaults
    tconcentric-t --bouncemax 15 --recordmax 16 --groupvel --finebndtex $* 
}

tconcentric-tt-dindex()
{
    tconcentric-tt \
         --dindex=2,4,5,6,7,9,10,11,13,14 $*

    #--dindex=999999,999997,999996,999995,999994,999993,999992,999991,999990,999989
    #--dindex=95324,166006,178463,206278,266703,304171,372458,384384,436024,471027,492290,500284,503639,527858,569752,667682,875192
}

tconcentric-tt-dbg()
{
    tconcentric-tt \
            --debugger $*

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

tconcentric-torchconfig()
{
    ## tconcentric traditionally uses a random point source from center point
    ## however whilst debugging using a laser source simplifies interpretation in tconcentric_distrib.py 
    local photons=$(tconcentric-photons)
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
    echo "$(join _ ${torch_config[@]})" 
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



tconcentric-args(){  echo  --tag $(tconcentric-tag) --det $(tconcentric-det) --src $(tconcentric-src) ; }
tconcentric-py(){    tconcentric.py  $(tconcentric-args) $* ; } 
tconcentric-i(){     ipython --profile=g4opticks -i $(which tconcentric.py) --  $(tconcentric-args) $* ; } 
tconcentric-d(){     ipython --profile=g4opticks -i $(which tconcentric_distrib.py) --  $(tconcentric-args) $* ; } 

tconcentric-rehist(){ tconcentric-d --noplot --rehist --sel 0:100  ; }

#tconcentric-sc(){   tconcentric-i --pfxseqhis .6d ; }
tconcentric-sc(){   tconcentric-i --pfxseqhis .6ccd ; }



