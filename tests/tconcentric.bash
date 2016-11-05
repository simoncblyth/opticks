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


tconcentric-photons(){
   local photons=1000000
   #local photons=100000
   #local photons=10000
   #local photons=100
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
       1|10|100|1000|10000) g4ppe=$photons ;;
    esac

    op.sh \
       --g4ppe $g4ppe \
       --test --testconfig "$(tconcentric-testconfig)" \
       --torch --torchconfig "$(tconcentric-torchconfig)" \
       --animtimemax 30 \
       --timemax 30 \
       --cat $(tconcentric-det) --tag $(tconcentric-tag) --save  \
       --eye 0.5,0.5,0.0 \
       $* 
}

tconcentric-t()
{
    tconcentric-
    tconcentric-- --okg4 --compute $*
}
tconcentric-tt()
{
    tconcentric-t --bouncemax 15 --recordmax 16 $* 
}
tconcentric-tt-pflags()
{
    tconcentric-tt \
          --oindex $(tconcentric-oindex 100) \
          --dindex='3352,12902,22877,23065,41882,60653,68073,69957,93373,114425,116759,119820,121160,128896,140920,144796,149640,155511,172178,173508,181946,197721,206106,218798,226414,229472,245012,246679,247048,250061,256632,273737,277009,278330,283792,284688,302431,302522,310912,312485,322121,327125,328934,344304,348955,363391,385856,398678,405719,413374,427982,435697,440670,470050,474196,477693,479219,479671,482244,482334,483690,493571,499519,510053,512631,520014,528665,537688,572302,580525,582218,592832,603216,605660,609385,613092,616980,632731,643197,647969,648445,651609,652951,659879,661157,663245,666346,667822,668744,673617,685642,688649,699598,700202,710936,728978,733667,742167,745397,764234,764506,772722,776790,785381,798323,799789,800795,801821,816920,817527,821113,840075,863428,872134,878479,879868,898266,900382,900808,905903,909591,911618,917897,919938,925473,929891,929984,961725,967547,976708,978573,994454'

}


tconcentric-tt-pflags-compare()
{
    tconcentric-tt \
          --oindex=63,115,124,200,225,270,307,338,342,423 \
          --dindex=3352,12902,22877,23065,41882,60653,68073,69957,93373,114425
}
tconcentric-tt-pflags-bad()
{
    tconcentric-tt \
         --dindex=93373,173508,302431 
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




