tpmt-source(){   echo $(opticks-home)/tests/tpmt.bash ; }
tpmt-vi(){       vi $(tpmt-source) ; }
tpmt-usage(){ cat << \EOU

tpmt- : Opticks Simulation PMT Tests 
================================================

`tpmt-vi`
    edit the bash functions 

`tpmt--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`tpmt-- --compute`
    create Opticks geometry, simulates photons in compute mode, saves evt file

`tpmt-- --okg4 --compute` 
    creates Geant4 geometry, performs both Opticks and Geant4 torch photon propagations, 
    saves both G4 and OK evt to file following negated G4 convention

`tpmt-- --vizg4 --load`
    visualize the Geant4 simulated photon propagation 

`tpmt-cf`
    compare Opticks and Geant4 material/flag sequence histories

`tpmt-cf-distrib`
    compare Opticks and Geant4 photon step distributions

`tpmt-test`
    simulates with Opticks and Geant4 and compares the results 



`tpmt-alt`
    visualize alternate geometry with the PMT inside a sphere of Mineral Oil 
    


`tpmt-skimmer`

::

    In [1]: run tpmt_skimmer.py --tag 10
    tpmt_skimmer.py --tag 10
    A(Op) PmtInBox/torch/ 10 : TO BT BR BR BT SA 
      0 z:    300.000    300.000    300.000   r:     97.134     99.863     98.499   t:      0.100      0.100      0.100   smry m1/m2   4/ 14 MO/Py  -28 ( 27)  13:TO  
      1 z:     63.787     74.133     68.960   r:     97.134     99.863     98.499   t:      1.217      1.268      1.243   smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      2 z:     43.983     54.228     49.106   r:     98.888    100.928     99.908   t:      1.315      1.364      1.339   smry m1/m2  14/ 11 Py/OV -125 (124)  11:BR  
      3 z:     33.546     37.886     35.716   r:     92.439     93.983     93.211   t:      1.401      1.424      1.413   smry m1/m2  14/  4 Py/MO   28 ( 27)  11:BR  
      4 z:     18.641     21.470     20.055   r:     88.695     90.252     89.473   t:      1.483      1.498      1.491   smry m1/m2   4/ 12 MO/Rk  124 (123)  12:BT  
      5 z:   -300.000   -300.000   -300.000   r:     23.101     48.215     35.658   t:      3.087      3.107      3.097   smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    B(G4) PmtInBox/torch/-10 : TO BT BR BR BT SA 
      0 z:    300.000    300.000    300.000   r:     97.103     99.947     98.525   t:      0.100      0.100      0.100   smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z:     63.366     74.224     68.795   r:     97.103     99.947     98.525   t:      1.291      1.349      1.320   smry m1/m2  14/  0 Py/?0?    0 ( -1)  12:BT  
      2 z:     43.873     54.576     49.225   r:     98.857    100.971     99.914   t:      1.392      1.439      1.415   smry m1/m2  14/  0 Py/?0?    0 ( -1)  11:BR  
      3 z:     33.500     38.609     36.055   r:     92.415     94.219     93.317   t:      1.476      1.498      1.487   smry m1/m2  14/  0 Py/?0?    0 ( -1)  11:BR  
      4 z:     18.595     21.708     20.151   r:     88.671     90.372     89.522   t:      1.558      1.576      1.567   smry m1/m2   4/  0 MO/?0?    0 ( -1)  12:BT  
      5 z:   -300.000   -300.000   -300.000   r:     22.626     48.456     35.541   t:      3.268      3.293      3.280   smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  



ANALYSIS EXERCISE
--------------------

* run *tpmt.py* from ipython

* run *tpmt_distrib.py* from ipython

* use the Evt class to select a class of photons and make plots of 
  quantities such as position and time 





EOU
}


tpmt-notes(){ cat << \EON

* failed to load the old style analytic PMT when using apmtidx=1, 
  switching to apmtidx=0 succeeds to load

* PMT raytrace is noshow

* bizarre raytrace box distortion when viewed from inside, 
  from some positions : probably a bad OptiX bbox clipping the
  actual box


Issue may be connected to this cryptic note::

    # AnalyticPMTIndex, apmtindex -> analytic_version
    #
    #    >1 : living without bbox
    # 

 

EON

}



tpmt-env(){      olocal- ;  }
tpmt-dir(){ echo $(opticks-home)/tests ; }
tpmt-cd(){  cd $(tpmt-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }


tpmt-tag(){ echo 10 ; }
tpmt-det(){ echo PmtInBox ; }
tpmt-src(){ echo torch ; }


tpmt-pmtpath(){ echo $OPTICKS_INSTALL_PREFIX/opticksdata/export/dpib/GMergedMesh/0 ; }

tpmt-testconfig()
{
    local material=MineralOil
    #local material=GdDopedLS

    local testverbosity=1
    local groupvelkludge=0
   # groupvel=$groupvelkludge   no longer supported/needed ? 

    local test_config=(
                 mode=PmtInBox
                 pmtpath=$(tpmt-pmtpath)
                 control=$testverbosity,0,0,0
                 analytic=1
                 node=box    parameters=0,0,0,400   boundary=Rock/NONE/perfectAbsorbSurface/$material
                   ) 

    echo "$(join _ ${test_config[@]})" 
}



tpmt-torchconfig()
{
    #local photons=10000
    local photons=500000
    #local photons=100000

    local zenith=0,1
    #local typ=disclin
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

    echo "$(join _ ${torch_config[@]})" 
}



tpmt--(){
   type $FUNCNAME

    local msg="=== $FUNCNAME :"

    local cmdline=$*
    local tag=$(tpmt-tag)

    [ -z "$OPTICKS_INSTALL_PREFIX" ] && echo missing envvar OPTICKS_INSTALL_PREFIX && return 

    local anakey
    if [ "${cmdline/--okg4}" != "${cmdline}" ]; then
        anakey=tpmt   ## compare OK and G4 evt histories
    else
        anakey=tevt    ## just dump OK history table
    fi 


    #local apmtidx=0  
    local apmtidx=2 

    ## 1 -> non-existing /usr/local/opticks/opticksdata/export/DayaBay/GPmt/1/
    
    # AnalyticPMTIndex, apmtindex -> analytic_version
    #
    #    >1 : living without bbox
    # 


   op.sh \
       --anakey $anakey \
       --save \
       --test --testconfig "$(tpmt-testconfig)" \
       --torch --torchconfig "$(tpmt-torchconfig)" \
       --cat $(tpmt-det) \
       --tag $tag \
       --timemax 10 \
       --animtimemax 10 \
       --eye 0.0,-0.5,0.0 \
       --geocenter \
       --apmtidx $apmtidx \
       --dbganalytic \
       --rendermode +global,+axis \
       $* 

}


tpmt-args() {        echo  --tag $(tpmt-tag) --det $(tpmt-det) --src $(tpmt-src) ; }
tpmt-cf() {         tpmt.py          $(tpmt-args) ; } 
tpmt-cf-distrib() { tpmt_distrib.py  $(tpmt-args) ; } 
tpmt-skimmer() {    tpmt_skimmer.py  $(tpmt-args) ; } 


tpmt-ana()
{
    tpmt-cf
    tpmt-cf-distrib
}

tpmt-t()
{
    tpmt-
    tpmt-- --okg4 --compute
}

tpmt-v() {    tpmt-- --load $* ; } 
tpmt-vg4() {  tpmt-- --load --vizg4 ; }   ## --vizg4 -> OpticksRun::getEvent returns G4 evt 

tpmt-alt(){
   local test_config=(
                 mode=PmtInBox
                 analytic=1

                 node=sphere
                 boundary=Rock//perfectAbsorbSurface/MineralOil
                 parameters=-1,1,0,300
                   ) 

   op.sh --tracer \
          --test --testconfig "$(join _ ${test_config[@]})" \
          --eye 0.5,0.5,0.0 \
          --rendermode +global,+axis \
           $*  
}


