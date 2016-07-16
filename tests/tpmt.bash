tpmt-source(){   echo $(opticks-home)/tests/tpmt.bash ; }
tpmt-vi(){       vi $(tpmt-source) ; }
tpmt-usage(){ cat << \EOU

TPMT : Opticks Simulation PMT Tests 
================================================



`tpmt-vi`
    edit the bash functions 

`tpmt--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`tpmt-- --compute`
    create Opticks geometry, simulates photons in compute mode, saves evt file

`tpmt-- --tcfg4` 
    create Geant4 geometry, simulates photons, saves evt file

`tpmt-- --tcfg4 --load`
    visualize the Geant4 simulated photon propagation 

`tpmt-cf`
    compare Opticks and Geant4 material/flag sequence histories

`tpmt-cf-distrib`
    compare Opticks and Geant4 photon step distributions

`tpmt-test`
    simulates with Opticks and Geant4 and compares the results 



`tpmt-alt`
    visualize alternate geometry with the PMT inside a sphere of Mineral Oil 
    


EOU
}
tpmt-env(){      olocal- ;  }
tpmt-dir(){ echo $(opticks-home)/tests ; }
tpmt-cd(){  cd $(tpmt-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }


tpmt-tag(){ echo 10 ; }
tpmt-det(){ echo PmtInBox ; }
tpmt-src(){ echo torch ; }

tpmt--(){
   type $FUNCNAME

    local msg="=== $FUNCNAME :"

    local cmdline=$*
    local tag=$(tpmt-tag)
    local det=$(tpmt-det)

    #local photons=500000
    local photons=100000

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


    local groupvelkludge=0
    local testverbosity=1
    local test_config=(
                 mode=PmtInBox
                 pmtpath=$OPTICKS_INSTALL_PREFIX/opticksdata/export/dpib/GMergedMesh/0
                 control=$testverbosity,0,0,0
                 analytic=1
                 groupvel=$groupvelkludge
                 shape=box
                 boundary=Rock/NONE/perfectAbsorbSurface/MineralOil
                 parameters=0,0,0,300
                   ) 

    if [ "${cmdline/--tcfg4}" != "${cmdline}" ]; then
        tag=-$tag  
    fi 


   op.sh \
       --test --testconfig "$(join _ ${test_config[@]})" \
       --torch --torchconfig "$(join _ ${torch_config[@]})" \
       --timemax 10 \
       --animtimemax 10 \
       --cat $det --tag $tag --save \
       --eye 0.0,-0.5,0.0 \
       --geocenter \
       $* 

}


tpmt-args() {        echo  --tag $(tpmt-tag) --det $(tpmt-det) --src $(tpmt-src) ; }
tpmt-cf() {         tpmt.py          $(tpmt-args) ; } 
tpmt-cf-distrib() { tpmt_distrib.py  $(tpmt-args) ; } 

tpmt-ana()
{
    tpmt-cf
    tpmt-cf-distrib
}

tpmt-test()
{
    tpmt--  --compute 
    tpmt--  --tcfg4
    tpmt-cf
    tpmt-cf-distrib
}

tpmt-viz-g4() { tpmt-- --load --tcfg4 ; } 
tpmt-viz() {    tpmt-- --load ; } 





tpmt-alt(){
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



