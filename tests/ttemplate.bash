ttemplate-source(){   echo $(opticks-home)/tests/ttemplate.bash ; }
ttemplate-vi(){       vi $(ttemplate-source) ; }
ttemplate-usage(){ cat << \EOU

ttemplate- : Integration Tests Template
================================================


`ttemplate-vi`
    edit the bash functions 

`ttemplate--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`ttemplate-- --compute`
    create Opticks geometry, simulates photons in compute mode, saves evt file

`ttemplate-- --tcfg4` 
    create Geant4 geometry, simulates photons, saves evt file

`ttemplate-- --tcfg4 --load`
    visualize the Geant4 simulated photon propagation 

`ttemplate-cf`
    compare Opticks and Geant4 material/flag sequence histories


`ttemplate-test`
    simulates with Opticks and Geant4 and compares the results 



EOU
}
ttemplate-env(){      olocal- ;  }
ttemplate-dir(){ echo $(opticks-home)/tests ; }
ttemplate-cd(){  cd $(ttemplate-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

ttemplate-tag(){ echo 10 ; }
ttemplate-det(){ echo PmtInBox ; }
ttemplate-src(){ echo torch ; }

ttemplate--(){
   type $FUNCNAME

    local msg="=== $FUNCNAME :"

    local cmdline=$*
    local tag=$(ttemplate-tag)
    local det=$(ttemplate-det)

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


ttemplate-args() {        echo  --tag $(ttemplate-tag) --det $(ttemplate-det) --src $(ttemplate-src) ; }
ttemplate-cf() {          ttemplate.py          $(ttemplate-args) ; } 
ttemplate-cf-distrib() {  ttemplate_distrib.py  $(ttemplate-args) ; } 

ttemplate-ana()
{
    ttemplate-cf
    ttemplate-cf-distrib
}

ttemplate-test()
{
    ttemplate--  --compute 
    ttemplate--  --tcfg4
    ttemplate-cf
    ttemplate-cf-distrib
}

ttemplate-viz-g4() { ttemplate-- --load --tcfg4 ; } 
ttemplate-viz() {    ttemplate-- --load ; } 




