tbox-source(){   echo $(opticks-home)/tests/tbox.bash ; }
tbox-vi(){       vi $(tbox-source) ; }
tbox-usage(){ cat << \EOU

tbox- : Pyrex Cube inside Mineral Oil Cube Test 
==================================================


`tbox-vi`
    edit the bash functions 

`tbox--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`tbox-- --compute`
    create Opticks geometry, simulates photons in compute mode, saves evt file

`tbox-- --tcfg4` 
    create Geant4 geometry, simulates photons, saves evt file

`tbox-- --tcfg4 --load`
    visualize the Geant4 simulated photon propagation 

`tbox-cf`
    compare Opticks and Geant4 material/flag sequence histories


`tbox-test`
    simulates with Opticks and Geant4 and compares the results 



EOU
}
tbox-env(){      olocal- ;  }
tbox-dir(){ echo $(opticks-home)/tests ; }
tbox-cd(){  cd $(tbox-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tbox-tag(){ echo 1 ; }
tbox-det(){ echo BoxInBox ; }
tbox-src(){ echo torch ; }

tbox--(){
    local cmdline=$*
    local tag=$(tbox-tag)
    if [ "${cmdline/--tcfg4}" != "${cmdline}" ]; then
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
       --cat $(tbox-det) --tag $tag --save  \
       --eye 0.5,0.5,0.0 \
       $* 
}
tbox-args(){  echo  --tag $(tbox-tag) --det $(tbox-det) --src $(tbox-src) ; }
tbox-py(){    tbox.py  $(tbox-args) $* ; } 
tbox-test()
{
    tbox-- --compute
    tbox-- --tcfg4
    tbox-py 
}


