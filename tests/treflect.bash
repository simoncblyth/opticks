treflect-source(){   echo $(opticks-home)/tests/treflect.bash ; }
treflect-vi(){       vi $(treflect-source) ; }
treflect-usage(){ cat << \EOU

treflect- : Fresnel reflection vs incident angle check  
==========================================================

A hemi-spherical S/P polarized light source focussed on cube face
is used to check the amount of reflection as a function of incident
angle matches expectation of the Fresnel formula. 


`treflect-vi`
    edit the bash functions 

`treflect--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 


TODO:

* calc chi2 against analytic S and P : so can check agreement automatically 
  


EOU
}
treflect-env(){      olocal- ;  }
treflect-dir(){ echo $(opticks-home)/tests ; }
treflect-cd(){  cd $(treflect-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

treflect-det(){ echo reflect ; }
treflect-src(){ echo torch ; }
treflect-stag(){ echo 1 ; }
treflect-ptag(){ echo 2 ; }

treflect--()
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

    case $pol in  
        s) tag=$(treflect-stag) ;;
        p) tag=$(treflect-ptag) ;;
    esac

    if [ "${cmdline/--tcfg4}" != "${cmdline}" ]; then
        tag=-$tag  
    fi 

    echo  pol $pol tag $tag


    local photons=1000000

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
            --save --tag $tag --cat $(treflect-det)
}


treflect-args(){ echo --stag $(treflect-stag) --ptag $(treflect-ptag) --det $(treflect-det) --src $(treflect-src) ; }
treflect-py(){   treflect.py $(treflect-args) $* ; } 
treflect-ipy(){  cat << EOI
# copy/paste below into ipython from ana directory
run treflect.py $(treflect-args)  
EOI
}

treflect-sp()
{
    treflect--  --spol $*
    treflect--  --ppol $*

}
treflect-s()
{
    treflect--  --spol $*
    treflect--  --spol --tcfg4
}
treflect-p()
{
    treflect--  --ppol $*
    treflect--  --ppol --tcfg4
}
treflect-test()
{
    treflect-s --compute
    treflect-p --compute
}





