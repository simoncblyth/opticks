om-source(){ echo $BASH_SOURCE ; }
om-vi(){ vi $(om-source) ; }
om-env(){  olocal- ; opticks- ; }

om-usage(){ cat << EOU

OM : Opticks Mimimal Approach to Configuring and Building
===========================================================

The below functions configure, build and test the Opticks subprojects.
They mostly take no arguments, using the invoking directory to determine what to do. 
When run from top level source or build dirs they loop over all the constituents. 


FUNCTIONS TAKING INVOKING DIRECTORY AS "ARGUMENT"
--------------------------------------------------

om-configure
   runs cmake to configure one/all subprojects 

om-build
   runs cmake to build and install one/all subprojects 

om-test
   runs ctest for one/all subprojects

om-pd
   cd from a source tree directory to the corresponding 
   directory in the build tree and vice versa


FUNCTIONS INSENSITIVE TO INVOKING DIRECTORY
-------------------------------------------------

om-visit-all
om-configure-all
om-build-all
om-test-all

om-testlog
    parse test logs from previous test runs and present summary totals 


    
EOU
} 

om-subnames(){ cat << EOS
okconf
sysrap
boostrap
npy
optickscore
ggeo 
assimprap
openmeshrap 
opticksgeo 
cudarap
thrustrap
optixrap
okop
oglrap  
opticksgl
ok
cfg4
okg4
EOS
}

om-sdir(){  echo $(opticks-home)/$1 ; }
om-bdir(){  echo $(opticks-prefix)/build/$1 ; }

om-visit-all(){     om-all ${FUNCNAME/-all} ; }
om-configure-all(){ om-all ${FUNCNAME/-all} ; }
om-build-all(){     om-all ${FUNCNAME/-all} ; }
om-test-all(){      om-all ${FUNCNAME/-all} ; om-testlog ; }
om-echo-all(){      om-all ${FUNCNAME/-all} ; }

om-testlog(){      CTestLog.py $(om-bdir)  ; }

om-all()
{
    local iwd=$(pwd)
    local func=$1
    local msg="=== $FUNCNAME $func :"
    local name
    om-subnames | while read name 
    do 
        local bdir=$(om-bdir $name)
        cd $bdir
        $func
    done 
    cd $iwd
}


om-echo(){ echo $(pwd) ; }

om-visit()
{
    local msg="=== $FUNCNAME :"
    local iwd=$(pwd)
    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" ]; then
        om-visit-all
    else
        local name=$(basename $iwd)
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        cd $bdir
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
    fi
}

om-configure()
{
    local msg="=== $FUNCNAME :"
    local iwd=$(pwd)
    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" ]; then
        om-configure-all
    else
        local name=$(basename $iwd)
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        cd $bdir
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
     
        if [ "$name" == "okconf" ]; then     
            cmake $sdir \
               -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
               -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
               -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
               -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
               -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
               -DCOMPUTE_CAPABILITY=$(opticks-compute-capability)
        else
            cmake $sdir \
               -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
               -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
               -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
               -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

        fi
    fi
}

om-build()
{   
    local iwd=$(pwd)
    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" ]; then
        om-build-all
    else
        local msg="=== $FUNCNAME :"
        local name=$(basename $iwd)
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
        cd $bdir
        cmake --build .  --target all
        [ "$(uname)" == "Darwin" ] && echo $msg kludge sleep 2s && sleep 2  
        cmake --build .  --target install
    fi 
    cd $iwd
}

om-test()
{
    local iwd=$(pwd)
    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" ]; then
        om-test-all
    else
        local msg="=== $FUNCNAME :"
        local name=$(basename $iwd)
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
        cd $bdir
        local log=ctest.log
        date          | tee $log
        ctest $* --interactive-debug-mode 0 2>&1 | tee -a $log
        date          | tee -a $log
    fi
    cd $iwd
}

om-pdir() 
{ 
    local here=$(pwd -P);
    local stop=$(om-sdir);
    local btop=$(om-bdir);
    stop=${stop%/}  # remove trailing slash 
    btop=${btop%/}   
    case $here in 
        $stop)  echo $btop ;;
        $btop)  echo $stop ;;
        $stop*) echo $btop/${here/$stop\/} ;;
        $btop*) echo $stop/${here/$btop\/} ;;
             *) echo "" ;;
    esac
    return 0 
}

om-pd()
{
    local msg="=== $FUNCNAME :"
    local iwd=$(pwd -P)
    local pdir=$(om-pdir);
    [ -z "$pdir" ] && echo pwd $iwd is not inside Opticks source tree or its counterpart build tree && return 1;
    [ -n "$pdir" ] && cd $pdir
    #printf "%s %-60s to %-60s \n"  "$msg" $name $iwd $pdir
    pwd
}

