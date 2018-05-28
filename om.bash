om-source(){ echo $BASH_SOURCE ; }
om-vi(){ vi $(om-source) ; }
om-env(){  olocal- ; opticks- ; }

om-usage(){ cat << EOU

OM : Opticks Mimimal Approach to Configuring and Building
===========================================================

The below functions configure, build and test the Opticks subprojects
in the appropriate dependency order.


SUBPROJ FUNCTIONS 
-----------------

om-subs
   list subprojects in dependency order,
   an argument can be used to select a
   starting subproject, eg "cudarap:" starts from there or ":" 
   starts from the current directory

   Such arguments can be used for all the subproj functions

om-conf
   runs cmake to configure one/all/range of subprojects 

om-make
   builds and install one/all/range of subprojects 

om-test
   runs ctest for one/all/range of subprojects

om-visit 
   debugging
 
om-echo
   debugging
 


OTHER FUNCTIONS
-----------------

om-pd
   cd from a source tree directory to the corresponding 
   directory in the build tree and vice versa


FUNCTIONS INSENSITIVE TO INVOKING DIRECTORY
-------------------------------------------------

om-visit-all
om-conf-all
om-make-all
om-test-all

om-testlog
    parse test logs from previous test runs and present summary totals 


    
EOU
} 

om-subs-(){ cat << EOS
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

om-subs(){
  local arg=$1

  local iwd=$(pwd)
  local name=$(basename $iwd)

  if [ -z "$arg" ]; then 
      om-subs-
  else
      [ "$arg" == ":" ] && arg="${name}:"
      local sel=0 
      local sub
      om-subs- | while read sub ; do
         [ "${sub}:" == "$arg" ] && sel=1 
         [ "$sel" == "1" ] && echo $sub
      done 
  fi
}


om-sdir(){  echo $(opticks-home)/$1 ; }
om-bdir(){  echo $(opticks-prefix)/build/$1 ; }

om-visit-all(){     om-all ${FUNCNAME/-all} $* ; }
om-conf-all(){      om-all ${FUNCNAME/-all} $* ; }
om-make-all(){      om-all ${FUNCNAME/-all} $* ; }
om-test-all(){      om-all ${FUNCNAME/-all} $* ; om-testlog ; }
om-echo-all(){      om-all ${FUNCNAME/-all} $* ; }

om-testlog(){      CTestLog.py $(om-bdir)  ; }

om-all()
{
    local rc
    local iwd=$(pwd)
    local func=$1
    shift
    local msg="=== $FUNCNAME $func :"
    local name
    om-subs $* | while read name 
    do 
        local bdir=$(om-bdir $name)
        cd $bdir
        $func
        rc=$?
        [ "$rc" != "0" ] && echo $msg ERROR bdir $bdir : non-zero rc $rc && return $rc
    done 
    cd $iwd
    return $rc
}


om-echo(){ echo $(pwd) ; }

om-visit()
{
    local arg=$1  # not normally used
    local msg="=== $FUNCNAME :"
    local iwd=$(pwd)

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o "${arg/:}" != "$arg" ]; then

        om-visit-all $arg

    else
        local name=$(basename $iwd)
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        cd $bdir
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
    fi
}

om-conf()
{
    local rc 
    local arg=$1  # not normally used
    local iwd=$(pwd)
    local msg="=== $FUNCNAME :"

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o  "${arg/:}" != "$arg" ]; then

        om-conf-all $arg 

    else
        local name=$(basename $iwd)
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)

        if [ "$arg" == "clean" ]; then
             echo $msg removed bdir $bdir as directed by clean argument
             rm -rf $bdir
        fi 

        if [ ! -d "$bdir" ]; then
             echo $msg bdir $bdir does not exist : creating it
             mkdir -p $bdir
        fi 

        cd $bdir
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir

        #echo sdir $sdir
        #echo bdir $bdir
        #echo pwd $(pwd)
    
        if [ "$name" == "okconf" ]; then     
            cmake $sdir \
               -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
               -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
               -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
               -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
               -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
               -DCOMPUTE_CAPABILITY=$(opticks-compute-capability)
            rc=$?
        else
            cmake $sdir \
               -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
               -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
               -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
               -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

            rc=$?
        fi
    fi

    if [ "$rc" != "0" ]; then
       echo $msg non-zero rc $rc 
    fi 
    return $rc 
}

om-make()
{   
    local rc=0
    local arg=$1  # not normally used
    local iwd=$(pwd)

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o "${arg/:}" != "$arg" ]; then

        om-make-all $arg

    else
        local msg="=== $FUNCNAME :"
        local name=$(basename $iwd)
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
        cd $bdir
        cmake --build .  --target all
        rc=$?
        [ "$rc" != "0" ] && return $rc
        [ "$(uname)" == "Darwin" ] && echo $msg kludge sleep 2s && sleep 2  
        cmake --build .  --target install
        rc=$?
    fi 
    cd $iwd
    return $rc
}

om-test()
{
    local iwd=$(pwd)
    local name=$(basename $iwd)

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o "${arg/:}" != "$arg" ]; then

        om-test-all $arg

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

