# === func-gen- : opticksop/opop fgp opticksop/opop.bash fgn opop fgh opticksop
opop-src(){      echo opticksop/opop.bash ; }
opop-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opop-src)} ; }
opop-vi(){       vi $(opop-source) ; }
opop-usage(){ cat << EOU

Opticks Operations
====================

::

   opop-;opop-index --dbg


EOU
}

opop-sdir(){ echo $(env-home)/opticksop ; }
opop-idir(){ echo $(local-base)/env/opticksop ; }
opop-bdir(){ echo $(opop-idir).build ; }
opop-bin(){  echo $(opop-idir)/bin/${1:-OpIndexerTest} ; }

opop-scd(){  cd $(opop-sdir); }
opop-cd(){  cd $(opop-sdir); }

opop-icd(){  cd $(opop-idir); }
opop-bcd(){  cd $(opop-bdir); }
opop-name(){ echo OpticksOp ; }

opop-wipe(){
   local bdir=$(opop-bdir)
   rm -rf $bdir
}

opop-env(){
    elocal-
    optix-
    optix-export
}

opop-options(){
   echo -n
}

opop-cmake(){
   local iwd=$PWD

   local bdir=$(opop-bdir)
   mkdir -p $bdir

   opop-bcd

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opop-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(opop-options) \
       $(opop-sdir)


   cd $iwd
}

opop-make(){
   local iwd=$PWD

   opop-bcd
   make $*

   cd $iwd
}

opop-install(){
   opop-make install
}


opop--()
{
    opop-wipe
    opop-cmake
    opop-make
    opop-install
}

opop-index(){

    local cmdline=$*
    local dbg=0
    if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
       dbg=1
    fi
    case $dbg in  
       0) $(opop-bin) --tag -5 --cat rainbow   ;;
       1) lldb $(opop-bin) -- --tag -5 --cat rainbow   ;;
    esac
}

