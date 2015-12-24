# === func-gen- : opticksop/opop fgp opticksop/opop.bash fgn opop fgh opticksop
opop-src(){      echo opticksop/opop.bash ; }
opop-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opop-src)} ; }
opop-vi(){       vi $(opop-source) ; }
opop-usage(){ cat << EOU

Opticks Operations
====================

::

   opop-;opop-index --dbg

   opop-;opop-index -5 rainbow
   opop-;opop-index -6 rainbow

   opop-;opop-index -1 reflect
   opop-;opop-index -2 reflect


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
    local msg="=== $FUNCNAME : "
    local tag=${1:--5}
    local cat=${2:-rainbow}
    local typ=${3:-torch}

    local shdir=$(opop-index-path sh $tag $cat $typ)
    if [ -d "$shdir" ]; then 
        echo $msg index exists already tag $tag cat $cat typ $typ shdir $shdir
        return 
    else
        echo $msg index does not exist for tag $tag cat $cat typ $typ shdir $shdir
    fi

    local cmdline=$*
    local dbg=0
    if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
       dbg=1
    fi
    case $dbg in  
       0) $(opop-bin) --tag $tag --cat $cat   ;;
       1) lldb $(opop-bin) -- --tag $tag --cat $cat   ;;
    esac
}


opop-index-path(){
    local cmp=${1:-ps}
    local tag=${2:-5}
    local cat=${3:-rainbow}
    local typ=${4:-torch}
    local base=$LOCAL_BASE/env/opticks
    case $cmp in 
        ps|rs) echo $base/$cat/$cmp$typ/$tag.npy  ;;
        sh|sm) echo $base/$cat/$cmp$typ/$tag/     ;;
    esac 
}

opop-index-op(){
   local tag=-5
   local cat=rainbow
   local typ=torch
   local cmps="ps rs sh sm"
   local path 
   local cmp
   for cmp in $cmps ; do
       #echo $cmp $tag $cat $typ
       path=$(opop-index-path $cmp $tag $cat $typ)  
       echo $path
   done
}




