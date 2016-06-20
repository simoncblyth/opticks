# === func-gen- : opticksop/opticksop fgp opticksop/opticksop.bash fgn opticksop fgh opticksop
opticksop-rel(){      echo opticksop ; }
opticksop-src(){      echo opticksop/opticksop.bash ; }
opticksop-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticksop-src)} ; }
opticksop-vi(){       vi $(opticksop-source) ; }
opticksop-usage(){ cat << EOU

Opticks Operations
====================

::

   opticksop-;opticksop-index --dbg

   opticksop-;opticksop-index -5 rainbow
   opticksop-;opticksop-index -6 rainbow

   opticksop-;opticksop-index -1 reflect
   opticksop-;opticksop-index -2 reflect



Classes
---------

OpEngine
    Very high level control:: 

       void prepareOptiX();
       void setEvent(OpticksEvent* evt);
       void preparePropagator();
       void seedPhotonsFromGensteps();
       void initRecords();
       void propagate();
       void saveEvt();
       void indexSequence();
       void cleanup();

OpIndexer
    Very high level control of Thrust indexer

OpSeeder
    Seeding distributes genstep indices into the photon buffer
    according to the known number of photons generated for each genstep.
    This is accomplished entirely on the GPU using Thrust. 

OpZeroer
    Scrubbing GPU buffers.

OpIndexerApp
    Used by standalone evt loading and indexing app 



EOU
}



opticksop-env(){
    elocal-
    optix-
    optix-export
    opticks-
}

opticksop-sdir(){ echo $(env-home)/opticksop ; }
opticksop-idir(){ echo $(opticks-idir); }
opticksop-bdir(){ echo $(opticks-bdir)/$(opticksop-rel) ; }


opticksop-bin(){  echo $(opticksop-idir)/bin/${1:-OpIndexerTest} ; }

opticksop-scd(){  cd $(opticksop-sdir); }
opticksop-cd(){  cd $(opticksop-sdir); }

opticksop-icd(){  cd $(opticksop-idir); }
opticksop-bcd(){  cd $(opticksop-bdir); }
opticksop-name(){ echo OpticksOp ; }

opticksop-wipe(){
   local bdir=$(opticksop-bdir)
   rm -rf $bdir
}

opticksop-options(){
   echo -n
}

opticksop-cmake-deprecated(){
   local iwd=$PWD

   local bdir=$(opticksop-bdir)
   mkdir -p $bdir

   opticksop-bcd

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticksop-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(opticksop-options) \
       $(opticksop-sdir)


   cd $iwd
}

opticksop-make(){
   local iwd=$PWD

   opticksop-bcd
   make $*

   cd $iwd
}

opticksop-install(){
   opticksop-make install
}

opticksop-full()
{
   opticksop-make clean
   opticksop-make
   opticksop-install
}

opticksop-config(){ echo Debug ; }
opticksop--()
{
   local iwd=$PWD;
   opticksop-bcd;
   cmake --build . --config $(opticksop-config) --target ${1:-install};
   cd $iwd
}

opticksop-index(){
    local msg="=== $FUNCNAME : "
    local tag=${1:--5}
    local cat=${2:-rainbow}
    local typ=${3:-torch}

    local shdir=$(opticksop-index-path sh $tag $cat $typ)
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
       0) $(opticksop-bin) --tag $tag --cat $cat   ;;
       1) lldb $(opticksop-bin) -- --tag $tag --cat $cat   ;;
    esac
}


opticksop-index-path(){
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

opticksop-index-op(){
   local tag=-5
   local cat=rainbow
   local typ=torch
   local cmps="ps rs sh sm"
   local path 
   local cmp
   for cmp in $cmps ; do
       #echo $cmp $tag $cat $typ
       path=$(opticksop-index-path $cmp $tag $cat $typ)  
       echo $path
   done
}




