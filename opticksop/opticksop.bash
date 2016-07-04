# === func-gen- : opticksop/opticksop fgp opticksop/opticksop.bash fgn opticksop fgh opticksop
opticksop-rel(){      echo opticksop ; }
opticksop-src(){      echo opticksop/opticksop.bash ; }
opticksop-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(opticksop-src)} ; }
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



Still There 
----------------------------

Flaky issue that tends to occur at first run::

    2016-06-27 15:01:55.616 INFO  [12827887] [OpEngine::prepareOptiX@129] OpEngine::prepareOptiX (OGeo)
    2016-06-27 15:01:55.617 INFO  [12827887] [OGeo::convert@166] OGeo::convert nmm 2
    GGeoViewTest(45352,0x7fff74d63310) malloc: *** error for object 0x7ff5a895d408: incorrect checksum for freed object - object was probably modified after being freed.
    *** set a breakpoint in malloc_error_break to debug
    /Users/blyth/env/bin/op.sh: line 374: 45352 Abort trap: 6           /usr/local/opticks/lib/GGeoViewTest


Familiar Issue
----------------

::

        recsel_attr 0/ 1 vnpy       rsel   5000000 npy 500000,10,1,4 npy.hasData 0
    2016-06-25 14:37:38.597 INFO  [12495653] [App::prepareOptiX@961] App::prepareOptiX create OpEngine 
    2016-06-25 14:37:38.597 INFO  [12495653] [Timer::operator@38] OpEngine:: START
    GGeoViewTest(69035,0x7fff74d63310) malloc: *** error for object 0x109e11208: incorrect checksum for freed object - object was probably modified after being freed.
    *** set a breakpoint in malloc_error_break to debug
    Process 69035 stopped
    (lldb) bt
    * thread #1: tid = 0xbeab25, 0x00007fff8ea02866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8ea02866 libsystem_kernel.dylib`__pthread_kill + 10
        ...
        frame #13: 0x00000001025df04a liboptix.1.dylib`rtBufferMap + 122
        frame #14: 0x000000010355e2cf libOptiXRap.dylib`optix::BufferObj::map(this=0x000000011ec51f00) + 47 at optixpp_namespace.h:3755
        frame #15: 0x0000000103564875 libOptiXRap.dylib`OPropertyLib::makeTexture(this=0x000000011ec51f80, buffer=0x0000000111d21e10, format=RT_FORMAT_FLOAT, nx=1024, ny=1, empty=false) + 805 at OPropertyLib.cc:44
        frame #16: 0x00000001035672ea libOptiXRap.dylib`OSourceLib::makeSourceTexture(this=0x000000011ec51f80, buf=0x0000000111d21e10) + 762 at OSourceLib.cc:40
        frame #17: 0x0000000103566fd4 libOptiXRap.dylib`OSourceLib::convert(this=0x000000011ec51f80) + 276 at OSourceLib.cc:18
        frame #18: 0x00000001044009ce libOpticksOp.dylib`OpEngine::prepareOptiX(this=0x0000000106553d10) + 3950 at OpEngine.cc:124
        frame #19: 0x0000000104532e96 libGGeoView.dylib`App::prepareOptiX(this=0x00007fff5fbfe350) + 326 at App.cc:963
        frame #20: 0x000000010000c38f GGeoViewTest`main(argc=2, argv=0x00007fff5fbfe4c8) + 1071 at GGeoViewTest.cc:70
        frame #21: 0x00007fff89e755fd libdyld.dylib`start + 1
        frame #22: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 


::

    (lldb) f 16
    frame #16: 0x0000000102bb02ea libOptiXRap.dylib`OSourceLib::makeSourceTexture(this=0x000000011fd0d480, buf=0x0000000112cdbd70) + 762 at OSourceLib.cc:40
       37   
       38       float step = 1.f/float(nx) ;
       39       optix::float4 domain = optix::make_float4(0.f , 1.f, step, 0.f );
    -> 40       optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT, nx, ny);
       41   
       42       m_context["source_texture"]->setTextureSampler(tex);
       43       m_context["source_domain"]->setFloat(domain);
    (lldb) p nx
    (unsigned int) $1 = 1024
    (lldb) p ny
    (unsigned int) $2 = 1

    (lldb) f 15 
    frame #15: 0x0000000102bad875 libOptiXRap.dylib`OPropertyLib::makeTexture(this=0x000000011fd0d480, buffer=0x0000000112cdbd70, format=RT_FORMAT_FLOAT, nx=1024, ny=1, empty=false) + 805 at OPropertyLib.cc:44
       41          //    
       42          //
       43   
    -> 44           memcpy( optixBuffer->map(), buffer->getBytes(), numBytes );
       45           optixBuffer->unmap(); 
       46       }
       47   
    (lldb) p numBytes
    (unsigned int) $3 = 4096

    (lldb) f 14
    frame #14: 0x0000000102ba72cf libOptiXRap.dylib`optix::BufferObj::map(this=0x000000011fd0d400) + 47 at optixpp_namespace.h:3755
       3752   inline void* BufferObj::map()
       3753   {
       3754     void* result;
    -> 3755     checkError( rtBufferMap( m_buffer, &result ) );
       3756     return result;
       3757   }
       3758 





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
    olocal-
    optix-
    optix-export
    opticks-
}

opticksop-sdir(){ echo $(opticks-home)/opticksop ; }
opticksop-tdir(){ echo $(opticks-home)/opticksop/tests ; }
opticksop-idir(){ echo $(opticks-idir); }
opticksop-bdir(){ echo $(opticks-bdir)/$(opticksop-rel) ; }

opticksop-bin(){  echo $(opticksop-idir)/bin/${1:-OpIndexerTest} ; }

opticksop-cd(){   cd $(opticksop-sdir); }
opticksop-scd(){  cd $(opticksop-sdir); }
opticksop-tcd(){  cd $(opticksop-tdir); }
opticksop-icd(){  cd $(opticksop-idir); }
opticksop-bcd(){  cd $(opticksop-bdir); }

opticksop-name(){ echo OpticksOp ; }
opticksop-tag(){  echo OKOP ; }

opticksop-wipe(){ local bdir=$(opticksop-bdir) ; rm -rf $bdir ; } 
opticksop--(){                   opticks-- $(opticksop-bdir) ; } 
opticksop-ctest(){               opticks-ctest $(opticksop-bdir) $* ; } 
opticksop-genproj() { opticksop-scd ; opticks-genproj $(opticksop-name) $(opticksop-tag) ; } 
opticksop-gentest() { opticksop-tcd ; opticks-gentest ${1:-OExample} $(opticksop-tag) ; } 
opticksop-txt(){ vi $(opticksop-sdir)/CMakeLists.txt $(opticksop-tdir)/CMakeLists.txt ; } 







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


