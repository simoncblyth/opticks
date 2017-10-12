# === func-gen- : opticksop/opticksop fgp opticksop/opticksop.bash fgn opticksop fgh opticksop
okop-rel(){      echo okop ; }
okop-src(){      echo okop/okop.bash ; }
okop-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(okop-src)} ; }
okop-vi(){       vi $(okop-source) ; }
okop-usage(){ cat << EOU

Opticks Operations
====================

::

   okop-;okop-index --dbg

   okop-;okop-index -5 rainbow
   okop-;okop-index -6 rainbow

   okop-;okop-index -1 reflect
   okop-;okop-index -2 reflect



Zeroing With OpenGL ?
------------------------

* https://www.opengl.org/sdk/docs/man/html/glClearBufferSubData.xhtml



SDUGPU
--------

::

	2017-09-15 15:12:06.505 INFO  [108721] [OContext::createProgram@190] OContext::createProgram START  filename solve_callable.cu.ptx progname SolveCubicCallable
	2017-09-15 15:12:06.505 VERB  [108721] [OConfig::createProgram@65] OConfig::createProgram path /usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx
	2017-09-15 15:12:06.505 DEBUG [108721] [OConfig::createProgram@71] OConfig::createProgram /usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx:SolveCubicCallable
	terminate called after throwing an instance of 'optix::Exception'
	  what():  Parse error (Details: Function "RTresult _rtProgramCreateFromPTXFile(RTcontext, const char*, const char*, RTprogram_api**)" caught exception: /usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx: error: Failed to parse input PTX string
	/usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx, line 10; fatal   : Unsupported .target 'sm_37'
	Cannot parse input PTX string
	)
	/home/simon/opticks/bin/op.sh: line 704: 108721 Aborted                 /usr/local/opticks/lib/OpTest --size 1920,1080,1 --snap --j1707 --gltf 3 --tracer --target 12 --eye 0.85,0.85,0.
	/home/simon/opticks/bin/op.sh RC 134
	[simon@localhost opticks]$ 





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



okop-env(){
    olocal-
   # optix-
   # optix-export
   # opticks-
}

okop-sdir(){ echo $(opticks-home)/okop ; }
okop-tdir(){ echo $(opticks-home)/okop/tests ; }
okop-idir(){ echo $(opticks-idir); }
okop-bdir(){ echo $(opticks-bdir)/$(okop-rel) ; }

okop-bin(){  echo $(okop-idir)/bin/${1:-OpIndexerTest} ; }

okop-c(){    cd $(okop-sdir); }
okop-cd(){   cd $(okop-sdir); }
okop-scd(){  cd $(okop-sdir); }
okop-tcd(){  cd $(okop-tdir); }
okop-icd(){  cd $(okop-idir); }
okop-bcd(){  cd $(okop-bdir); }

okop-name(){ echo OKOP ; }
okop-tag(){  echo OKOP ; }


okop-apihh(){  echo $(okop-sdir)/$(okop-tag)_API_EXPORT.hh ; }
okop---(){     touch $(okop-apihh) ; okop--  ; } 



okop-wipe(){ local bdir=$(okop-bdir) ; rm -rf $bdir ; } 
okop--(){                   opticks-- $(okop-bdir) ; } 
okop-t(){                  opticks-t $(okop-bdir) $* ; } 
okop-genproj() { okop-scd ; opticks-genproj $(okop-name) $(okop-tag) ; } 
okop-gentest() { okop-tcd ; opticks-gentest ${1:-OExample} $(okop-tag) ; } 
okop-txt(){ vi $(okop-sdir)/CMakeLists.txt $(okop-tdir)/CMakeLists.txt ; } 



okop-options(){
   echo -n
}

okop-cmake-deprecated(){
   local iwd=$PWD

   local bdir=$(okop-bdir)
   mkdir -p $bdir

   okop-bcd

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(okop-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(okop-options) \
       $(okop-sdir)


   cd $iwd
}

okop-index(){
    local msg="=== $FUNCNAME : "
    local tag=${1:--5}
    local cat=${2:-rainbow}
    local typ=${3:-torch}

    local shdir=$(okop-index-path sh $tag $cat $typ)
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
       0) $(okop-bin) --tag $tag --cat $cat   ;;
       1) lldb $(okop-bin) -- --tag $tag --cat $cat   ;;
    esac
}

okop-index-path(){
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

okop-index-op(){
   local tag=-5
   local cat=rainbow
   local typ=torch
   local cmps="ps rs sh sm"
   local path 
   local cmp
   for cmp in $cmps ; do
       #echo $cmp $tag $cat $typ
       path=$(okop-index-path $cmp $tag $cat $typ)  
       echo $path
   done
}



okop-snap-tag(){ echo 0 ; }
okop-snap-base(){ echo /tmp/okop_snap ; }
okop-snap-dir(){  echo $(okop-snap-base)/$(okop-snap-tag) ; }
okop-snap-cd(){  cd $(okop-snap-dir) ; }

okop-snap()
{
    ## intended to give same snap as okop-snap-gui, must OpticksHub::setupCompositionTargetting for this to be so
    local snapdir=$(okop-snap-dir)
    mkdir -p $snapdir

    local snapconfig="steps=100,eyestartz=0.,eyestopz=0.1,prefix=${snapdir}/,postfix=.ppm"

    op --snap --j1707 --gltf 3 --tracer --target 12 --eye 0.85,0.85,0. --snapconfig $snapconfig
}

okop-snap-mp4()
{
    okop-snap-cd
    local tag=$(basename $PWD)
    local mp4=${tag}.mp4

    ffmpeg -i %05d.ppm -pix_fmt yuv420p $mp4

    scp $mp4 D:
}



okop-propagate()
{
    ## intended to give same snap as okop-snap-gui, must OpticksHub::setupCompositionTargetting for this to be so
    op --snap --j1707 --gltf 3 --tracer --target 12 --eye 0.85,0.85,0.
    libpng-;libpng-- /tmp/snap.ppm
}




okop-snap-gui()
{
    ## to make a snap, need to switch to OptiX render mode with "O" key once GUI has launched
    op  --j1707 --gltf 3 --tracer --target 12 --eye 0.85,0.85,0. --rendermode +bb0,+in1,+in2,+in3,-global
    libpng-;libpng-- /tmp/snap.ppm
}




