geocache-source(){ echo $BASH_SOURCE ; }
geocache-vi(){ vi $(geocache-source) ; }
geocache-sdir(){ echo $(dirname $(geocache-source)) ; }
geocache-scd(){  cd $(geocache-dir) ; }
geocache-usage(){ cat << EOU

Screen Capture Movies 
--------------------------

Avoiding White Flicker in movies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* make sure all executable logging (including obs) is 
  redirected to file, otherwise get white line flicker in the screen captures 

Window placement tip
~~~~~~~~~~~~~~~~~~~~~

* position the obs window with "start recording" button 
  to the right of the bottom right of the viz window, so 
  initial near scanning can work more smoothing  

Script
~~~~~~~~~~

1. obs-;obs-run
2. generate flightpath with ana/mm0prim2.py  
3. before starting the recording  

   a) run the viz geocache-;geocache-movie leaving window in orginal position, so screen recording gets it all 
   b) press O (wait) then O again # switch to composite raytrace render, then back to normal rasterized  
   c) make sure nothing obscuring the viz window 
   d) press alt-Y and adjust standard view 

4. starting recording 

   a) make sure obs "Start Recording" button is visible, and cursor is nearby
   b) press U : start the flightpath interpolated view
   c) press "Start Recording" in obs interface
  
4. during the recording : things to do 

   a) press Q, to switch off global a few times
   b) press X, and drag up/down to show the view along some parallel paths
      while doing this press O to switch to raytrace 
   c) press D, for orthographic view
   d) when photons are visible : press G and show the history selection 
   e) press . to stop event and geometry animations, take a look around, 
      press H to return to home for resumption of animation

5. ending the recording

   a) pick a good area to end with, eg chimney region



Choreography : 2018-10-29_13-24-36  ~6 min
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Movie is good enough if dont have time 
to try and record a better one.::

    epsilon:Movies blyth$ du -h 2018-10-29_13-24-36.mp4
    212M	2018-10-29_13-24-36.mp4

What can be improved:

* a few minutes shorter would be good
* forgot to use Q to switch off global during the flight, 
  this would have been particularly useful when R:rotating 
  view to look at PMTs 


::

    O: raytrace "eye" 1st frame

    [START RECORDING]

    click bottom right frame

    N: perspective near scan in/out
    D: ortho near scan in/out -> back to "eye"
    O: rasterized ortho, near scan in/out -> back to "eye"
    D: rasterized perspective, near scan in/out with raytrace flips 

    U: start flight 
    P: photon style

    .: pause when reach chimney region
    N: adjust near
    X: pan up with O flips to see chimney and TT
    H: home (back to flightpath)
    .: resume flight  

    on way down, X: out to see pool PMTs
    H: home (back to flightpath)

    on way up, D: ortho flip and back to see photons

    .: pause again when reach chimney region
    R: rotate and look up at PMTs,  N:scanning 
    H: home to flightpath 
    .:  resume
    .: pause once back outside
    D:ortho O:raytrace N:near scan to half G:photon select 


Issues 
--------

FIXED : awkward raytrace composite requiring hiding of rasterized geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formerly O was a 3 way: raster/raytrace/composite
composite is good to showing photons together with raytrace geometry :
but that requires to switch off rasterized geometry

Fixed the awkwardness by making it a 2-way raster/composite and 
always switching off rasterized geometry within composite mode



EOU
}

geocache-env(){ echo -n ; }
geocache-export()
{
   export IDPATH2=/usr/local/opticks-cmake-overhaul/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1
}

geocache-paths(){  echo $IDPATH/$1 $IDPATH2/$1 ; }
geocache-diff-(){  printf "\n======== $1 \n\n" ; diff -y $(geocache-paths $1) ; }
geocache-diff()
{
   geocache-export 
   geocache-diff- GItemList/GMaterialLib.txt 
   geocache-diff- GItemList/GSurfaceLib.txt 
   geocache-diff- GNodeLib/PVNames.txt
   geocache-diff- GNodeLib/LVNames.txt
}


geocache-paths-pv(){ geocache-paths GNodeLib/PVNames.txt ; }
geocache-diff-pv(){ geocache-diff- GNodeLib/PVNames.txt ; }
geocache-diff-lv(){ geocache-diff- GNodeLib/LVNames.txt ; }

geocache-info(){  cat << EOI

   IDPATH  : $IDPATH
   IDPATH2 : $IDPATH2

EOI
}

geocache-py()
{
   geocache-scd
   ipython -i geocache.py 
}



geocache-info(){ cat << EOI

  OPTICKS_KEY     :  ${OPTICKS_KEY}
  geocache-keydir : $(geocache-keydir)
  geocache-tstdir : $(geocache-tstdir)
      directory derived from the OPTICKS_KEY envvar 

EOI
}

geocache-keydir()
{
    local k=$OPTICKS_KEY
    local arr=(${k//./ })
    [ "${#arr[@]}" != "4" ] && echo $msg expecting OPTICKS_KEY envvar with four fields separate by dot && sleep 100000
    local exe=${arr[0]}
    local cls=${arr[1]}
    local top=${arr[2]}
    local dig=${arr[3]}
    echo $LOCAL_BASE/opticks/geocache/${exe}_${top}_g4live/g4ok_gltf/$dig/1 
    #echo $LOCAL_BASE/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
}

geocache-tstdir(){ echo $(geocache-keydir)/g4codegen/tests ; }

geocache-kcd(){ cd $(geocache-keydir) ; }
geocache-tcd(){ cd $(geocache-tstdir) ; }


geocache-tmp(){ echo /tmp/$USER/opticks/$1 ; }
geocache-j1808()
{
    local iwd=$PWD
    local tmp=$(geocache-tmp $FUNCNAME)
    mkdir -p $tmp && cd $tmp
         
    type $FUNCNAME
    opticksdata- 
    #gdb --args OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --csgskiplv 22,32,33
    #gdb --args OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --csgskiplv 22,32
    #gdb --args OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --csgskiplv 22 --X4 debug --NPY debug  

    cd $iwd
}

geocache-j1808-v2()
{
    local iwd=$PWD
    local tmp=$(geocache-tmp $FUNCNAME)
    mkdir -p $tmp && cd $tmp

    type $FUNCNAME
    opticksdata- 

    #gdb --args 
    OKX4Test --gdmlpath $(opticksdata-jv2) --g4codegen --csgskiplv 22 

    ## --X4 debug --NPY debug

    cd $iwd
}

geocache-j1808-v3()
{
    local iwd=$PWD
    local tmp=$(geocache-tmp $FUNCNAME)
    mkdir -p $tmp && cd $tmp

    type $FUNCNAME
    opticksdata- 

    gdb --args OKX4Test --gdmlpath $(opticksdata-jv3) --csgskiplv 22 

    cd $iwd
}





geocache-j1808-notes(){ cat << EON
$FUNCNAME
----------------------

With OptiX_600 CUDA 10.1 this parses the gdml, creates geocache, pops up OpenGL gui, 
switching to ray trace works but as soon as navigate into region where torus is needed
get the Misaligned address issue, presumably quartic double problem.

Torus strikes, see notes/issues/torus_replacement_on_the_fly.rst for the fix::

    [blyth@localhost issues]$ geocache-j1808
    geocache-j1808 is a function
    geocache-j1808 () 
    { 
        type \$FUNCNAME;
        opticksdata-;
        OKX4Test --gdmlpath \$(opticksdata-j) --g4codegen --csgskiplv 22
    }
    2019-04-15 10:45:36.211 INFO  [150689] [main@74]  parsing /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00.gdml
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/juno1808/g4_00.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    ...

    019-04-15 10:47:27.086 FATAL [150689] [ContentStyle::setContentStyle@98] ContentStyle norm inst 1 bbox 0 wire 0 asis 0 m_num_content_style 0 NUM_CONTENT_STYLE 5
    2019-04-15 10:47:32.590 INFO  [150689] [RenderStyle::setRenderStyle@95] RenderStyle R_COMPOSITE
    2019-04-15 10:47:32.820 INFO  [150689] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(1920,1080) ZProj.zw (-1.04082,-17316.9) front 0.5824,0.8097,-0.0719
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuEventSynchronize( m_event ) returned (716): Misaligned address)
    ^CKilled
    [blyth@localhost issues]$ 





EON
}


geocache-target(){ echo 352854 ; }

geocache-view()
{
    type $FUNCNAME
    env | grep OPTICKS_KEY
    env | grep CUDA
    #OKTest --envkey --xanalytic  --tracer --target $(geocache-target)
    OKTest --envkey --xanalytic 
}

geocache-movie-()
{
    # with --scintillation tried kludge symbolic link in opticksdata/gensteps g4live -> juno1707
    # but that gives applyLookup fails 
    #   --near 1000    not working, presumably overridden by basis aim  
    # --rendermode +global,+in0,+in1,+in2,+in3,+in4,+axis
    #
    # B2:viz instances
    # Q0:viz global

    env | grep OPTICKS_KEY
    OKTest --envkey \
            --xanalytic \
            --timemax 400 \
            --animtimemax 400 \
            --target $(geocache-target) \
            --eye -2,0,0 \
            --rendercmd B2,Q0 
}

geocache-movie(){ $FUNCNAME- 2>&1 > /tmp/$FUNCNAME.log ; }



geocache-bench-()
{
   type $FUNCNAME
   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""
   $dbg OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=5,eyestartz=-1,eyestopz=-0.5" --size 5120,2880,1 --embedded  $* 
}

geocache-bench()
{
   echo "TITAN RTX"
   CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=0 $FUNCNAME- 
   CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 $FUNCNAME- 
   echo "TITAN V" 
   CUDA_VISIBLE_DEVICES=0 OPTICKS_RTX=0 $FUNCNAME- 
   CUDA_VISIBLE_DEVICES=0 OPTICKS_RTX=1 $FUNCNAME- 
}



