geocache-source(){ echo $BASH_SOURCE ; }
geocache-vi(){ vi $(geocache-source) ; }
geocache-sdir(){ echo $(dirname $(geocache-source)) ; }
geocache-scd(){  cd $(geocache-dir) ; }
geocache-usage(){ cat << EOU

Movie Script
----------------

1. obs-;obs-run
2. generate flightpath with ana/mm0prim2.py  
3. starting the recording  

   a) run the viz geocache-;geocache-sc leaving window in orginal position, so screen recording gets it all 
   b) press O (wait) then O and O again # switch to raytrace render, then composite and back to normal  
   c) adjust camera near: press n + drag down half a screen, press n again to toggle off
   d) adjust window position to make sure nothing obscuring the viz window 
   e) make sure obs "Start Recording" button is visible, and cursor is nearby
   f) press U : start the flightpath interpolated view
   g) press "Start Recording" in obs interface
  
4. during the recording : things to do 

   a) press Q, to switch off global a few times
   b) press X, and drag up/down to show the view along some parallel paths
      while doing this press O to switch to raytrace 
   c) press D, for orthographic view
   d) which photons are visible : press G and show the history selection 

5. ending the recording

   a) pick a good area to end with, eg chimney region


Issues
--------

* pressing O is a 3-way cycle including composite,
  composite is good for showing photons together with the raytrace, which 
  means have to switch off the rasterized geometry : to avoid coincidence
  between the two geometries : but thats quite a few keys and then it switch it back   
  on again : better to auto switch off rasterized when switch to composite 



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

geocache-kcd(){ cd $(geocache-keydir) ; }


geocache-j1808()
{
    type $FUNCNAME
    opticksdata- 
    OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --csgskiplv 22 
}


geocache-target(){ echo 352854 ; }

geocache-view()
{
    type $FUNCNAME
    OKTest --envkey --xanalytic  --tracer --target $(geocache-target)
    #OKTest --envkey --xanalytic 
}

geocache-sc()
{
    # with --scintillation tried kludge symbolic link in opticksdata/gensteps g4live -> juno1707
    # but that gives applyLookup fails 

    OKTest --envkey --xanalytic --timemax 400 --animtimemax 400 --target $(geocache-target) --near 1000
}



