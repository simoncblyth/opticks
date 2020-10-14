##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

geocache-source(){ echo $BASH_SOURCE ; }
geocache-vi(){ vi $(geocache-source) ; geocache- ; }
geocache-sdir(){ echo $(dirname $(geocache-source)) ; }
geocache-scd(){  cd $(geocache-dir) ; }
geocache-usage(){ cat << EOU


Geocache
==========


Directories
-------------

laptop::


    epsilon:boostrap blyth$ opticks-geocachedir
    /Users/blyth/.opticks/geocache

    opticks-geocachedir () 
    { 
        echo $(opticks-geocache-prefix)/geocache
    }

    opticks-geocache-prefix () 
    { 
        echo ${OPTICKS_GEOCACHE_PREFIX:-$(opticks-sharedcache-prefix-default)}
    }

    epsilon:boostrap blyth$ echo ${OPTICKS_GEOCACHE_PREFIX}

    opticks-sharedcache-prefix-default () 
    { 
        echo $HOME/.opticks
    }


Used to be:

    /usr/local/opticks/geocache



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

geocache-paths(){  echo $IDPATH/$1 $IDPATH2/$1 ; }
geocache-diff-(){  printf "\n======== $1 \n\n" ; diff -y $(geocache-paths $1) ; }
geocache-diff()
{
   export IDPATH2=/usr/local/opticks-cmake-overhaul/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1
   geocache-diff- GItemList/GMaterialLib.txt 
   geocache-diff- GItemList/GSurfaceLib.txt 
   geocache-diff- GNodeLib/PVNames.txt
   geocache-diff- GNodeLib/LVNames.txt
}


geocache-paths-pv(){ geocache-paths GNodeLib/PVNames.txt ; }
geocache-diff-pv(){ geocache-diff- GNodeLib/PVNames.txt ; }
geocache-diff-lv(){ geocache-diff- GNodeLib/LVNames.txt ; }

geocache-py()
{
   geocache-scd
   ipython -i geocache.py 
}

geocache-info(){ cat << EOI
$FUNCNAME
===============

  IDPATH  : $IDPATH
  IDPATH2 : $IDPATH2
       dependency on IDPATH on way out 

  (input)   OPTICKS_KEY        :  ${OPTICKS_KEY}
  (derived) OPTICKS_KEYDIR     :  ${OPTICKS_KEYDIR}    

  geocache-keydir    : $(geocache-keydir)
  geocache-keydir-py : $(geocache-keydir-py)
      above two directories should match 

  geocache-tstdir    : $(geocache-tstdir)
      directory derived from the OPTICKS_KEY envvar 

  geocache-dir       : $(geocache-dir) 
     formerly placed in install location $LOCAL_BASE/opticks/geocache 




  (default)  opticks-sharedcache-prefix-default : $(opticks-sharedcache-prefix-default) 
  (override) OPTICKS_GEOCACHE_PREFIX            : $OPTICKS_GEOCACHE_PREFIX   
  (result)   opticks-geocache-prefix            : $(opticks-geocache-prefix) 

  (result)   opticks-geocachedir                : $(opticks-geocachedir)
  

   
 




EOI

  geocache-keydir-test 
}

geocache-keydir-test()
{
   local a=$(geocache-keydir)
   local b=$(geocache-keydir-py)
   [ "$a" != "$b" ] && echo $msg MISMATCH $a $b && sleep 1000000000000
}


geocache-keydir-()
{
    local k=$OPTICKS_KEY
    local arr=(${k//./ })
    [ "${#arr[@]}" != "4" ] && echo $msg expecting OPTICKS_KEY envvar with four fields separate by dot && sleep 100000
    local exe=${arr[0]}
    local cls=${arr[1]}
    local top=${arr[2]}
    local dig=${arr[3]}
    echo ${exe}_${top}_g4live/g4ok_gltf/$dig/1 
}

geocache-keydir(){        echo $(geocache-dir)/$(geocache-keydir-) ; }
geocache-keydir-shared(){ echo $(geocache-dir-shared)/$(geocache-keydir-) ; }

geocache-dir-shared(){ echo /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache ; }

geocache-keydir-copy-to-shared(){
    local src=$(geocache-keydir)
    local dst=$(geocache-keydir-shared)

    [ ! -d "$src" ] && echo $msg src $src does not exist && return 1 
    [ -d "$dst" ] && echo $msg dst $dst exists already && return 1

    mkdir -p $(dirname $dst)
    echo cp -r $src $(dirname $dst)/ 
}





geocache-keydir-py(){ key.py ; }


geocache-dir(){ echo $(opticks-geocachedir) ; }
geocache-cd(){ cd $(geocache-dir) ; }
geocache-tstdir(){ echo $(geocache-keydir)/g4codegen/tests ; }
geocache-kcd(){ 
   local msg="=== $FUNCNAME :"
   if [ -z "$OPTICKS_KEY" ] ; then 
      echo $msg OPTICKS_KEY is not defined  
   else
      echo $msg OPTICKS_KEY $OPTICKS_KEY 
      local keydir=$(geocache-keydir)

      if [ -d "$keydir" ]; then  
          cd $keydir 
          pwd 
          cat runcomment.txt 
      else
          echo $msg keydir $keydir does not exist 
      fi 
   fi 

 }
geocache-tcd(){ cd $(geocache-tstdir) ; pwd ; }

geocache-tmp(){ echo /tmp/$USER/opticks/$1 ; }  


geocache-create-()
{
    local iwd=$PWD
    local tmp=$(geocache-tmp $FUNCNAME)
    mkdir -p $tmp && cd $tmp
         
    $(opticks-prefix)/bin/o.sh --okx4test --g4codegen --deletegeocache $*

    cd $iwd
}

geocache-create-notes(){ cat << EON
$FUNCNAME
----------------------

This parses the gdml, creates geocache, pops up OpenGL gui, 

EON
}


# comment 5 constant property that fail to parse 

#geocache-gdml-constant-fix(){ perl -pi -e 's,^\s*(<constant.*/>),<!-- $1  -->,g' $1 ; }
geocache-gdml-kludge(){        perl -pi -e 's,^(\s*<property name="(\S*)" ref="\2"/>),<!-- $1  -->,g' $1 ; }
geocache-gdml-kludge-dump(){   perl -n -e 'm,<property name="(\S*)" ref="\1"/>, && print ' $1 ; }  ## dump property with same @name @ref 
geocache-gdml-kludge-notes(){ cat << EON

Five props with name and ref the same, causing parse problem::

      <property name="SCINTILLATIONYIELD" ref="SCINTILLATIONYIELD"/>
      <property name="RESOLUTIONSCALE" ref="RESOLUTIONSCALE"/>
      <property name="FASTTIMECONSTANT" ref="FASTTIMECONSTANT"/>
      <property name="SLOWTIMECONSTANT" ref="SLOWTIMECONSTANT"/>
      <property name="YIELDRATIO" ref="YIELDRATIO"/>

EON
}



geocache-tds-gdmlpath()
{
   #local label=tds
   #local label=tds_ngt_pcnk
   local label=tds_ngt_pcnk_sycg
   local gdml=$(opticks-prefix)/$label.gdml
   echo $gdml
}



geocache-tds(){     
    local msg="=== $FUNCNAME :"

    #export GMaterialLib=INFO
    #export GSurfaceLib=INFO
    #export X4Solid=INFO
    #export X4PhysicalVolume=INFO
    #export GMesh=INFO
    #export OGeo=INFO
    export GGeo=INFO

    local gdml=$(geocache-tds-gdmlpath)
    echo $msg gdml $gdml
    geocache-gdml-kludge      $gdml
    geocache-gdml-kludge-dump $gdml

    geocache-create- --gdmlpath $gdml -D $*

} 

geocache-g4ok(){
    local msg="=== $FUNCNAME :"
    export GGeo=INFO

    local gdml=$(geocache-tds-gdmlpath)
    echo $msg gdml $gdml

    geocache-gdml-kludge      $gdml
    geocache-gdml-kludge-dump $gdml

    local iwd=$PWD
    local tmp=$(geocache-tmp $FUNCNAME)
    mkdir -p $tmp && cd $tmp
         
    $(opticks-prefix)/bin/o.sh --g4oktest --g4codegen --deletegeocache --gdmlpath $gdml -D $*

    cd $iwd
}
 

geocache-j1808-todo(){ cat << EOT

   opticksdata- needs to be replaced with opticksaux-

EOT
}

geocache-j1808(){     opticksdata- ; geocache-create- --gdmlpath $(opticksdata-j)  --X4 debug --NPY debug $*  ; }
geocache-j1808-v2(){  opticksdata- ; geocache-create- --gdmlpath $(opticksdata-jv2) --csgskiplv 22  ; }
geocache-j1808-v3(){  opticksdata- ; geocache-create- --gdmlpath $(opticksdata-jv3) --csgskiplv 22  ; }
geocache-j1808-v3(){  opticksdata- ; geocache-create- --gdmlpath $(opticksdata-jv3) --csgskiplv 22  ; }
geocache-j1808-v4-(){ opticksdata- ; geocache-create- --gdmlpath $(opticksdata-jv4) $* ; }
geocache-j1808-v5-(){ opticksdata- ; geocache-create- --gdmlpath $(opticksdata-jv5) $* ; }

geocache-j1808-v6-(){ opticksdata- ; geocache-create- --gdmlpath $(opticksdata-jv5) --digestextra v6  $* ; }   
geocache-j1808-vt-(){ opticksdata- ; geocache-create- --gdmlpath $(opticksdata-jv5) --digestextra vt  $* ; }   


geocache-j1808-v4-comment(){ echo torus-less-skipping-just-lv-22-maskVirtual ; }
geocache-j1808-v4-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce ; }
geocache-j1808-v4-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4(){  geocache-j1808-v4- --csgskiplv 22 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }  

geocache-j1808-v4-t1-comment(){ echo leave-just-21-see-notes/issues/review-analytic-geometry.rst ; }
geocache-j1808-v4-t1-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.5cc3de75a98f405a4e483bad34be348f ; }
geocache-j1808-v4-t1-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4-t1(){ geocache-j1808-v4- --csgskiplv 22,17,20,18,19 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }

geocache-j1808-v4-t2-comment(){ echo skip-22-virtualMask+20-almost-degenerate-inner-pyrex-see-notes/issues/review-analytic-geometry.rst ; }
geocache-j1808-v4-t2-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.781dc285412368f18465809232634d52 ; }
geocache-j1808-v4-t2-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4-t2(){ geocache-j1808-v4- --csgskiplv 22,20 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }

geocache-j1808-v4-t3-comment(){ echo skip-22-virtualMask+17-mask+20-almost-degenerate-inner-pyrex-see-notes/issues/review-analytic-geometry.rst ; }
geocache-j1808-v4-t3-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec ; }
geocache-j1808-v4-t3-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4-t3(){ geocache-j1808-v4- --csgskiplv 22,17,20 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $*  ; }

geocache-j1808-v4-t4-comment(){ echo skip-22-virtualMask+17-mask+20-almost-degenerate-inner-pyrex+19-remainder-vacuum-see-notes/issues/review-analytic-geometry.rst ; }
geocache-j1808-v4-t4-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.078714e5894f31953fc9afce731c77f3 ; }
geocache-j1808-v4-t4-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4-t4(){ geocache-j1808-v4- --csgskiplv 22,17,20,19 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }

geocache-j1808-v4-t5-comment(){ echo just-18-hemi-ellipsoid-cathode-cap-see-notes/issues/review-analytic-geometry.rst ; }
geocache-j1808-v4-t5-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.732c52dd2f92338b4c570163ede44230 ; }
geocache-j1808-v4-t5-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4-t5(){ geocache-j1808-v4- --csgskiplv 22,17,21,20,19 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }

geocache-j1808-v4-t6-comment(){ echo just-19-vacuum-remainder-see-notes/issues/review-analytic-geometry.rst ; }
geocache-j1808-v4-t6-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.d4157cb873000b4e19f77654134c3196 ; }
geocache-j1808-v4-t6-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4-t6(){ geocache-j1808-v4- --csgskiplv 22,17,21,20,18 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }

geocache-j1808-v4-t7-comment(){ echo just-18-19-vacuum-cap-and-remainder-see-notes/issues/review-analytic-geometry.rst ; }
geocache-j1808-v4-t7-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.e13cbdbe8782ca4ca000b735f0c4d61a ; }
geocache-j1808-v4-t7-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4-t7(){ geocache-j1808-v4- --csgskiplv 22,17,21,20 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }

geocache-j1808-v4-t8-comment(){ echo just-21-18-19-outer-pyrex+vacuum-cap-and-remainder-see-notes/issues/review-analytic-geometry.rst ; }
geocache-j1808-v4-t8-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec ; }       ## NB this matches geocache-j1808-v4-t3
geocache-j1808-v4-t8-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v4-t8(){ geocache-j1808-v4- --csgskiplv 22,17,20 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) ; }

geocache-j1808-v4-notes(){ cat << EON

$FUNCNAME : varying the volumes of the 20-inch PMT
==================================================================

The above geocache-j1808-v4-.. functions use the v4 (torus-less) JUNO j1808 GDML geometry
using OKX4Test to directly create the geocache from the GDML.
The t1,t2 etc functions vary the volumes of the 20inch PMT by using different csgskiplv, 
allowing benchmark checks of RTX mode on TITAN V and TITAN RTX with different geometry.  

NB geo digests still not trusted here, so that means rerun geocache conversions 


======================    ====================  ===============   ============================================================ 
 func                       --csgskiplv           leaving 
======================    ====================  ===============   ============================================================ 
geocache-j1808-v4           22                                      full 5 volumes   
geocache-j1808-v4-t1        22,17,20,18,19         21               outer pyrex
geocache-j1808-v4-t2        22,20                  17,21,18,19      acrylic mask + outer pyrex + vacuum cap + remainder  
geocache-j1808-v4-t3        22,17,20               21,18,19         outer pyrex + cathode cap + remainder 
geocache-j1808-v4-t4        22,17,20,19            21,18            outer pyrex + cathode cap
geocache-j1808-v4-t5        22,17,21,20,19         18               just-18-hemi-ellipsoid-cathode-cap
geocache-j1808-v4-t6        22,17,21,20,18         19               just-19-vacuum-remainder
geocache-j1808-v4-t7        22,17,21,20            18,19            just-18-19-vacuum-cap-and-remainder 
geocache-j1808-v4-t8        22,17,20               21,18,19         just-21-18-19-outer-pyrex+vacuum-cap-and-remainder
======================    ====================  ===============   ============================================================ 


GNodeLib/PVNames.txt 1-based index from vim, first 20inch::

     63555 lFasteners_phys0x4c31eb0

     63556 lMaskVirtual_phys0x4c9a510          22      csgskipped
     63557 pMask0x4c3bf20                      17 *   7 parts : difference of two ellipsoid cylinder unions 

     63558 PMT_20inch_log_phys0x4ca16b0        21 *   7 parts : union of el+co+cy  (5 parts, but seven as complete tree)
     63559 PMT_20inch_body_phys0x4c9a7f0       20 *   7 parts : union of el+co+cy  (ditto)
                 
     63560 PMT_20inch_inner1_phys0x4c9a870     18 *   1 part  : el                               cathode vacuum cap
     63561 PMT_20inch_inner2_phys0x4c9a920     19 *   7 parts : union of el+co+cy  (ditto)       remainder vacuum 
                                                   -----------------------------------
                                                      29 parts 
                                                   ------------------------------------

geocache-j1808-v4-t8
     for a while used this t8 geometry to put back a bug : --dbg_with_hemi_ellipsoid_bug 
     in order see its performance impact


The export functions setup the OPTICKS_KEY envvar in order to use the geometry. 
To switch geometry edit the below geocache-bashrc-export function and start a new shell
by opening a tab. 

See:

    notes/issues/review-analytic-geometry.rst 
    notes/issues/benchmarks.rst 

EON
}



geocache-j1808-v5-comment(){ echo fix-lv10-coincidence-speckle ; }
geocache-j1808-v5-key(){     
   case $(uname) in 
       Linux)  echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce  ;;
       Darwin) echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.a717fae122a5eda2027f6cec1d4a3f67  ;;
   esac 
}
geocache-j1808-v5-export(){  geocache-export ${FUNCNAME/-export} ; }
geocache-j1808-v5(){  geocache-j1808-v5- --csgskiplv 22 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }  


geocache-j1808-v6-key(){ echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.41b09364643f9479b9ab4f31451ce538 ; }
geocache-j1808-v6-comment(){ echo same-gdml-as-v5-but-fixes-lack-of-SD-with-isSensor-based-on-surface-EFFICIENCY ; }
geocache-j1808-v6(){  geocache-j1808-v6- --csgskiplv 22 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }  
geocache-j1808-v6-notes(){ cat << EON

v6 uses same gdml as v5, but 

1. initally changed conversion arguments such as adding --lvsdname inch_inner1_log
   but this old world approach is not appropriate when GDML is succeeding to persist surface props

2. instead changed to base GPropertyMap::isSensor on the existance of an EFFICIENCY property, this
   avoids the need to add kludge SensorSurfaces : regaining SD 

See notes/issues/missing-SD-again.rst

EON
}



geocache-j1808-vt-key(){ echo -n ; }
geocache-j1808-vt-comment(){ echo for-technical-tests-of-geocache-creation-with-no-expectation-of-longevity ; }
geocache-j1808-vt(){  X4PhysicalVolume=FATAL geocache-j1808-vt- --csgskiplv 22 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }  
geocache-j1808-vt-notes(){ cat << EON

vt uses same gdml as v6,v5, but its intended to be used for tests of geocache conversion after code changes, 
eg the restriction on balanced solids exports

See notes/issues/investigate-sams-x375-a-solid-that-breaks-balancing.rst

EON
}








geocache-dx0-notes(){ cat << EON

--x4polyskip 211,232
     to avoid ~infinite loops in G4SubtractionSolid::CreatePolyhedron called from X4Mesh::polygonize

EON
}


geocache-dx0-(){  opticksaux- ; geocache-create- --gdmlpath $(opticksaux-dx0) --x4polyskip 211,232  --geocenter --noviz $* ; }     
geocache-dx0-comment(){ echo export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation.rst ; }     
geocache-dx0(){   geocache-dx0- --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; } 
geocache-dx0-key(){ 
   case $(uname) in 
      Linux)  echo OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c ;;
      Darwin) echo OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3 ;;
   esac
}
geocache-dx0-keydir(){ OPTICKS_KEY=$(geocache-dx0-key) geocache-keydir ; }



geocache-dx1-notes(){ cat << EON

Manually modified dx0 GDML to make dx1, with addition of bordersurface with non-zero associated efficiency.

EON
}
geocache-dx1-(){  opticksaux- ; geocache-create- --gdmlpath $(opticksaux-dx1) --x4polyskip 211,232  --geocenter --noviz $* ; }     
geocache-dx1-comment(){ echo sensors-gdml-review.rst ; }     
geocache-dx1(){   geocache-dx1- --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; } 
geocache-dx1-key(){ 
   case $(uname) in 
      Linux)  echo OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c ;;
      Darwin) echo OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3 ;;
   esac
}
geocache-dx1-keydir(){ OPTICKS_KEY=$(geocache-dx1-key) geocache-keydir ; }









geocache-create-notes(){ cat << EON

Default geocache geometry is now Dayabay Near site detector (dx0)
which is a lot smaller geometry than the JUNO detector (j1808-v5)

EON
}

#geocache-create(){ geocache-j1808-v5 $* ; }  
geocache-create(){ geocache-dx0 $* ; }     



geocache-key-export(){   geocache-j1808-v5-export ; }


geocache-j1808-v5-notes(){ cat << EON
$FUNCNAME : avoiding a coincidence speckle in box subtract cylinder
============================================================================

See notes/issues/tboolean-proxy-scan-LV10-coincidence-speckle.rst for ascii art explaining the change.

v4::

       555   <solids>
       556     <tube aunit="deg" deltaphi="360" lunit="mm" name="Upper_LS_tube0x5b2e9f0" rmax="400" rmin="0" startphi="0" z="3500"/>
       557     <opticalsurface finish="3" model="1" name="UpperChimneyTyvekOpticalSurface" type="0" value="0.2">
       558       <property name="REFLECTIVITY" ref="REFLECTIVITY0x5b300c0"/>
       559     </opticalsurface>
       560     <tube aunit="deg" deltaphi="360" lunit="mm" name="Upper_Steel_tube0x5b2eb10" rmax="407" rmin="402" startphi="0" z="3500"/>
       561     <tube aunit="deg" deltaphi="360" lunit="mm" name="Upper_Tyvek_tube0x5b2ec30" rmax="402" rmin="400" startphi="0" z="3500"/>
       562     <tube aunit="deg" deltaphi="360" lunit="mm" name="Upper_Chimney0x5b2e8e0" rmax="412" rmin="0" startphi="0" z="3500"/>
       563     <box lunit="mm" name="sBar0x5b34ab0" x="6860" y="26" z="10"/>
       564     <box lunit="mm" name="sBar0x5b34920" x="6860" y="26.3" z="10.3"/>
       565     <box lunit="mm" name="sModuleTape0x5b34790" x="6860" y="1691.3" z="12.1"/>
       566     <box lunit="mm" name="sModule0x5b34600" x="6861.2" y="1692.5" z="13.3"/>
       567     <box lunit="mm" name="sPlane0x5b34470" x="6861.2" y="6770.3" z="13.3"/>
       568     <box lunit="mm" name="sWall0x5b342e0" x="6861.2" y="6861.2" z="27.6"/>
       569     <box lunit="mm" name="BoxsAirTT0x5b33e60" x="48000" y="48000" z="5000"/>
       570     <tube aunit="deg" deltaphi="360" lunit="mm" name="Cylinder0x5b33ef0" rmax="500" rmin="0" startphi="0" z="4000"/>
                                                                                                                  ^^^^^^^^^^^^^^^
       571     <subtraction name="sAirTT0x5b34000">
       572       <first ref="BoxsAirTT0x5b33e60"/>
       573       <second ref="Cylinder0x5b33ef0"/>
       574       <position name="sAirTT0x5b34000_pos" unit="mm" x="0" y="0" z="-500"/>
                                                                              ^^^^^^^^^
       575     </subtraction>
       576     <box lunit="mm" name="sExpHall0x4bcd390" x="48000" y="48000" z="18600"/>
       577     <box lunit="mm" name="sTopRock0x4bccfc0" x="54000" y="54000" z="21600"/>
       578     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="sTarget_bottom_ball0x4bd40d0" rmax="17700" rmin="0" startphi="0" starttheta="0"/>
       579     <tube aunit="deg" deltaphi="360" lunit="mm" name="sTarget_top_tube0x4bd4260" rmax="400" rmin="0" startphi="0" z="124.520351230938"/>


v5::

       569     <box lunit="mm" name="BoxsAirTT0x5b33e60" x="48000" y="48000" z="5000"/>
       570     
       571     <!-- see notes/issues/tboolean-proxy-scan-LV10-coincidence-speckle.rst -->
       572     <!--tube aunit="deg" deltaphi="360" lunit="mm" name="Cylinder0x5b33ef0" rmax="500" rmin="0" startphi="0" z="4000"/-->
       573     <tube aunit="deg" deltaphi="360" lunit="mm" name="Cylinder0x5b33ef0" rmax="500" rmin="0" startphi="0" z="4002"/>
       574     
       575     <subtraction name="sAirTT0x5b34000">
       576       <first ref="BoxsAirTT0x5b33e60"/>
       577       <second ref="Cylinder0x5b33ef0"/>
       578       <!--position name="sAirTT0x5b34000_pos" unit="mm" x="0" y="0" z="-500"/-->
       579       <position name="sAirTT0x5b34000_pos" unit="mm" x="0" y="0" z="-501"/>
       580     </subtraction>
       581 

::

    [blyth@localhost issues]$ opticksdata-j-info

       opticksdata-j   : /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00.gdml
       opticksdata-jv2 : /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v2.gdml
       opticksdata-jv3 : /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v3.gdml     
       opticksdata-jv4 : /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v4.gdml     opticksdata-jv4-vi
       opticksdata-jv5 : /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v5.gdml     opticksdata-jv5-vi 

    [blyth@localhost issues]$ cp /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v4.gdml /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v5.gdml




EON
}





geocache-export()
{
    local geofunc=$1
    export OPTICKS_GEOFUNC=$geofunc
    export OPTICKS_KEY=$(${geofunc}-key)
    export OPTICKS_COMMENT=$(${geofunc}-comment)

    #[ -t 1 ] && geocache-desc     ## only when connected to terminal 
}
geocache-desc()
{
    printf "%-16s : %s \n" "OPTICKS_GEOFUNC" $OPTICKS_GEOFUNC
    printf "%-16s : %s \n" "OPTICKS_KEY"     $OPTICKS_KEY
    printf "%-16s : %s \n" "OPTICKS_COMMENT" $OPTICKS_COMMENT
}



geocache-target(){ echo 352854 ; }
geocache-target-notes(){ cat << EON

Find targets by geocache-kcd and looking at GNodeLib/GTreePresent.txt eg::

    62588     62587 [  9:   0/   1]    0 ( 3)            pBar0x5b3a400  sBar0x5b34ab0
    62589     62588 [  1:   1/   2]    1 ( 0)    pBtmRock0x4bd2650  sBottomRock0x4bcd770
    62590     62589 [  2:   0/   1]    1 ( 0)     pPoolLining0x4bd25b0  sPoolLining0x4bd1eb0
    62591     62590 [  3:   0/   1] 2308 ( 0)      pOuterWaterPool0x4bd2b70  sOuterWaterPool0x4bd2960
    62592     62591 [  4:   0/2308]    1 ( 0)       pCentralDetector0x4bd4930  sReflectorInCD0x4bd3040
    62593     62592 [  5:   0/   1] 55274 ( 0)        pInnerWater0x4bd4700  sInnerWater0x4bd3660
    62594     62593 [  6:   0/55274]    1 ( 0)         pAcylic0x4bd47a0  sAcrylic0x4bd3cd0
    62595     62594 [  7:   0/   1]    0 ( 0)          pTarget0x4bd4860  sTarget0x4bd4340
    62596     62595 [  6:   1/55274]    0 ( 4)         lSteel_phys0x4bd4d60  sStrut0x4bd4b80
    62597     62596 [  6:   2/55274]    0 ( 4)         lSteel_phys0x4bd4ec0  sStrut0x4bd4b80
    62598     62597 [  6:   3/55274]    0 ( 4)         lSteel_phys0x4bd4fe0  sStrut0x4bd4b80
    62599     62598 [  6:   4/55274]    0 ( 4)         lSteel_phys0x4bd50d0  sStrut0x4bd4b80
    ...
    65589    352852 [  7:   2/   3]    0 ( 0)          pLowerChimneySteel0x5b318b0  sChimneySteel0x5b314f0
    65590    352853 [  6:55273/55274]    1 ( 0)         lSurftube_phys0x5b3c810  sSurftube0x5b3ab80
    65591    352854 [  7:   0/   1]    0 ( 0)          pvacSurftube0x5b3c120  svacSurftube0x5b3bf50
    65592    352855 [  4:   1/2308]    2 ( 2)       lMaskVirtual_phys0x5cc1ac0  sMask_virtual0x4c36e10
    65593    352856 [  5:   0/   2]    0 ( 2)        pMask0x4c3bf20  sMask0x4ca38d0
    65594    352857 [  5:   1/   2]    1 ( 2)        PMT_20inch_log_phys0x4ca16b0  PMT_20inch_pmt_solid0x4c81b40



::

    OKTest --target 62590 --xanalytic    
    ## 62590 : pOuterWaterPool0x4bd2b70  sOuterWaterPool0x4bd2960  
    ## with this target calib photons appear at center (by eye with orthographic view anyhow)


    OKTest --target 62590 --xanalytic --eye -0.9,0,0
    ## canonical position



EON
}


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


geocache-gui()
{
   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""

   ## NB cvd slots can change between reboots
   ## for interop to work have to see only the GPU being used for display (TITAN RTX)
   ## BUT beware nvidia-smi and UseOptiX sometimes disagree on the slots : its UseOptiX that matters

   local cvd=1
   UseOptiX --cvd $cvd 

   $dbg OKTest \
                --cvd $cvd \
                --rtx 1 \
                --envkey \
                --xanalytic \
                --timemax 400 \
                --animtimemax 400 \
                --target 352851 \
                --eye -1,-1,-1  \
                 $*   
}

geocache-360()
{
   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""

   local cvd=1
   UseOptiX --cvd $cvd 

   local cameratype=2  # EQUIRECTANGULAR

   $dbg OKTest \
                --cvd $cvd \
                --envkey \
                --xanalytic \
                --target 62594  \
                --eye 0,0,0  \
                --tracer \
                --look 0,0,1  \
                --up 1,0,0 \
                --cameratype $cameratype \
                --enabledmergedmesh 1,2,3,4,5 \
                --rendercmd O1 \
                --rtx 1 \
                 $*   
}

geocache-360-notes(){ cat << EON

     --enabledmergedmesh 1,2,3,4,5 \
           list of mesh indices to include : note that global 0 is excluded in order to see PMTs

     --fullscreen \

     --rendercmd O1 \
           directly to composite : but actually there is is no projective for EQUIRECTANGULAR


360 degree view of all PMTs from the center of the 
scintillator (raytrace only as distinctly non-trivial to do 
using the rasterization pipeline).

Uses an equirectangular "projection" : actually 
just a mapping from pixels (x,y) to 
azimuthal (-pi:pi) and polar(-pi/2:pi/2) angles. 

Default frame size is 1920x1080 
Typical presentation size is 1280x720    --size 1280,720,1
Double that 2560x1440

To take a frame snap press TAB, which saves to /tmp/Frame.ppm

::

    geocache-360 --size 2560,1440,1

Hmm doing this, end up with 2560x1340 seems to be constrained by monitor size


Convert to png with 

   env-;libpng-;libpng-- Frame.ppm




EON
}

geocache-size-notes(){ cat << EON

   factor 4: 58.98 M pixels
   factor 2: 14.75 M pixels

EON
}

geocache-size()
{ 
   local factor=${1:-2}
   local width=$((  2560*factor ))
   local height=$(( 1440*factor ))
   echo $width,$height,1
}
geocache-pixels()
{
   local factor=${1:-2}
   local width=$((  2560*factor ))
   local height=$(( 1440*factor ))
   echo $(( $width * $height / 1000000 ))
}


geocache-bench360(){ OPTICKS_GROUPCOMMAND="$FUNCNAME $*" geocache-rtxcheck $FUNCNAME $* ; }
geocache-bench360-()
{
   type $FUNCNAME
   UseOptiX $*

   local factor=4
   local cameratype=2  # EQUIRECTANGULAR : all PMTs in view 
   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""
   $dbg OpSnapTest --envkey \
                   --target 62594  \
                   --eye 0,0,0  \
                   --look 0,0,1  \
                   --up 1,0,0 \
                   --snapconfig "steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25" \
                   --size $(geocache-size $factor) \
                   --enabledmergedmesh 1,2,3,4,5 \
                   --cameratype $cameratype \
                   --embedded \
                   $* 
}

geocache-bench360-notes(){ cat << EON

 geocache-rtxcheck 
       feeds in arguments : --cvd  --rtx  --run*  

EON
}

geocache-snap360()
{
   local cvd=1
   UseOptiX --cvd $cvd 

   local cameratype=2  # EQUIRECTANGULAR : all PMTs in view 
   #local factor=4     # 4: 58.98 M pixels
   #local factor=2     # 2: 14.75 M pixels
   local factor=1      # 1: 2560x1440 : twice typical presentation resolution

   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""
   $dbg OpSnapTest \
                   --cvd $cvd \
                   --rtx 1 \
                   --envkey \
                   --target 62594  \
                   --xanalytic \
                   --eye 0,0,0  \
                   --look 0,0,1  \
                   --up 1,0,0 \
                   --enabledmergedmesh 2 \
                   --cameratype $cameratype \
                   --snapconfig "steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25" \
                   --size $(geocache-size $factor) \
                   --embedded \
                   $* 

}




geocache-save()
{
   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""
   $dbg OKTest \
       --cvd 0,1 \
       --envkey \
       --xanalytic \
       --compute \
       --save
}

geocache-load()
{
   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""

   local cvd=1
   UseOptiX --cvd $cvd 

   $dbg OKTest \
        --cvd $cvd \
        --rtx 0 \
        --envkey \
        --xanalytic \
        --timemax 400 \
        --animtimemax 400 \
        --load
}




geocache-gui-notes(){ cat << EON


Adding --save option fails even after setting CUDA_VISIBLE_DEVICES=1::

    2019-05-09 15:26:57.701 INFO  [67138] [OpEngine::downloadEvent@149] .
    2019-05-09 15:26:57.701 INFO  [67138] [OContext::download@587] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void**)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)
    Aborted (core dumped)
    [blyth@localhost opticks]$ 


Adding "--compute" with the "--save" succeeds to save 



EON
}




geocache-tour-()
{
   type $FUNCNAME
   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""
   $dbg OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=10,eyestartz=-1,eyestopz=5" --size 2560,1440,1 --embedded  $* 
}


# preserving spaces through multiple levels of argument passing is painful but its trivial to do via envvar 
geocache-bench(){  OPTICKS_GROUPCOMMAND="$FUNCNAME $*" geocache-rtxcheck $FUNCNAME $* ; } 
geocache-bench-()
{
   type $FUNCNAME
   UseOptiX $*
   local dbg
   [ -n "$DBG" ] && dbg="gdb --args" || dbg=""
   $dbg OpSnapTest --envkey --target $(geocache-bench-target) --eye -1,-1,-1 --snapconfig "steps=5,eyestartz=-1,eyestopz=-0.5" --size 5120,2880,1 --embedded  $* 
}
geocache-bench-target(){ echo ${GEOCACHE_BENCH_TARGET:-352851} ; }   # chimney region default
geocache-bench-check(){  geocache-bench- --cvd 1 --rtx 0 --runfolder $FUNCNAME --runstamp $(date +%s)  --xanalytic ; }


geocache-machinery-notes(){ cat << EON

Check the bach function preparation of arguments for a 
group of runs using and with the UseOptiX executable.

EON
}

geocache-machinery(){ geocache-rtxcheck $FUNCNAME $* ; }
geocache-machinery-()
{
   UseOptiX $* --cmdline 
}


geocache-rtxcheck-notes(){ cat << EON
$FUNCNAME
========================

0. used by functions such as geocache-bench360 which pass their FUNCNAME
   as the first argument, this funcname with a dash appended is then
   invoked with additional arguments that vary the RTX settings 

1. arguments after the first are passed to the function, this is typically
   used to switch to analytic with --xanalytic 

2. depends on UseOptiX that needs to be manually built::

   cd examples/UseOptiX
   ./go.sh

3. UseOptiX --uniqrec returns eg Quadro_RTX_8000/0

EON
}


geocache-rtxcheck()
{
   local name=${1:-geocache-bench}
   shift

   local stamp=$(date +%s)
   local ndev=$(UseOptiX --num)

   local uniqrec
   local uniqname
   local ordinal

   local scan=1
   if [ $scan -eq 1 ] 
   then
       UseOptiX --uniqrec | while read uniqrec ; do 
           ordinal=$(basename $uniqrec)
           uniqname=$(dirname $uniqrec)

           $name- --cvd $ordinal --rtx 0 --runfolder $name --runstamp $stamp --runlabel "R0_$uniqname" $*
           $name- --cvd $ordinal --rtx 1 --runfolder $name --runstamp $stamp --runlabel "R1_$uniqname" $*
           $name- --cvd $ordinal --rtx 2 --runfolder $name --runstamp $stamp --runlabel "R2_$uniqname" $*
       done
   fi

   if [ $ndev -eq 2 ]
   then
       local dev0=$(dirname $(UseOptiX --cvd 0 --uniqrec))
       local dev1=$(dirname $(UseOptiX --cvd 1 --uniqrec))
       $name- --cvd 0,1 --rtx 0 --runfolder $name --runstamp $stamp --runlabel "R0_${dev0}_AND_${dev1}" $*
   fi 

   if [ $ndev -gt 2 ]
   then
       local cvd
       geocache-cvd $ndev | while read cvd ; do
           $name- --cvd $cvd --rtx 0 --runfolder $name --runstamp $stamp $*
           $name- --cvd $cvd --rtx 1 --runfolder $name --runstamp $stamp $*
       done 
   fi 

   bench.py --name $name
}



geocache-cvd(){ geocache-cvd-even ; }

geocache-cvd-even(){  cat << EOC
0
0,1
0,1,2
0,1,2,3
4
4,5
4,5,6
4,5,6,7
EOC
}

geocache-cvd-linear()
{
   local ndev=${1:-8}
   local cvd
   local i=0
   while [ $i -lt $ndev ]; do
      [ -n "$cvd" ] &&  cvd=${cvd},$i || cvd=$i
      echo $cvd
      i=$(( $i + 1 )) 
   done
}



geocache-bench-results()
{
   bench.py $*
}


geocache-runfolder-names(){ cat << EON
geocache-bench
geocache-bench360
EON
}

geocache-runfolder-collect()
{
   local rnode=${1:-L7}
   local rdir=$OPTICKS_RESULTS_PREFIX/results
   [ ! -d "$rdir" ] && echo $msg missing rdir $rdir && return 

   local rfn
   local gdir
   geocache-runfolder-names | while read rfn 
   do 
      gdir=$rdir/$rfn  
      [ ! -d "$gdir" ] && echo $msg missing $gdir && return 
      cd 
      scp -r $rnode:g/local/opticks/results/$rfn/* $gdir
   done
   ## hmm simple when no name overlap between the cluster and workstation rungroup names  
   ## hmm better to rsync
}



geocache-bench-notes(){ cat << EON
$FUNCNAME
=======================

========  ====================================
  rtx       action 
========  ====================================
   -1       ASIS
    0       OFF
    1       ON  
    2       ON with optix::GeometryTriangles
========  ====================================


* cannot enable RTX when simultaneously using both TITAN V(Volta) 
  and TITAN RTX(Turing) as only Turing has the RT Cores.

* it is possible to enable RTX on Volta, sometimes resulting in a small speedup
  see notes/issues/benchmarks.rst

::

    geocache-bench-results --include xanalytic
    geocache-bench-results --exclude xanalytic

Former default location to write results was $LOCAL_BASE/opticks/results
but that doesnt make sense for multiple users running from the same 
install, so shift default to $TMP/results ie /tmp/$USER/opticks/results

Can write results elsewhere by setting envvar OPTICKS_RESULTS_PREFIX 
see BOpticksResource::ResolveResultsPrefix


EON
}



