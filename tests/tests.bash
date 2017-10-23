tests-rel(){      echo tests ; }
tests-src(){      echo tests/tests.bash ; }
tests-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(tests-src)} ; }
tests-vi(){       vi $(tests-source) ; }
tests-usage(){ cat << \EOU

tests- : Opticks Integration Tests
=====================================

The below high level bash function tests all use the **op.sh** script :doc:`../bin/op` 
to simulate optical photons and save event files.  The functions then use 
python analysis scripts :doc:`../ana/index`   to compare events with each other 
and analytic expectations.

Note to see the plots produced by the tests during development you will 
need to use ipython and invoke them with **run** as shown below.  
See :doc:`../ana/tools`.

All the tests can be invoked via the `tests-t` function::

    simon:ana blyth$ tests-t
    ==                     treflect-t ==  ->    0 
    ==                         tbox-t ==  ->    0 
    ==                       tprism-t ==  ->    0 
    ==                      tnewton-t ==  ->    0 
    ==                       twhite-t ==  ->    0 
    ==                         tpmt-t ==  ->    0 
    ==                     trainbow-t ==  ->    0 
    ==                        tlens-t ==  ->    0 
    ==                       tg4gun-t ==  ->    0 





.. toctree::

    overview

    tviz
    tpmt 
    trainbow
    tnewton
    tprism
    tbox
    treflect
    twhite
    tlens
    tg4gun


EOU
}

tests-status(){ cat << \EOS


STATUS OF INTEGRATION TESTS
===============================

The note "working to some extent" means qualitatively operational, 
ie the tname-- function yields what is visually expected 
without any analysis checks of the events produced.

tboolean
    mostly working 
tbox
    working to some extent
tboxlaser
    working to some extent
tconcentric
    working to some extent
tdefault
    working to some extent
tg4gun
    working to some extent
tgltf
    working to some extent

    * odd viewpoint 
    * disabled numPrim == numSolids sanity check in OGeo::makeAnalyticGeometry

tjuno
    working to some extent

    Thoughts:

    * implemented raylod with instances in mind but 
      the benefits of outershelling to speedup raytrace will actual be felt 
      much more for the global geometry : so when PMTs not visible you dont 
      have to pay the price for them  

    * same comment is true of the OpenGL geometry, have so far mainly considered 
      the instances wrt LOD

tlaser
    working to some extent
tlens
    NOT WORKING 

    * old style zlens not working : noshow in raytrace
    * TODO: migrate to using python CSG geometry to define lens shapes, eliminate the old zlens

tnewton
    working to some extent

    * using old style manual prism definition, would benefit from CSG trapezoid

tpmt
    now working to some extent, see :doc:`../notes/issues/tpmt_broken_by_OpticksCSG_enum_move` 

    TODO: commit update enum analytic geometry into opticksdata slot 0 

tprism
    working to some extent
trainbow
    working to some extent
treflect
    working to some extent
tviz
    DOESNT FOLLOW PATTERN : NEEDS REVISIT, USING OLD OPTIONS ?
twhite
    working to some extent


EOS
}







tests-env(){
    olocal-
    opticks-
}

tests-dir(){  echo $(opticks-home)/tests ; }
tests-cd(){   cd $(tests-dir); }


tests-bash(){ tests-bash- | perl -p -e 's,.bash,,g' -  ; }
tests-bash-(){
   tests-cd
   ls -1 *.bash | grep -v ttemplate.bash | grep -v tests.bash 
}




tests-enabled-(){ cat << EOT
treflect
tbox
tprism
tnewton
twhite
tpmt
trainbow
tlens
tg4gun
EOT
}

tests-t-note(){ cat << EON
*tests-t*
     runs all enabled .bash in tests directory, 
     stopping at any script that yields a non-zero return code 

     Failure to load events SHOULD cause failures, 
     as the tests are expected to generate/simulate
     the events needed first.

     Tests should fail when expectations not fulfilled.
EON
}


tests-t(){
   local bash
   local rc
   tests-cd
   tests-enabled- | while read stem ; do

       bash=${stem}.bash

       printf "== %30s == " ${stem}-t
       source $bash

       ${stem}-t > /dev/null 2>&1
       rc=$?

       printf " -> %4s \n" $rc 

       if [ "$rc" != "0" ]; then 
           echo $msg WARNING : FAILURE OF $bash : RC $rc
           break 
       fi
   done   
}

tests-shim(){
  tests-cd 
  local bash
  local rst
  ls -1  *.bash | grep -v tests.bash | while read bash ; do
      rst=${bash/.bash}.rst
      if [ ! -f "$rst" ];  then
          echo $msg writing shim $rst for $bash
          tests-shim- $bash > $rst 
      fi
  done 
}

tests-shim-(){ cat << EOR 

.. include:: ${1:-dummy} 
   :start-after: cat << \EOU
   :end-before: EOU

EOR
}


