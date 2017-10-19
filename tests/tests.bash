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


tests-env(){
    olocal-
    opticks-
}

tests-dir(){  echo $(opticks-home)/tests ; }
tests-cd(){   cd $(tests-dir); }

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
           #break 
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


