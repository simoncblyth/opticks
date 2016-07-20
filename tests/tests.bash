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

tests-test-note(){ cat << EON
*tests-test*
     runs all enabled .bash in tests directory, 
     stopping at any script that yields a non-zero return code 

     Failure to load events SHOULD cause failures, 
     as the tests are expected to generate/simulate
     the events needed first.

     Tests should fail when expectations not fulfilled.
EON
}


tests-test(){
   local bash
   local rc
   tests-cd
   tests-enabled- | while read stem ; do

       bash=${stem}.bash
       echo $msg $bash $stem

       source $bash
       ${stem}-test
       rc=$?

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


