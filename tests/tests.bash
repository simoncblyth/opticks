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

.. code-block:: py

    delta:ana blyth$ ipython
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    IPython profile: g4opticks

    In [1]: run tprism.py --tag 1
    tprism.py --tag 1
    INFO:__main__:sel prism/torch/  1 : TO BT BT SA 20160716-1941 /tmp/blyth/opticks/evt/prism/fdomtorch/1.npy 
    INFO:__main__:prism Prism(array([  60.,  300.,  300.,    0.]),Boundary Vacuum///GlassSchottF2 ) alpha 60.0  
    ...

.. toctree::

    tpmt 
    trainbow
    tnewton
    tprism
    tbox
    treflect



TODO:

* tsource
* tlens
* tg4gun



EOU
}


tests-env(){
    olocal-
    opticks-
}

tests-cd(){   cd $(tests-sdir); }

tests-bash-(){
   tests-cd
   ls -1 *.bash 
}
tests-enabled-(){
   local bash
   tests-bash- | while read bash ; do
      [ ! -x $bash ] && continue
      echo $bash
   done
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
   local stem
   local rc
   tests-enabled- | while read bash ; do

       stem=${bash/.bash}
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
  ls -1  *.bash | while read bash ; do
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


