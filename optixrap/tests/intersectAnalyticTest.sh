#!/bin/bash -l

ia-usage(){ cat << EOU

::

   cd ~/opticks/optixrap/tests
   ./intersectAnalyticTest.sh 

EOU

}

ia-cus(){ 
   local iwd=$PWD
   cd cu
   ls -1 ia*.cu
   cd $iwd
}

ia-banner(){ cat << EOB
 
##########  $*  ############  

EOB
}

ia-test(){
  local msg="=== $FUNCNAME :"
  local cus=$(ia-cus)
  local cu
  local cmd
  local rc
  for cu in $cus ; do 
     cmd="intersectAnalyticTest --cu $cu"
     ia-banner $cmd 
     eval $cmd
     rc=$?
     [ $rc -ne 0 ] && echo $msg ERROR non-zero RC $rc
  done
}


ia-test



