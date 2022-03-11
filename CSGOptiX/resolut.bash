#!/bin/bash -l 
resolut-usage(){ cat << EOU
resolut.bash
=======================

EOU
}
resolut--(){ source $BASH_SOURCE && resolut-scan $* ; }
resolut-factors(){ echo "1 2 4 8 16 32 64 128 256" ; }
resolut-scan(){
   local factors=$(resolut-factors)
   local fac
   for fac in $factors ; do resolut-info $fac ; done 
}
resolut-info()
{
   local msg="=== $FUNCNAME :"
   local factor=${1:-1}
   local sz=$(resolut-size  $factor)
   local px=$(resolut-pixels $sz)
   local mpx=$(resolut-mpixels $sz)
   printf " factor %5d size %15s px %15s mpx %10s \n" $factor $sz $px $mpx 
}
resolut-size()
{ 
   local factor=${1:-1}
   local width=$(( 1280*factor ))
   local height=$(( 720*factor ))
   echo $width,$height,1
}
resolut-pixels()
{
   local sz=$1
   local expr=${sz//,/*}
   if [ -n "$MEGA" ]; then 
       echo $(( $expr/1000000 ))  
   else
       echo $(( $expr ))  
   fi 
}
resolut-mpixels(){ MEGA=1 resolut-pixels $* ; }


