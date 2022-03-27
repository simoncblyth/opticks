#!/bin/bash -l
usage(){ cat << EOU
openline.sh
============
 
::

   ./openline.sh 0   # typically the slowest elv 
   ./openline.sh 1   
   ./openline.sh 137



Open path specified by a line of a file, for example
create the list of path using grabsnap.sh. 
See also image_grid.sh 

If the path is a standard geocache path then the meshname 
is looked up by this script. 

EOU
}
 
idx=${1:-0}
idx1=$(( $idx + 1 ))

tname=/tmp/ana_snap.txt
line=$(sed "${idx1}q;d" $tname) 


if [ -f "$line" ]; then 
   echo $msg line idx $idx idx1 $idx1  of file $tname is $line 
   open $line

   base=${line/CSG_GGeo*}

   if [ "$base" != "$line" ]; then
       meshname=$base/CSG_GGeo/CSGFoundry/meshname.txt
       tline=${line/_moi*} 
       midx0=${tline##*_}
       midx1=$(( $midx0 + 1 ))
       echo meshname $meshname
       mn=$(sed "${midx1}q;d" $meshname)
       echo $msg midx0 $midx0 midx1 $midx1 mn $mn 
   fi  

else
   echo $msg error line $idx of file $tname is $line : BUT that is not an existing path 
fi 

