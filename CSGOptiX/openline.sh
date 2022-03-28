#!/bin/bash -l
usage(){ cat << EOU
openline.sh
============
 
::

   ./openline.sh 0   # typically the slowest elv 
   ./openline.sh 1   
   ./openline.sh 63   
        # its common to limit the jpg to the 64 slowest for image_grid.sh in that case 63 is the last 

   ./openline.sh 137  

Open path specified by a line of a file, for example
create the list of path using grabsnap.sh. 
See also image_grid.sh 

If the path is a standard geocache path then the meshname 
is looked up by this script. 

EOU
}

msg="=== $BASH_SOURCE :" 
idx0=${1:-0}
idx1=$(( $idx0 + 1 ))

pathlist=/tmp/ana_snap.txt
PATHLIST=${PATHLIST:-$pathlist}
line=$(sed "${idx1}q;d" $PATHLIST) 

if [ -f "$line" ]; then 
   echo $msg line idx0 $idx0 idx1 $idx1  of file $PATHLIST is $line 
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
   echo $msg error line idx0 $idx0 idx1 $idx1  of PATHLIST $PATHLIST is line $line : BUT that is not an existing path 
fi 


