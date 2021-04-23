#!/bin/bash -l 

triplet=1:9/
tmpdir=$TMP/GMeshLibTest 

mkdir -p $tmpdir
cd $tmpdir 

# write solid names and indices to separate files
ggeo.py $triplet --sonames --errout 1>soidx.txt 2>sonames.txt

# lookup the indices from the names 
GMeshLibTest $(cat $tmpdir/sonames.txt) 1>/dev/null 2>soidx2.txt

# check the indices from the two routes match 
diff soidx.txt soidx2.txt
[ $? -ne 0 ] && echo $0 : discrepancy between solid indices && exit 1

pwd 
exit 0 
