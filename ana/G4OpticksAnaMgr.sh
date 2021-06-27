#!/bin/bash -l 

outdir="/tmp/G4OpticksAnaMgr" 
mkdir -p $outdir

from=P:$outdir
to=$outdir


sync_cmd(){ cat << EOC
rsync -zarv --progress --include='*/' --include='*.npy' --include='*.json' --exclude='*' "${from}/" "${to}/"
EOC
}

cmd=$(sync_cmd)
echo $cmd
eval $cmd





