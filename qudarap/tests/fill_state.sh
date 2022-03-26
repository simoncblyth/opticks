#!/bin/bash -l 

m1_refractive_index=1.5
m2_refractive_index=1.0

export M1_REFRACTIVE_INDEX=${M1_REFRACTIVE_INDEX:-$m1_refractive_index}
export M2_REFRACTIVE_INDEX=${M2_REFRACTIVE_INDEX:-$m2_refractive_index}


fill-state-desc(){ cat << EOD
$BASH_SOURCE:$FUNCNAME
=========================

   M1_REFRACTIVE_INDEX  : $M1_REFRACTIVE_INDEX
   M2_REFRACTIVE_INDEX  : $M2_REFRACTIVE_INDEX


EOD
}

fill-state-desc


