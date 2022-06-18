#!/bin/bash -l 
usage(){ cat << EOU
U4RandomTest.sh
=================

::

   ./U4RandomTest.sh

EOU
}

#seqdir="/tmp/$USER/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"
#export OPTICKS_RANDOM_SEQPATH=$seqdir
#export OPTICKS_RANDOM_SEQPATH=$seqdir/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy 

U4RandomTest 
[ $? -ne 0 ] && echo run FAIL && exit 1


exit 0 


