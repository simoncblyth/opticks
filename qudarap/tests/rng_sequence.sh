#!/bin/bash -l

usage(){ cat << EOU
rng_sequence.sh 
==================

Usage::

   ./rng_sequence.sh run      # generates and persists precooked random .npy arrays

   ./rng_sequence.sh ana      # load the random .npy arrays 


EOU
}


arg=${1:-ana}

TEST=rng_sequence ./QSimTest.sh $arg


