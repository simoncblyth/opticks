#!/bin/bash 

DIR=$(dirname $(realpath $BASH_SOURCE))
export CUDA_VISIBLE_DEVICES=1
$DIR/SGLFW_SOPTIX_Scene_test.sh $*

