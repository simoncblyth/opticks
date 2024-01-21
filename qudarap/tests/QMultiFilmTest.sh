#!/bin/bash
#Author Name:Yuxiang Hu
#Creation date:2023-12-31
#Description: 

cmd=$1

if [ ${cmd/run} != ${cmd} ]
then
    #export ARTPATH=/tmp/debug_multi_film_table/wv_sample_32_aoi_sample_64.npy
    #export ARTPATH=/tmp/debug_multi_film_table/wv_sample_64_aoi_sample_128.npy
    export ARTPATH=/tmp/debug_multi_film_table/wv_sample_128_aoi_sample_256.npy
    #export ARTPATH=/tmp/debug_multi_film_table/multifilm.npy
    QMultiFilmTest
fi
