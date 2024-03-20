#!/bin/bash
#Author Name:Yuxiang Hu
#Creation date:2023-12-31
#Description: 

cmd=$1
export TMP=/data/ihep/tmp

if [ "${cmd/run}" != "${cmd}" ]
then
    #export ARTPATH=/tmp/debug_multi_film_table/wv_sample_32_aoi_sample_64.npy
    #export ARTPATH=/tmp/debug_multi_film_table/wv_sample_64_aoi_sample_128.npy
    #export ARTPATH=${TMP}/debug_multi_film_table/CreateSamples/wv_sample_32_aoi_sample_64.npy
    export ARTPATH=${TMP}/debug_multi_film_table/multifilm.npy
    #export ARTPATH=/tmp/debug_multi_film_table/wv_sample_128_aoi_sample_256.npy
    #export ARTPATH=/tmp/debug_multi_film_table/multifilm.npy
    QMultiFilmTest
fi

if [ "${cmd/ana}" != "${cmd}" ]
then
    python QMultiFilmTest.py
fi
