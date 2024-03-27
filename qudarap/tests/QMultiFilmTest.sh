#!/bin/bash
#Author Name:Yuxiang Hu
#Creation date:2023-12-31
#Description: 

cmd=$1
art_fold=/data/ihep/tmp 
export ART_FOLD=${ART_FOLD:-${art_fold}}

qmultifilm_test_fold=/data/ihep/tmp/debug_multi_film_table/GPUTestArray/
export QMultiFilmTest_FOLD=${QMultiFilmTest_FOLD:-$qmultifilm_test_fold}

if [ "${cmd/run}" != "${cmd}" ]
then
    #export ARTPATH=/tmp/debug_multi_film_table/wv_sample_32_aoi_sample_64.npy
    #export ARTPATH=/tmp/debug_multi_film_table/wv_sample_64_aoi_sample_128.npy
    #export ARTPATH=${TMP}/debug_multi_film_table/CreateSamples/wv_sample_32_aoi_sample_64.npy
    export ARTPATH=${ART_FOLD}/debug_multi_film_table/multifilm.npy
    #export ARTPATH=/tmp/debug_multi_film_table/wv_sample_128_aoi_sample_256.npy
    #export ARTPATH=/tmp/debug_multi_film_table/multifilm.npy
    echo "ARTPATH: ${ARTPATH}"
    QMultiFilmTest
fi

if [ "${cmd/ana}" != "${cmd}" ]
then
    python QMultiFilmTest.py
fi
