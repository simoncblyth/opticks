#!/usr/bin/env bash

usage(){ cat << EOU

~/o/sysrap/tests/SEvt__createInputGenstep_configuredTest_SML.sh

These small/medium/large.npy are used from k6 load-testing::

   ~/np/tests/np_curl_test/np_curl_test.sh k6_load

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

names="small medium large"

for name in $names
do
    GS_NAME=$name.npy ./SEvt__createInputGenstep_configuredTest.sh
done

