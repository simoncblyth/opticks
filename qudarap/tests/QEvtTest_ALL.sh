#!/bin/bash
usage(){ cat << EOU

~/o/qudarap/tests/QEvtTest_ALL.sh 

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

tests=$(sed 's/#.*//' << EOT

one
sliced
many
loaded
checkEvt
quad6

EOT
)

source ALL_TEST_runner.sh QEvtTest.sh 


