#!/bin/bash
usage(){ cat << EOU

~/o/qudarap/tests/QEventTest_ALL.sh 

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

source ALL_TEST_runner.sh QEventTest.sh 


