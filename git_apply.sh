#!/bin/bash

usage(){ cat << EOU
git_apply.sh
=============

Usage example::

    cd ~/opticks
    ./git_apply.sh opticks_makepropertyfold_const.patch

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

patch=$1
[ -z "$patch" ] && echo $BASH_SOURCE - ERROR - REQUIRE PATCH FILE ARGUMENT && usage && exit 1
[ ! -f "$patch" ] && echo $BASH_SOURCE - ERROR - REQUIRE VALID PATH $patch && usage && exit 1

git apply --check $patch
[ $? -ne 0 ] && echo $BASH_SOURCE - ERROR - git apply --check - ERROR && exit 2

git apply $patch
[ $? -ne 0 ] && echo $BASH_SOURCE - ERROR - git apply - ERROR && exit 3

exit 0

