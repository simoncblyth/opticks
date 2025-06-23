#!/bin/bash
usage(){ cat << EOU
opticks-ctest.sh
=================

This is the binary release equivalent of opticks-t which
operates from released binary directories as opposed to
opticks-t which operates in the build environment and tree.

EOU
}

BINDIR=$(dirname $(realpath $BASH_SOURCE))
PREFIX=$(dirname $BINDIR)
source $PREFIX/bashrc
source $PREFIX/tests/ctest.sh


