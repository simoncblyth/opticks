#!/bin/bash -l 

usage(){ cat << EOU
CSGFoundryLoadTest.sh
========================



EOU

}

source $OPTICKS_HOME/bin/GEOM_.sh 

export SSim__load_tree_load=1 

loglevels()
{
    export CSGFoundry=INFO
    export SSim=INFO
}
loglevels





CSGFoundryLoadTest



