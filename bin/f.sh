#!/bin/bash
usage(){ cat << EOU
~/opticks/bin/f.sh
===================

Opens invoking directory into ipython with opticks.ana.fold:Fold

Usage example::

    GEOM std
    ~/opticks/bin/f.sh

Note a very similar script from np repo::

    ~/np/f.sh

EOU
}
DIR=$(cd $(dirname $BASH_SOURCE) && pwd)
${IPYTHON:-ipython} --pdb -i $DIR/f.py

