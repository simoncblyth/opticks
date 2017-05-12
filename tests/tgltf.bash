tgltf-source(){   echo $(opticks-home)/tests/tgltf.bash ; }
tgltf-vi(){       vi $(tgltf-source) ; }
tgltf-usage(){ cat << \EOU

tgltf- 
======================================================


EOU
}

tgltf-env(){      olocal- ;  }
tgltf-dir(){ echo $(opticks-home)/tests ; }
tgltf-cd(){  cd $(tgltf-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tgltf-tag(){  echo 1 ; }
tgltf-det(){  echo gltf ; }
tgltf-src(){  echo torch ; }
tgltf-args(){ echo  --det $(tgltf-det) --src $(tgltf-src) ; }




tboolean--(){

    tboolean-

    local msg="=== $FUNCNAME :"
    local cmdline=$*

    op.sh  \
            $cmdline \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 1,0,0 \
            --dbganalytic \
            --test --testconfig "$testconfig" \
            --torch --torchconfig "$(tboolean-torchconfig)" \
            --tag $(tboolean-tag) --cat $(tboolean-det) \
            --save 
}



tgltf--(){

    tgltf-

    local cmdline=$*
    local tgltfpath=${TGLTFPATH:-$TMP/nd/scene.gltf}

    op.sh  \
            $cmdline \
            --debugger \
            --gltf 1 \
            --gltfbase $(dirname $tgltfpath) \
            --gltfname $(basename $tgltfpath) \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 1,0,0 \
            --dbganalytic \
            --tag $(tgltf-tag) --cat $(tgltf-det) \
            --save 
}



tgltf-pmt(){  TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- ; }
tgltf-pmt-(){ $FUNCNAME- | python $* ; }
tgltf-pmt--(){ cat << EOP

import os, logging, sys, numpy as np

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.gdml import GDML
from opticks.ana.pmt.gdml_builder import make_gdml, tostring_
from opticks.dev.csg.sc import Sc

args = opticks_main()

gg = make_gdml(worldref="/dd/Geometry/PMT/lvPmtHemi0xc133740")
wgg = GDML.wrap(gg) 

tree = Tree(wgg.world)

#target = tree.findnode(gsel=0, gidx=None)

sc = Sc()
tg = sc.add_tree_gdml( tree.root, maxdepth=0 )

path = "$TMP/tgltf/$FUNCNAME.gltf"
gltf = sc.save(path)
print path

EOP
}



