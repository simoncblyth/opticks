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
            --target 3 \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 1,0,0 \
            --dbganalytic \
            --tag $(tgltf-tag) --cat $(tgltf-det) \
            --save 
}



tgltf-make-gdml(){  TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- ; }
tgltf-make-gdml-(){ $FUNCNAME- | python $* ; }
tgltf-make-gdml--(){ cat << EOP

import os, logging, sys, numpy as np

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.analytic.treebase import Tree
from opticks.analytic.gdml import GDML
from opticks.analytic.gdml_builder import make_gdml, tostring_
from opticks.analytic.sc import Sc

args = opticks_main()

pmt = "/dd/Geometry/PMT/lvPmtHemi0xc133740"
oil = "/dd/Geometry/AD/lvOIL0xbf5e0b8"

#skey = "pmt1"
#skey = "pmt2"
skey = "pmt5"
#skey = "collar"
#skey = "collar2"

gg = make_gdml(worldref=oil, structure_key=skey )
wgg = GDML.wrap(gg) 

tree = Tree(wgg.world)

#target = tree.findnode(gsel=0, gidx=None)

sc = Sc()
sc.extras["verbosity"] = 1
tg = sc.add_tree_gdml( tree.root, maxdepth=0 )

path = "$TMP/tgltf/$FUNCNAME.gltf"
gltf = sc.save(path)
print path

EOP
}


tgltf-gdml(){  TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- ; }
tgltf-gdml-(){ $FUNCNAME- | python $* ; }
tgltf-gdml--(){ cat << EOP

import os, logging, sys, numpy as np

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.analytic.treebase import Tree
from opticks.analytic.gdml import GDML
from opticks.analytic.sc import Sc

args = opticks_main()

oil = "/dd/Geometry/AD/lvOIL0xbf5e0b8"
sel = oil
#sel = 3153
sel = 1
idx = 0 

wgg = GDML.parse()
tree = Tree(wgg.world)

target = tree.findnode(sel=sel, idx=idx)

sc = Sc(maxcsgheight=4)
sc.extras["verbosity"] = 1
tg = sc.add_tree_gdml( target, maxdepth=0)

path = "$TMP/tgltf/$FUNCNAME.gltf"
gltf = sc.save(path)
print path

EOP
}


