##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

tgltf-source(){   echo $BASH_SOURCE; }
tgltf-dir(){ echo $(dirname $BASH_SOURCE) ; }
tgltf-vi(){       vi $(tgltf-source) ; }
tgltf-usage(){ cat << \EOU

tgltf- 
======================================================

FUNCTIONS
-----------

tgltf-make-gdml
     testing small snippets of GDML

tgltf-gdml
     testing full analytic geometry mode, as switched on with 
     with "--gltf" option, and paths locating the gltf file

     handy options for debugging::
 
         --restrictmesh 0  # just the global 
         --restrictmesh 1  # just ridx:1 (eg PMTCollar, depending on root node)
         --restrictmesh 2  # just ridx:2 (the PMT assembly)   


EOU
}

tgltf-env(){      olocal- ;  }
tgltf-cd(){  cd $(tgltf-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tgltf-tag(){  echo 1 ; }
tgltf-det(){  echo gltf ; }
tgltf-src(){  echo torch ; }
tgltf-args(){ echo  --det $(tgltf-det) --src $(tgltf-src) ; }


tgltf-pretty(){
    local msg="=== $FUNCNAME :"
    local path=$1
    local pretty=${path/.gltf}.pretty.gltf
    cat $path | python -m json.tool > $pretty
    echo $msg wrote prettified gltf to $pretty
}


tgltf--()
{
    local msg="=== $FUNCNAME :"
    tgltf-

    local cmdline=$*
    local tgltfpath=${TGLTFPATH:-$TMP/nd/scene.gltf}

    local gltf=1
    #local gltf=4  # early exit at start of GScene::init
    #local gltf=44  # early exit at middle of GScene::init
    #local gltf=444  # early exit at end of GScene::init

     #--geocenter \

    #local target=$(tgltf-target)
    local target=0   # moving to absolute tree indexing following triangulated

    #        --eye 1,0,0 \

    local tmax=20   # too small for geometry yields maltese crosses 

    
       # --dbgnode 3159 \

    op.sh  \
            $cmdline \
            --debugger \
            --gltf $gltf \
            --gltfbase $(dirname $tgltfpath) \
            --gltfname $(basename $tgltfpath) \
            --gltftarget $target \
            --target 3 \
            --animtimemax $tmax \
            --timemax $tmax \
            --dbganalytic \
            --tag $(tgltf-tag) --cat $(tgltf-det) \
            --save \
            --rendermode +global,+axis,+in
}


tgltf-rip(){ local fnpy=$1 ; local py=$TMP/$fnpy.py ; $fnpy > $py ;  ipython -i $py ; }
tgltf-gdml-rip(){ tgltf-rip ${FUNCNAME/-rip}--  ; }  ## jump into ipython running the below script
tgltf-gdml-q(){  TGLTFPATH=$TMP/tgltf/${FUNCNAME/-q}--.gltf tgltf-- $* ; }


tgltf-target(){ echo 3153 ; }





tgltf-t-notes(){ cat << EON

Hmm currently the geoselection is applied to the gltf json 
at python level... which means must keep regenerating the 
gltf from the gdml every time want to change geoselection. 
Thats a pain... better to apply the selection at C++ level 
on loading the GLTF ? 

Attempting to change geoselection via envvars in python
or at this level in bash is too fragile ... instead 
make OPTICKS_QUERY settings with op-vi under
an op options such as --dsst, in this way the 
same geoselection applies to both branches and 
to python scripts : so long as they are 
orchestrated by op.sh.

When debugging geometry and changing geoselection, remember that need
to regenerate geocache after geoselection or precache code changes... 

::

   tgltf-;tgltf-t --dsst -G 




EON
}





tgltf-t() { TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- $* ; } 
tgltf-t-path(){ echo $TMP/${FUNCNAME/-path}-/sc.gltf ; }
tgltf-t-rm(){ local gltfpath=$(tgltf-t-path) ; rm $gltfpath ;}
tgltf-t-vi(){ local gltfpath=$(tgltf-t-path) ; vi ${gltfpath/.gltf}.pretty.gltf ;}
tgltf-t-ls(){ local gltfpath=$(tgltf-t-path) ; ls -l ${gltfpath/.gltf}* ; }
tgltf-t-()
{
    local gltfpath=$(tgltf-t-path)

    #tgltf-t-rm  # force regenerate every time

    [ ! -f "$gltfpath" ] && gdml2gltf.py --gltfpath $gltfpath
    echo $gltfpath
}


tgltf-tt(){  TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- $* ; }
tgltf-tt-(){ $FUNCNAME- | python $* ; }
tgltf-tt--(){ cat << EOP

# need same env in python and C++, 
# and also in analytic/non-analytic branches 
# so set OPTICKS_QUERY with op-vi underneath an op argument like --dsst
#

import os, logging, sys, numpy as np

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.analytic.treebase import Tree
from opticks.analytic.gdml import GDML
from opticks.analytic.sc import Sc

args = opticks_main()

gdml = GDML.parse()

tree = Tree(gdml.world)  

tree.apply_selection(args.query)   # sets node.selected "volume mask" 
    
sc = Sc(maxcsgheight=3)

sc.extras["verbosity"] = 1 
sc.extras["targetnode"] = 0   # args.query.query_range[0]   # hmm get rid of this ?

tg = sc.add_tree_gdml( tree.root, maxdepth=0)

debug = True
if debug:
    nd = sc.get_node(3159)
    nd.boundary = "Acrylic//perfectAbsorbSurface/GdDopedLS"
pass



gltf = sc.save(args.gltfpath)
print args.gltfpath

EOP
}  






tgltf-gdml(){  TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- $* ; }
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

#sel = oil
sel = 3153
#sel = 1
#sel = 0
idx = 0 

wgg = GDML.parse()
tree = Tree(wgg.world)

target = tree.findnode(sel=sel, idx=idx)
assert target.index == $(tgltf-target)



sc = Sc(maxcsgheight=3)
sc.extras["verbosity"] = 1
sc.extras["targetnode"] = target.index

tg = sc.add_tree_gdml( target, maxdepth=0)

path = "$TMP/tgltf/$FUNCNAME.gltf"
gltf = sc.save(path)

print path      ## <-- WARNING COMMUNICATION PRINT

#TODO: instead of just passing a path pass a config line or json snippet with the target

EOP
}


tgltf-gdml-placeholders()
{
    $OPTICKS_HOME/analytic/sc.py --lvnlist $TMP/tgltf/PLACEHOLDER_FAILED_POLY.txt 
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


