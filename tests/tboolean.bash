tboolean-source(){   echo $(opticks-home)/tests/tboolean.bash ; }
tboolean-vi(){       vi $(tboolean-source) ; }
tboolean-usage(){ cat << \EOU

tboolean- 
======================================================


TODO
--------

* CSG geometry config using python that writes
  a serialization turns out to be really convenient...
  Howabout attaching emission of torch photons to pieces
  of geometry ?
  

NOTES
--------

tracetest option
~~~~~~~~~~~~~~~~~~~

When using tracetest option only a single intersect is
done using oxrap/cu/generate.cu:tracetest and a special 
format of the photon buffer is used, for analysis by 
ana/tboolean.py 

However in tracetest mode the record buffer filling 
is not implemented so the visualization 
of photon paths is not operational.


bash test geometry configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* CSG tree is defined in breadth first or level order

* parameters of boolean operations currently define adhoc box 
  intended to contain the geometry, TODO: calculate from bounds of the contained tree 

* offsets arg identifies which nodes belong to which primitives by pointing 
  at the nodes that start each primitive

::

     1  node=union        parameters=0,0,0,400           boundary=Vacuum///$material 
     2  node=difference   parameters=0,0,100,300         boundary=Vacuum///$material
     3  node=difference   parameters=0,0,-100,300        boundary=Vacuum///$material
     4  node=box          parameters=0,0,100,$inscribe   boundary=Vacuum///$material
     5  node=sphere       parameters=0,0,100,$radius     boundary=Vacuum///$material
     6  node=box          parameters=0,0,-100,$inscribe  boundary=Vacuum///$material
     7  node=sphere       parameters=0,0,-100,$radius    boundary=Vacuum///$material

Perfect tree with n=7 nodes is depth 2, dev/csg/node.py (root2)::
 
                 U1                
                  o                
         D2              D3        
          o               o        
     b4      s5      b6      s7    
      o       o       o       o         


* nodes identified with 1-based levelorder index, i
* left/right child of node i at l=2i, r=2i+1, so long as l,r < n + 1


python test geometry configuration 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* python config is much more flexible than bash, allowing 
  more natural tree construction and node reuse

Running functions such as tboolean-box-sphere-py- 
construct CSG node trees for each "solid" of the geometry.
Typically the containing volume is a single node tree 
and the contained volume is a multiple node CSG tree.

These trees are serialized into numpy arrays and written to 
files within directories named after the bash function eg 
"/tmp/blyth/opticks/tboolean-box-sphere-py-". 

The bash function emits to stdout only the name of 
this directory which is captured and used in the 
commandline testconfig csgpath slot.


testconfig modes
~~~~~~~~~~~~~~~~~~

PmtInBox

     * see tpmt- for this one

BoxInBox

     * CSG combinations not supported, union/intersection/difference nodes
       appear as placeholder boxes

     * raytrace superficially looks like a union, but on navigating inside 
       its apparent that its just overlapped individual primitives


PyCsgInBox

     * requires csgpath identifying directory containing serialized CSG trees
       and csg.txt file with corresponding boundary spec strings


CsgInBox

     * DECLARED DEAD, USE PyCsgInBox
     * requires "offsets" identifying node splits into primitives eg offsets=0,1 
     * nodes are specified in tree levelorder, trees must be perfect 
       with 1,3,7 or 15 nodes corresponding to trees of height 0,1,2,3

EOU
}

tboolean-env(){      olocal- ;  }
tboolean-dir(){ echo $(opticks-home)/tests ; }
tboolean-cd(){  cd $(tboolean-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tboolean-tag(){  echo 1 ; }
tboolean-det(){  echo boolean ; }
tboolean-src(){  echo torch ; }
tboolean-args(){ echo  --det $(tboolean-det) --src $(tboolean-src) ; }

tboolean--(){

    tboolean-

    local msg="=== $FUNCNAME :"
    local cmdline=$*

    local testconfig
    if [ -n "$TESTCONFIG" ]; then
        testconfig=${TESTCONFIG}
    else
        testconfig=$(tboolean-testconfig)
    fi 

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

tboolean-tracetest()
{
    tboolean-- --tracetest $*
}

tboolean-enum(){
   local tmp=$TMP/$FUNCNAME.exe
   clang $OPTICKS_HOME/optixrap/cu/boolean-solid.cc -lstdc++ -I$OPTICKS_HOME/optickscore -o $tmp && $tmp $*
}

tboolean-testconfig-py-()
{
    # prepare the testconfig using python 
    local fn=$1
    shift 
    local csgpath=$($fn- $* | python)
    local test_config=( 
                       mode=PyCsgInBox
                       name=$fn
                       analytic=1
                       csgpath=$csgpath
                     ) 
    echo "$(join _ ${test_config[@]})" 
}


tboolean-torchconfig()
{
    local pol=${1:-s}
    local wavelength=500
    local identity=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000

    #local photons=1000000
    local photons=100000
    #local photons=1

    local torch_config_disc=(
                 type=disc
                 photons=$photons
                 mode=fixpol
                 polarization=0,1,0
                 frame=-1
                 transform=$identity
                 source=0,0,599
                 target=0,0,0
                 time=0.1
                 radius=110
                 distance=200
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength 
               )


    local discaxial_target=0,0,0
    local torch_config_discaxial=(
                 type=discaxial
                 photons=$photons
                 frame=-1
                 transform=$identity
                 source=$discaxial_target
                 target=0,0,0
                 time=0.1
                 radius=100
                 distance=400
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength 
               )

    echo "$(join _ ${torch_config_discaxial[@]})" 
}



#tboolean-material(){ echo MainH2OHale ; }
tboolean-material(){ echo GlassSchottF2 ; }
tboolean-container(){ echo Rock//perfectAbsorbSurface/Vacuum ; }
tboolean-object(){ echo Vacuum///GlassSchottF2 ; }


tboolean-bib-box()
{
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=box      parameters=0,0,0,1000               boundary=$(tboolean-container)
                 node=box      parameters=0,0,0,100                boundary=$(tboolean-object)

                    )
     echo "$(join _ ${test_config[@]})" 
}


tboolean-bib-box-small-offset-sphere()
{
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=sphere           parameters=0,0,0,1000          boundary=$(tboolean-container)
 
                 node=${1:-difference} parameters=0,0,0,300           boundary=$(tboolean-object)
                 node=box              parameters=0,0,0,200           boundary=$(tboolean-object)
                 node=sphere           parameters=0,0,200,100         boundary=$(tboolean-object)
               )
     echo "$(join _ ${test_config[@]})" 
}

tboolean-bib-box-sphere()
{
    local operation=${1:-difference}
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)
 
                 node=$operation   parameters=0,0,0,300           boundary=$(tboolean-object)
                 node=box          parameters=0,0,0,$inscribe     boundary=$(tboolean-object)
                 node=sphere       parameters=0,0,0,200           boundary=$(tboolean-object)
               )

     echo "$(join _ ${test_config[@]})" 
}







tboolean-box-py(){ tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-box-py-(){ cat << EOP 
from opticks.dev.csg.csg import CSG  

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20" )

im = dict(poly="IM", resolution="50", verbosity="1", ctrl="0" )
#tr = dict(translate="0,0,100", rotate="1,1,1,45", scale="1,1,2")
tr = dict(scale="2,2,2", rotate="1,1,1,45")

kwa = {}
kwa.update(im)
kwa.update(tr)

box = CSG("box", param=[0,0,0,200], boundary="$(tboolean-object)", **kwa )

CSG.Serialize([container, box], "$TMP/$FUNCNAME" )
EOP
}

tboolean-sphere-py(){ tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-sphere-py-(){ cat << EOP 
from opticks.dev.csg.csg import CSG  

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20" )


im = dict(poly="IM", resolution="50", verbosity="1", ctrl="0" )
#tr = dict(scale="1,1,2")
tr = dict(translate="0,0,100", rotate="1,1,1,45", scale="1,1,2")

kwa = {}
kwa.update(im)
kwa.update(tr)

sphere = CSG("sphere", param=[0,0,0,100], boundary="$(tboolean-object)", **kwa )

CSG.Serialize([container, sphere], "$TMP/$FUNCNAME" )
EOP
}

tboolean-box-small-offset-sphere-py(){ tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-box-small-offset-sphere-py-(){ cat << EOP
from opticks.dev.csg.csg import CSG  

container = CSG("sphere",           param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution="10" )

box = CSG("box",    param=[0,0,0,200], boundary="$(tboolean-object)", rotate="0,0,1,45" )
sph = CSG("sphere", param=[0,0,0,100], boundary="$(tboolean-object)", translate="0,0,200", scale="1,1,0.5" )

object = CSG("${1:-difference}", left=box, right=sph, boundary="$(tboolean-object)", poly="IM", resolution="50" )

CSG.Serialize([container, object], "$TMP/$FUNCNAME" )
EOP
}




tboolean-bsu(){ TESTCONFIG=$(tboolean-csg-box-sphere-py union)        tboolean-- ; }
tboolean-bsd(){ TESTCONFIG=$(tboolean-csg-box-sphere-py difference)   tboolean-- ; }
tboolean-bsi(){ TESTCONFIG=$(tboolean-csg-box-sphere-py intersection) tboolean-- ; }
tboolean-csg-box-sphere-py(){ tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-csg-box-sphere-py-(){ cat << EOP 
import math
from opticks.dev.csg.csg import CSG  

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20" )
  
radius = 200 
inscribe = 1.3*radius/math.sqrt(3)

box = CSG("box", param=[0,0,0,inscribe])


rtran = dict(translate="100,0,0")
sph = CSG("sphere", param=[0,0,0,radius], **rtran)

object = CSG("${1:-difference}", left=box, right=sph, boundary="$(tboolean-object)", poly="IM", resolution="50" )

CSG.Serialize([container, object], "$TMP/$FUNCNAME" )
EOP
}




tboolean-sphere-slab(){ TESTCONFIG=$(tboolean-csg-sphere-slab 2>/dev/null)    tboolean-- ; } 
tboolean-csg-sphere-slab(){  tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-csg-sphere-slab-(){ cat << EOP 
from opticks.ana.base import opticks_main
from opticks.dev.csg.csg import CSG  
args = opticks_main()

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20" )
  
slab   = CSG("slab", param=[0,0,1,0],param1=[-500,100,0,0] )
sphere = CSG("sphere", param=[0,0,0,500] )

object = CSG("intersection", left=sphere, right=slab, boundary="$(tboolean-object)", poly="IM", resolution="50" )

CSG.Serialize([container, object], "$TMP/$FUNCNAME" )

"""

0. Works 

Why tboolean-sphere-slab raytrace is OK but tboolean-sphere-plane has directional visibility issues ?

* suspect due to "sub-objects must be closed" limitation of the  algorithm that 
  my CSG implementation is based upon: "Kensler:Ray Tracing CSG Objects Using Single Hit Intersections"

* http://xrt.wikidot.com/doc:csg

    "The [algorithm] computes intersections with binary CSG objects using the
    [nearest] intersection. Though it may need to do several of these per
    sub-object, the usual number needed is quite low. The only limitation of this
    algorithm is that the sub-objects must be closed, non-self-intersecting and
    have consistently oriented normals."

It appears can get away with infinite slab, which isnt bounded also, 
as only unbounded in "one" direction whereas half-space is much more
unbounded : in half the directions.

"""
EOP
}


tboolean-sphere-plane(){ TESTCONFIG=$(tboolean-csg-sphere-plane 2>/dev/null)    tboolean-- ; }
tboolean-csg-sphere-plane(){  tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-csg-sphere-plane-(){ cat << EOP 
from opticks.ana.base import opticks_main
from opticks.dev.csg.csg import CSG  
args = opticks_main()

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20", verbosity="0" )
  
plane  = CSG("plane",  param=[0,0,1,100] )
sphere = CSG("sphere", param=[0,0,0,500] )

object = CSG("intersection", left=sphere, right=plane, boundary="$(tboolean-object)", poly="IM", resolution="50", verbosity="1" )

CSG.Serialize([container, object], "$TMP/$FUNCNAME" )

"""

0. Polygonization looks correct
1. only see the sphere surface from beneath the plane (ie beneath z=100)
2. only see the plane surface in shape of disc from above the plane 

"""
EOP
}

tboolean-box-plane(){ TESTCONFIG=$(tboolean-csg-box-plane 2>/dev/null)    tboolean-- ; }
tboolean-csg-box-plane(){  tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-csg-box-plane-(){ cat << EOP 
from opticks.ana.base import opticks_main
from opticks.dev.csg.csg import CSG  
args = opticks_main()

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20", verbosity="0" )

plane  = CSG("plane",  param=[0,0,1,100] )
box    = CSG("box", param=[0,0,0,200]  )
object = CSG("intersection", left=plane, right=box, boundary="$(tboolean-object)", poly="IM", resolution="50", verbosity="1" )

CSG.Serialize([container, object], "$TMP/$FUNCNAME" )

"""
#. Analogous issue to tboolean-sphere-plane
"""
EOP
}



tboolean-plane(){ TESTCONFIG=$(tboolean-csg-plane 2>/dev/null)    tboolean-- ; }
tboolean-csg-plane(){  tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-csg-plane-(){ cat << EOP 
from opticks.ana.base import opticks_main
from opticks.dev.csg.csg import CSG  
args = opticks_main()

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20", verbosity="0" )

bigbox = CSG("box", param=[0,0,0,999] )
plane  = CSG("plane",  param=[0,0,1,100] )
object = CSG("intersection", left=plane, right=bigbox, boundary="$(tboolean-object)", poly="IM", resolution="50", verbosity="1" )

CSG.Serialize([container, object], "$TMP/$FUNCNAME" )

"""

#. intersecting the plane with the container, leads to coincident surfaces and a flickery mess when 
   view from beneath the plane, avoided issue by intersecting instead with a bigbox slightly 
   smaller than the container

"""

EOP
}



tboolean-cylinder(){ TESTCONFIG=$(tboolean-csg-cylinder 2>/dev/null)    tboolean-- ; }
tboolean-csg-cylinder(){  tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-csg-cylinder-(){ cat << EOP 
import numpy as np
from opticks.ana.base import opticks_main
from opticks.dev.csg.csg import CSG  
args = opticks_main()

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20", verbosity="0" )



kwa = {}

im = dict(poly="IM", resolution="50")
transform = dict(scale="1,1,0.5", rotate="1,1,1,45", translate="100,100,100" )

kwa["verbosity"] = "1" 
kwa.update(im)
kwa.update(transform)

cylinder = CSG("cylinder", param=[0,0,0,200], param1=[400,0,0,0], boundary="$(tboolean-object)", **kwa )

PCAP = 0x1 << 0  # smaller z endcap
QCAP = 0x1 << 1  

#flags = PCAP          # bottom(-z) 
#flags = QCAP           # top(+z) o
flags = PCAP | QCAP   # both 
#flags = 0             # no-endcaps 

cylinder.param1.view(np.uint32)[1] = flags 

CSG.Serialize([container, cylinder], "$TMP/$FUNCNAME" )

"""

Issue:

1. FIXED:seeing endcaps when would expect to see the sides of the cylinder
2. FIXED:Not honouring transforms
3. polygonization does not honour endcap flags, but raytrace does

Note that endcaps and insides of the cylinder look dark from inside: 
this is correct as normals are rigidly attached to geometry pointing outwards.



"""
EOP
}





tboolean-unbalanced(){   TESTCONFIG=$(tboolean-csg-unbalanced-py)          tboolean-- ; }
tboolean-csg-unbalanced-py(){ tboolean-testconfig-py- $FUNCNAME ; }
tboolean-csg-unbalanced-py-()
{
    local material=$(tboolean-material)
    local base=$TMP/$FUNCNAME 
    cat << EOP 
import math
from opticks.dev.csg.csg import CSG  
  
radius = 200 
inscribe = 1.3*radius/math.sqrt(3)

lbox = CSG("box",    param=[100,100,-100,inscribe])
lsph = CSG("sphere", param=[100,100,-100,radius])
left  = CSG("difference", left=lbox, right=lsph, boundary="$(tboolean-object)" )

right = CSG("sphere", param=[0,0,100,radius])

object = CSG("union", left=left, right=right, boundary="$(tboolean-object)", poly="IM", resolution="60" )

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution="20")

CSG.Serialize([container, object], "$base" )

# marching cubes with nx=15 again makes a mess 
# the ray trace skips the sphere 

EOP
}







tboolean-pmt(){          TESTCONFIG=$(tboolean-csg-pmt-py 2>/dev/null)     tboolean-- ; }
tboolean-csg-pmt-py-check(){ tboolean-csg-pmt-py 2> /dev/null ; }
tboolean-csg-pmt-py(){ tboolean-testconfig-py- $FUNCNAME ; }
tboolean-csg-pmt-py-()
{
    local material=$(tboolean-material)
    local base=$TMP/$FUNCNAME 
    cat << EOP 
from opticks.ana.base import opticks_main

from opticks.ana.pmt.ddbase import Dddb
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.ncsgtranslator import NCSGTranslator

from opticks.dev.csg.csg import CSG  

args = opticks_main()

g = Dddb.parse(args.apmtddpath)

lv = g.logvol_("lvPmtHemi")
tr = Tree(lv)

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution="20")

objs = []
objs.append(container)

nn = tr.num_nodes()
assert nn == 5

# 0:Pyrex
# 1:Vacuum
# 2:Cathode needs zsphere z-slicing and innerRadius CSG differencing


for i in [0]:
    node = tr.get(i)
    obj = NCSGTranslator.TranslateLV( node.lv )
    obj.boundary = "$(tboolean-object)"
    obj.meta.update(poly="IM", resolution="30")
    objs.append(obj)
pass


CSG.Serialize(objs, "$base" )

EOP
}






tboolean-interlocked(){  TESTCONFIG=$(tboolean-csg-two-box-minus-sphere-interlocked-py) tboolean-- ; }
tboolean-csg-two-box-minus-sphere-interlocked-py(){ tboolean-testconfig-py- $FUNCNAME ; }
tboolean-csg-two-box-minus-sphere-interlocked-py-()
{
    local material=$(tboolean-material)
    local base=$TMP/$FUNCNAME 
    cat << EOP 
import math
from opticks.dev.csg.csg import CSG  
  
radius = 200 
inscribe = 1.3*radius/math.sqrt(3)

lbox = CSG("box",    param=[100,100,-100,inscribe])
lsph = CSG("sphere", param=[100,100,-100,radius])
left  = CSG("difference", left=lbox, right=lsph, boundary="$(tboolean-object)" )

rbox = CSG("box",    param=[0,0,100,inscribe])
rsph = CSG("sphere", param=[0,0,100,radius])


tran = dict(translate="0,0,200", rotate="1,1,1,45", scale="1,1,1.5" )
right = CSG("difference", left=rbox, right=rsph, boundary="$(tboolean-object)", **tran)

dcs = dict(poly="DCS", nominal="7", coarse="6", threshold="1", verbosity="0")

#seeds = "100,100,-100,0,0,300"
im = dict(poly="IM", resolution="64", verbosity="0", ctrl="0" )
object = CSG("union", left=left, right=right,  boundary="$(tboolean-object)", **im )

mc = dict(poly="MC", nx="20")

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution="20" )

CSG.Serialize([container, object], "$base" )
# marching cubes with nx=15 makes a mess with this 

EOP
}



tboolean-testconfig()
{
    # token BoxInBox 
    #tboolean-bib-box
    #tboolean-bib-box-small-offset-sphere
    #tboolean-bib-box-sphere


    #tboolean-box-py
    #tboolean-sphere-py

    #tboolean-box-small-offset-sphere-py difference
    #tboolean-box-small-offset-sphere-py intersection
    #tboolean-box-small-offset-sphere-py union

    #tboolean-box-sphere-py intersection 
    #tboolean-box-sphere-py difference
    tboolean-box-sphere-py union

    #tboolean-csg-unbalanced-py
    #tboolean-csg-pmt-py 2>/dev/null

    #tboolean-csg-two-box-minus-sphere-interlocked-py
}





