tboolean-source(){   echo $(opticks-home)/tests/tboolean.bash ; }
tboolean-vi(){       vi $(tboolean-source) ; }
tboolean-usage(){ cat << \EOU

tboolean- 
======================================================

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

CsgInBox

     * requires "offsets" identifying node splits into primitives eg offsets=0,1 
     * nodes are specified in tree levelorder, trees must be perfect 
       with 1,3,7 or 15 nodes corresponding to trees of height 0,1,2,3

PyCsgInBox

     * requires csgpath identifying directory containing serialized CSG trees
       and csg.txt file with corresponding boundary spec strings


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

    op.sh  \
            $cmdline \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 1,0,0 \
            --dbganalytic \
            --test --testconfig "$(tboolean-testconfig)" \
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


tboolean-box()
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

tboolean-box-py(){ tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-box-py-(){ cat << EOP 
from opticks.dev.csg.csg import CSG  

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20" )

im = dict(poly="IM", resolution="100", verbosity="1", ctrl="0" )
box = CSG("box", param=[0,0,100,100], boundary="$(tboolean-object)", **im )

CSG.Serialize([container, box], "$TMP/$FUNCNAME" )
EOP
}



tboolean-sphere-py(){ tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-sphere-py-(){ cat << EOP 
from opticks.dev.csg.csg import CSG  

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20" )

im = dict(poly="IM", resolution="100", verbosity="1", ctrl="0" )
sphere = CSG("sphere", param=[0,0,0,100], boundary="$(tboolean-object)", **im )

CSG.Serialize([container, sphere], "$TMP/$FUNCNAME" )
EOP
}







tboolean-box-small-offset-sphere()
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

tboolean-box-small-offset-sphere-py(){ tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-box-small-offset-sphere-py-(){ cat << EOP
from opticks.dev.csg.csg import CSG  

container = CSG("sphere",           param=[0,0,0,1000], boundary="$(tboolean-container)" )

box = CSG("box",    param=[0,0,0,200], boundary="$(tboolean-object)")
sph = CSG("sphere", param=[0,0,200,100], boundary="$(tboolean-object)")

object = CSG("${1:-difference}", left=box, right=sph, boundary="$(tboolean-object)")

CSG.Serialize([container, object], "$TMP/$FUNCNAME" )
EOP
}





tboolean-box-sphere()
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

tboolean-box-sphere-py(){ tboolean-testconfig-py- $FUNCNAME $* ; } 
tboolean-box-sphere-py-(){ cat << EOP 
import math
from opticks.dev.csg.csg import CSG  

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx="20" )
  
radius = 200 
inscribe = 1.3*radius/math.sqrt(3)

box = CSG("box", param=[0,0,0,inscribe])
sph = CSG("sphere", param=[0,0,0,radius])

object = CSG("${1:-difference}", left=box, right=sph, rtranslate="100,0,0", boundary="$(tboolean-object)", poly="IM", resolution="50", seeds="0,0,0" )

CSG.Serialize([container, object], "$TMP/$FUNCNAME" )
EOP
}


tboolean-csg-two-box-minus-sphere-interlocked()
{
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=union        parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)

                      node=box          parameters=100,100,-100,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=100,100,-100,200           boundary=$(tboolean-object)

                      node=box          parameters=0,0,100,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=0,0,100,200           boundary=$(tboolean-object)
 
                      )
    echo "$(join _ ${test_config[@]})" 
}


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
right = CSG("difference", left=rbox, right=rsph, boundary="$(tboolean-object)" )

dcs = dict(poly="DCS", nominal="7", coarse="6", threshold="1", verbosity="3")

seeds = "100,100,-100,0,0,300"
im = dict(poly="IM", resolution="64", verbosity="1", ctrl="0" )

# hmm need to transform the seeds too ? 
# to avoid having to do that manually to find geometry 
# ... seeds are a pain can they be automated entirely ?
# 
# * only root node metadata currently persisted
# * perhaps just harvest the center positions of all the primitives
#   then can easily apply any global transforms
#
#

object = CSG("union", left=left, right=right, rtranslate="0,0,200", boundary="$(tboolean-object)", **im )

# log2size 7 -> size 128, ie -64:64
# log2size,theshold 7,100 ... broken, great big voids
# log2size,theshold 7,10  ... a bit crooked 
# log2size,theshold 7,1   ... a bit crooked, little different to 10
#
# log2size 8 -> size 256 ie -128:128
# log2size,theshold 8,1   ... pretty mesh, but perhaps x10 times slower than log2size 8
# log2size,theshold 8,10  ... still pretty, but still slow 

mc = dict(poly="MC", nx="20")

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution="20" )

CSG.Serialize([container, object], "$base" )
# marching cubes with nx=15 makes a mess with this 

EOP
}



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

object = CSG("union", left=left, right=right, boundary="$(tboolean-object)")

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)" )

CSG.Serialize([container, object], "$base" )

# marching cubes with nx=15 again makes a mess 
# the ray trace skips the sphere 

EOP
}








tboolean-csg-shells2-notes(){ cat << EON

* difference of spheres makes a shell
* intersection of shells makes a ring that has a twisty surface
* difference of overlapping shells makes a shell with a ringlike cut 
* union makes a compound shell with crossed surfaces 
* difference of unions of offset spheres makes a kidney bean shell 

* so far have not seen errors from height 2 trees like these


           D
       U       U
     s   s   s   s
    

EON
}



tboolean-csg-shells2()
{
    local boundary=$(tboolean-object)

    local o=200
    local i=190
    local s=50

    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=difference   parameters=0,0,0,500           boundary=$boundary

                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary

                      node=sphere       parameters=-$s,-$s,-$s,$o      boundary=$boundary
                      node=sphere       parameters=$s,$s,$s,$o         boundary=$boundary

                      node=sphere       parameters=-$s,-$s,-$s,$i      boundary=$boundary
                      node=sphere       parameters=$s,$s,$s,$i         boundary=$boundary
 
                      )

    echo "$(join _ ${test_config[@]})" 
}


tboolean-csg-shells3-notes(){ cat << EON

* https://en.wikipedia.org/wiki/Tetrahedron#Formulas_for_a_regular_tetrahedron

Tetrahedron: (1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)

* four difference of sphere bubbles centered on tetrahedron vertices

Causes errors at 3-way intersections


                  U

        U                    U

   D         D          D         D
 s   s     s   s      s   s     s   s


Hmm quite difficult to construct height 3 trees using levelorder 


EON
}


tboolean-csg-shells3()
{
    local boundary=$(tboolean-object)

    local o=200
    local i=190
    local s=100   # tetrahedron vertex distance, at 150 the shells dont overlap 
    local t=100

    local shape=sphere

    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=union        parameters=0,0,0,500           boundary=$boundary

                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary

                      node=difference   parameters=0,0,0,500           boundary=$boundary
                      node=difference   parameters=0,0,0,500           boundary=$boundary

                      node=difference   parameters=0,0,0,500           boundary=$boundary
                      node=difference   parameters=0,0,0,500           boundary=$boundary

                      node=$shape       parameters=$s,$s,$s,$o         boundary=$boundary
                      node=$shape       parameters=$s,$s,$s,$i         boundary=$boundary

                      node=$shape       parameters=$s,-$s,-$s,$o       boundary=$boundary
                      node=$shape       parameters=$s,-$s,-$s,$i       boundary=$boundary

                      node=$shape       parameters=-$t,$t,-$t,$o       boundary=$boundary
                      node=$shape       parameters=-$t,$t,-$t,$i       boundary=$boundary

                      node=$shape       parameters=-$t,-$t,$t,$o       boundary=$boundary
                      node=$shape       parameters=-$t,-$t,$t,$i       boundary=$boundary
 
                      )

    echo "$(join _ ${test_config[@]})" 
}




tboolean-csg-shells3-alt()
{
    local boundary=$(tboolean-object)

    local o=200
    local i=190
    local s=100 

    local shape=sphere

    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=difference   parameters=0,0,0,500           boundary=$boundary

                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary


                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary


                      node=$shape       parameters=$s,$s,$s,$o         boundary=$boundary
                      node=$shape       parameters=$s,-$s,-$s,$o       boundary=$boundary

                      node=$shape       parameters=-$s,$s,-$s,$o       boundary=$boundary
                      node=$shape       parameters=-$s,-$s,$s,$o       boundary=$boundary
                     
 
                      node=$shape       parameters=$s,-$s,-$s,$i       boundary=$boundary
                      node=$shape       parameters=$s,$s,$s,$i         boundary=$boundary

                      node=$shape       parameters=-$s,$s,-$s,$i       boundary=$boundary
                      node=$shape       parameters=-$s,-$s,$s,$i       boundary=$boundary
 
                      )

    echo "$(join _ ${test_config[@]})" 
}


tboolean-csg-triplet()
{
    local material=$(tboolean-material)
    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=intersection   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=box          parameters=0,0,0,150           boundary=$(tboolean-object)
                      node=sphere       parameters=0,0,0,200           boundary=$(tboolean-object)
 
                      )

    echo "$(join _ ${test_config[@]})" 
}

tboolean-csg-triplet-new()
{ 
    local csgpath=$($FUNCNAME- | python)
    local test_config=( 
                       name=$FUNCNAME
                       analytic=1
                       csgpath=$csgpath
                     ) 
    echo "$(join _ ${test_config[@]})" 
    #np.py $csgpath/0.npy
    #np.py $csgpath/1.npy
    #NCSGTest $csgpath
}
tboolean-csg-triplet-new-()
{
    local material=$(tboolean-material)
    local base=$TMP/$FUNCNAME 
    cat << EOP 
from opticks.dev.csg.csg import CSG  

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)" )
   
s = CSG("sphere", param=[0,0,0,200])
b = CSG("box", param=[0,0,0,150])
sib = CSG("intersection", left=s, right=b, boundary="$(tboolean-object)")
smb = CSG("difference",   left=s, right=b, boundary="$(tboolean-object)")

CSG.Serialize([container, smb], "$base" )

EOP
}





tboolean-csg-four-box-minus-sphere()
{
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local radius=200
    local s=100
    #local s=200  # no error when avoid overlap between subtrees 

    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=union        parameters=0,0,0,500           boundary=$(tboolean-object)

                      node=union        parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=union        parameters=0,0,0,500           boundary=$(tboolean-object)

                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)

                      node=box          parameters=$s,$s,-$s,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=$s,$s,-$s,200           boundary=$(tboolean-object)

                      node=box          parameters=-$s,-$s,-$s,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=-$s,-$s,-$s,200           boundary=$(tboolean-object)

                      node=box          parameters=$s,-$s,$s,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=$s,-$s,$s,200           boundary=$(tboolean-object)

                      node=box          parameters=-$s,$s,$s,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=-$s,$s,$s,200           boundary=$(tboolean-object)
 
                      )

    echo "$(join _ ${test_config[@]})" 


}



tboolean-testconfig()
{
    #tboolean-box
    #tboolean-box-py

    #tboolean-sphere-py

    #tboolean-box-small-offset-sphere difference

    #tboolean-box-small-offset-sphere-py difference
    #tboolean-box-small-offset-sphere-py intersection
    #tboolean-box-small-offset-sphere-py union

    #tboolean-box-sphere intersection 
    #tboolean-box-sphere union
    #tboolean-box-sphere difference

    #tboolean-box-sphere-py intersection 
    #tboolean-box-sphere-py difference
    #tboolean-box-sphere-py union


    #tboolean-csg-two-box-minus-sphere-interlocked
    tboolean-csg-two-box-minus-sphere-interlocked-py

    #tboolean-csg-unbalanced-py



    #tboolean-csg-four-box-minus-sphere
    #tboolean-csg-shells2
    #tboolean-csg-shells3
    #tboolean-csg-shells3-alt

    #tboolean-csg


    #tboolean-csg-triplet
    #tboolean-csg-triplet-new
}

