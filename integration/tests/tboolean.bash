#!/usr/bin/env bash
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


tboolean-source(){   echo $BASH_SOURCE ; }
tboolean-vi(){       vi $(tboolean-source) ; }
tboolean-usage(){ cat << \EOU

tboolean- 
======================================================


Using opticks-tboolean-shortcuts
-----------------------------------
    
Define the shortcut bash functions with::

   opticks-tboolean-shortcuts
   type opticks-tboolean-shortcuts  # see what they are 


*ts* aligned bi-simulation::

   ts 10
   ts box   
   ts truncate -D   # -D for running under debugger

*tv* *tv4* visualize geometry and event propagation:: 

   tv 10
   tv box
   tv truncate

*ta* load and analyse propagation events in ipython::
   
   ta 10      
   ta box
   ta truncate



viewpoint control
---------------------

::

    tv 34 --size 1280,720,1 --eye 1,0,0 --up 0,-1,0 

        # view from +X, with +Y up puts -Z to the right
        # view from +X, with -Y up puts +Z to the right

        Useful for making screenshots because changing the 
        initial view changes V rotation axis.




common issues
-------------------

ts running gives a commandline error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ts box ::

    2019-06-24 14:06:37.188 INFO  [205521] [OpticksHub::loadGeometry@517] --test modifying geometry
    2019-06-24 14:06:37.188 ERROR [205521] [BStr::ekv_split@620] ignoring malformed kv [--torch]
    2019-06-24 14:06:37.188 ERROR [205521] [BStr::ekv_split@621] line [--torch]

This means there is a problem with the python geometry setup code in tboolean-box, to find whats the problem
use the "--chk" option which just parses the python config setup and serializes the geometry into a directory
beneath $TMP::

   ts box --chk 



tboolean testing
-------------------

*tboolean* runs Opticks executables in "--test" mode which 
constructs simple GGeoTest "Russian-doll" geometries specified by python 
lists of solids (CSG instances), with boundaries specifying omat/osur/isur/imat 
that have strictly consistent omat/imat pairs between self and parent.
Each solid can be a simple primitive or a CSG node tree.

Most of the functions follow the below pattern:

tboolean-name--
    emit to stdout python source description of the geometry

tboolean-name-
    pipe the above source to python, which writes a serialization of the geometry 
    into directory $TMP/tboolean-name-- and emits to stdout a testconfig string 
    of the below form. Currently only mode PyCsgInBox is supported within tboolean.
    ::

       analytic=1_csgpath=/tmp/blyth/opticks/tboolean-name--_mode=PyCsgInBox_outerfirst=1_name=tboolean-name--

tboolean-name
    runs the above geometry serialization, capturing the testconfig string and passing it
    to the op.sh script via the tboolean-- function which runs the opticks executables 
    in --test mode constructing the geometry, performing the simulation and writing 
    events to file

tboolean-name-g  
    some *name* have the g variant which tests geometry construction via tboolean-g-

tboolean-name-a
    invokes tboolean-ana- on the events written by the tboolean-name function 

tboolean-name-p
    invokes tboolean-py- on the events written by the tboolean-name function 

tboolean-name-ip
    invokes tboolean-ipy- (ipython) on the events written by the tboolean-name function 
    which jumps into interactive python with the event loaded



FEATURE : emitconfig cfg4 chisq too good as not indep
------------------------------------------------------

With *emitconfig* OK/G4 samples are not independant, 
as the input photons are identical. BUT the chisq comparison 
machinery was setup for comparing indep samples. 

For history aligned photons a direct deviation comparison 
is done and reported in addition to the seq list.

Note that even without RNG alignment there is some 
"accidental" history aligment, especially for the top of the pops
categories which do not consume randoms 

::

    .                pflags_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             600000    600000         2.28/9 =  0.25  (pval:0.986 prob:0.014)  
    0000             1880    337842    337692             0.03        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO|BT|SA
    0001             1080    215778    215777             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO|SA
    0002             1480     23427     23472             0.04        0.998 +- 0.007        1.002 +- 0.007  [3 ] TO|BR|SA
    0003             1c80     22170     22282             0.28        0.995 +- 0.007        1.005 +- 0.007  [4 ] TO|BT|BR|SA
    0004             10a0       255       241             0.40        1.058 +- 0.066        0.945 +- 0.061  [3 ] TO|SA|SC
    0005             18a0       160       174             0.59        0.920 +- 0.073        1.087 +- 0.082  [4 ] TO|BT|SA|SC
    0006             1808       152       150             0.01        1.013 +- 0.082        0.987 +- 0.081  [3 ] TO|BT|AB
    0007             1ca0       101        89             0.76        1.135 +- 0.113        0.881 +- 0.093  [5 ] TO|BT|BR|SA|SC
    0008             1c20        67        65             0.03        1.031 +- 0.126        0.970 +- 0.120  [4 ] TO|BT|BR|SC
    0009             1008        31        34             0.14        0.912 +- 0.164        1.097 +- 0.188  [2 ] TO|AB
    0010             14a0        10        11             0.00        0.909 +- 0.287        1.100 +- 0.332  [4 ] TO|BR|SA|SC
    0011             1c08         6        13             0.00        0.462 +- 0.188        2.167 +- 0.601  [4 ] TO|BT|BR|AB
    0012             1408         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|BR|AB



TODO 
------

Prime objective is Automation of these tests

* test compute mode operation, verify same as interop, 
  use compute mode for test harness running of lists of tests 

* rationalize where to get NGeoTestConfig from, 
  when using loaded test evt, must get from event, 
  when creating the event need to get from Opticks ?

  Perhaps can make the accessors from Opticks private and always
  set it onto the event in Opticks::makeEvent ?

  Need this for removing the emitconfig.deltashift kludge in OpticksEventAna

* detect russian-doll or sibling-solids based on boundaries, 
  and do the CTestDetector conversion to G4 accordingly  

* IDEA : perhaps a op.py that parses arguments and writes a config directory of 
  json for an Opticks invokation... with the path to the directory being 
  taken as a standard Opticks argument.

* migrate non-GGeoTest/non-NCSG tboolean-funcs into tgltf or elsewhere

* check the test surfaces:  perfectAbsorbSurface, perfectDetectSurface, perfectSpecularSurface, perfectDiffuseSurface

* check, is torchconfig still working, what happens with both emitconfig and torchconfig active ?


Configuring Photon Sources 
-----------------------------

There are two approaches, the older manual *torchconfig* which is 
defined separately from geometry and the newer more automated *emitconfig*,
which attaches emission properties to solids in the geometry.

A major difference is that *emitconfig* does the photon generation 
once only on the CPU with these photons being copied to GPU, thus precisely 
the same input photons are used for both Opticks and G4 simulations, 
whereas with *torchconfig* parameters in the form of gensteps 
are passed to the GPU where the generation is done, as well as
a CPU side G4 generation.   
*torchconfig* thus entails keeping two generation implementations in step, 
and does not yield exactly the same photons.

emitconfig
~~~~~~~~~~~~

Emission of photons from the surface of any CSG primitive is configured 
with the *emit* attribute. emit=1/-1 emits outwards/inwards and emit=0 
switches off emission.  The details of the emission can be 
controlled with the emitconfig attribute, which defaults to:: 

    tboolean-emitconfig(){ echo "photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1" ; }  

NB the kv delimiter is ":" to allow incorporation into GGeoTest config without interference 

The sheetmask configures which sheets of a solid emit (0x1 : sheet 0 only, 0x3f : sheets 0:6 )
eg a cube has 6 sheets, a truncated cone has 3 sheets (2 endcaps + body)
(This is not yet implemented for all primitives, eg cone trips an assert) 

NB *emitconfig* attributes can be applied to all solids without having any effect, it is necessary 
   to have in addition an *emit=1/-1* as well (*emit=0* is the default) 


opticksnpy/NEmitPhotonsNPY
    NEmitPhotonsNPY::init creates input photon buffer for an NCSG instance,  
    using points and normals from NNode::generateParPoints which 
    uses shape specializations of NNode::par_posnrm_model( pos, nrm, sheet, fu, fv );


torchconfig
~~~~~~~~~~~

opticksnpy/TorchStepNPY 
    parses config string and encodes params into genstep buffer for copying to GPU 

optixrap/cu/torchstep.h
    GPU side OptiX generation of photons from the genstep buffer, this 
    works by throwing two random numbers (ranges from zeaz:zenithazimuth)
    that are used in different ways based upon genstep params  

cfg4/CTorchSource
    CPU side G4 generation of photons from the genstep buffer, actually the TorchStepNPY instance


Deficiencies of torchconfig:

1. get bizarre results when the torch positions are outside the container that 
   defines the Opticks domain, forcing manually tweaking of the 
   torch positions for different containers : the problem with this
   is that it is then not possible to reproduce a prior torch setup via a 
   torchname for example. 

   *emitconfig* solves this issue by decoupling from position, but 
   is currently limited to all sheet/face generation.



Relevant Opticks Options
----------------------------

--okg4
    perform "bi-simulation" : both Opticks and G4 geometries are constructed and simulations
    performed in both, with separate events in OpticksEvent format being written to file

--load 
    inhibits performing the simulation, instead a prior one is loaded. If --okg4 
    was used, both G4 and Opticks events are loaded.

--vizg4/--vizopticks
    used together with --load these options pick which event to vizualize 

-D
    runs Opticks executable within debugger, lldb or gdb 

--testauto
    modifies test geometry emitconfig and boundaries to simplify photon histories
    allowing seqmap asserts to check NCSGIntersect positions, see below section for details

--anakey
    key identifies a python script to analyse the created events, this is typically 
    used with --okg4 to make chisq comparisons of bi-simulated Opticks and G4 events, 
    the script is run by OpticksAna and communicates with Opticks C++ just via
    the return code (0x0 - 0xFF)

--anakeyargs
    provides extra args to the anakey tboolean.py script which is run internally by OpticksAna
    for example to modify the chisq cut  "--c2max_0.5"  
    (note the encoding of spaces with underscores)

--noab/--nosc/--nore
    switch off absorb/scattering/reemission in all materials, see GMaterialLib 
    these options enable great simplification of photon histories

--xxab/--xxsc/--xxre
    enhance absorb/scattering/reemission in all materials, see GMaterialLib 

--fxab/--fxsc/--fxre
    set particular values of absorb/scattering/reemission in all materials, see GMaterialLib 

--tracetest 
    CAUTION: this has not been exercised recently, so probably not working

    with tracetest option only a single intersect is
    done using oxrap/cu/generate.cu:tracetest and a special 
    format of the photon buffer is used, for analysis by ana/tboolean.py 
    Note that in tracetest mode the record buffer filling 
    is not implemented so the visualization 
    of photon paths is not operational.


testauto details
-------------------

Note that testauto operates by changing boundaries to simplify 
the photon histories in order to allow intersect SDF  
(signed distance function) geometry checks.  
Because of this simplification the chisq comparisons are 
somewhat of a "cheat". Thus to validate it is necessary 
to do both with and without "--testauto" runs.
The without being in order to check photon histories. 

TODO: the intersects positions of any seqmap matches are still being
checked in non testauto, but the stats are probably going to be zero. 
Perhaps switch off ? 

IDEA: seqmap wildcards "*,SA:0" to check all SA arrive at tree 0 


Example::

    tboolean-;tboolean-zsphere0 --okg4 --testauto -D

    tboolean-;tboolean-zsphere0-a
         # subsequent OpticksEventAna run uses the metadata that travels with the evt 


1. default auto configuration resides in opticks/ana/base.py 
2. opticks/analytic/csg.py CSG.Serialize writes the testconfig string 
   which is picked up by bash and given to Opticks (OKTest or OKG4Test)
   as the --testconfig option
3. testconfig string is parsed by NGeoTestConfig
4. OpticksRun::annotateEvent OpticksEvent::setTestConfigString persist the 
   testconfig with events 
5. OpticksEvent::getTestConfig eg from loaded events can reparses the string
   into NGeoTestConfig
  

Workflow Examples
--------------------

::

    tboolean-;

    tboolean-torus --okg4 -D --dbgsurf
       ## bi-simulation, writing events 

    tboolean-torus --okg4 --load --vizg4
       ## visualize the G4 evt 

    tboolean-torus-a
       ## OpticksEventCompareTest OR other such exe

    tboolean-torus-a --vizg4 
       ## load the G4 event, for dumping etc..



Mostly Working (Sep 1, 2017) Other than those marked
---------------------------------------------------------

* some have poor/missing polygonizations
* all should have an OK raytrace, unless marked otherwise below

  
tboolean-box
tboolean-box3
tboolean-truncate
tboolean-cone
tboolean-prism
tboolean-icosahedron
tboolean-cubeplanes
tboolean-trapezoid
tboolean-uniontree
tboolean-parade
tboolean-complement
tboolean-zsphere1
tboolean-zsphere2
tboolean-union-zsphere
tboolean-difference-zsphere
tboolean-hybrid
tboolean-hyctrl
tboolean-uncoincide
tboolean-disc
tboolean-esr
    speckle in the hole
tboolean-sc
    constructs single solid obtained from GDML parsing 
tboolean-positivize
    NOT WORKING

tboolean-bsu
tboolean-bsd
tboolean-bsi
    boxsphere union, intersection, difference 

tboolean-segment
tboolean-cysegment
tboolean-cyslab
    NOT WORKING
tboolean-undefined
    NOT WORKING
   
tboolean-torus
tboolean-hyperboloid
tboolean-cubic
    NOT WORKING

tboolean-12
    PMT Neck Modelling With Hyperboloid, compared to cylinder - torus
    
tboolean-ellipsoid
tboolean-spseg
    NOT WORKING


tboolean-sphereslab
tboolean-sphereplane
    exhibits view dependent geometry, from use of unbounded plane in CSG 


tboolean-boxplane
    exhibits view dependent geometry, from use of unbounded plane in CSG 
tboolean-plane
tboolean-cy
tboolean-cylinder
    cylinder with 3 cylinders subtracted 
tboolean-fromstring
    geometry parsed from GDML string
tboolean-unbalanced
tboolean-deep

tboolean-0q
tboolean-0
    NOT WORKING
   
tboolean-gds0
tboolean-oav
    includes ribs at base
tboolean-iav
tboolean-sst
    NOT WORKING 
tboolean-gds
tboolean-pmt
    missing poly for several solids

tboolean-sstt
tboolean-sstt2
    trapezoid
tboolean-ntc
    pool cover flat lozenge

tboolean-p0
tboolean-dd
    revived detdesc to NCSG conversion python, possible z-offset/halving issue with 5th solid (cylinder)
 
tboolean-interlocked
    several box minus sphere, with gpuoffset



Not Working (Sep 1, 2017)
-----------------------------

tboolean-positivize
     AttributeError: 'CSG' object has no attribute 'subdepth'  

tboolean-cyslab
     AttributeError: type object 'Primitive' has no attribute 'deltaphi_slab_segment'

tboolean-spseg
    AttributeError: type object 'Primitive' has no attribute 'deltaphi_slab_segment'

tboolean-undefined
     2017-09-01 14:46:03.887 FATAL [2268207] [*NCSG::import_primitive@887] NCSG::import_primitive TYPECODE NOT IMPLEMENTED  idx 0 typecode 26 csgname undefined

tboolean-cubic
     invisible raytrace

tboolean-0q
tboolean-0
     Assertion failed: (join2 != JOIN_COINCIDENT), function znudge_umaxmin, file /Users/blyth/opticks/opticksnpy/NNodeNudger.cpp, line 413.

tboolean-sst
     Assertion failed: (join2 != JOIN_COINCIDENT), function znudge_umaxmin, file /Users/blyth/opticks/opticksnpy/NNodeNudger.cpp, line 413.



EOU
}

tboolean-env(){      olocal- ;  }
tboolean-dir(){ echo $(dirname $(tboolean-source)) ; }
tboolean-cd(){  cd $(tboolean-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tboolean-tag(){  echo ${TAG:-1} ; }
tboolean-det(){  echo boolean ; }
tboolean-src(){  echo torch ; }
tboolean-args(){ echo  --det $(tboolean-det) --src $(tboolean-src) ; }

tboolean-ls-(){ grep TESTCONFIG= $BASH_SOURCE ; }
tboolean-ls(){ $FUNCNAME- | perl -ne 'm/(\S*)\(\)/ && print "$1\n" ' -   ; }


tboolean-seqhis()
{ 
   case $1 in 
     "TO MI"            ) echo 0x3d ;;
     "TO AB"            ) echo 0x4d ;;
     "TO SC BT BR BT SA") echo 0x8cbc6d ;;
     "TO SC BT BT SA"   ) echo 0x8cc6d  ;;
     "TO BT BT SA"      ) echo 0x8ccd  ;;
     "TO SC SA"         ) echo 0x86d  ;;
     "TO SR SA"         ) echo 0x8ad  ;;
     "TO BR MI"         ) echo 0x3bd  ;;
     "TO BR SA"         ) echo 0x8bd  ;;
   esac
}



tboolean-a-(){

    #lldb OpticksEventAnaTest --   --torch  --tag $(tboolean-tag) --cat $(tboolean-testname)  --dbgnode 0  --dbgseqhis $(tboolean-dbgseqhis) $* 
          OpticksEventAnaTest      --torch  --tag $(tboolean-tag) --cat $(tboolean-testname)  --dbgnode 0  --dbgseqhis $(tboolean-dbgseqhis) $* 
}

tboolean-ana-(){
    local msg="$FUNCNAME :"

    local testname=${TESTNAME}
    [ -z "$testname" ] && echo $msg missing TESTNAME && sleep 1000000

    #local dbgseqhis=$(tboolean-seqhis "TO MI")
    #local dbgseqhis=$(tboolean-seqhis "TO BR MI")
    local dbgseqhis=$(tboolean-seqhis "TO BR SA")

    #local exe=OpticksEventAnaTest 
    local exe=OpticksEventCompareTest 
    #local exe=OpticksEventDumpTest 

    gdb --args $exe --torch  --tag $(tboolean-tag) --cat $testname  --pfx $testname --dbgnode 0  --dbgseqhis $dbgseqhis $* 
}

# TODO: how to pick a profile without being explicit about it ? so this doesnt depend on having it 






tboolean-testname-notes(){ cat << EON
$FUNCNAME
=========================

1. TESTNAME if defined
2. tboolean-proxy-LV if LV defined and an integer 0,1,2,...
3. tboolean-LV if LV defined and not an integer
4. tboolean-boxsphere the default 

EON
}
tboolean-testname-default(){ echo tboolean-boxsphere ; }
tboolean-testname()
{
    local testname
    if [ -n "$TESTNAME" ]; then
        testname=${TESTNAME}
    elif [ -n "$LV" ]; then 
        if [[ $LV =~ ^[0-9]+ ]]; then
           testname="tboolean-proxy-$LV"
        else
           testname="tboolean-$LV"
        fi 
    else
        testname=$(tboolean-testname-default)
    fi 
    echo $testname
}


tboolean-funcname-notes(){ cat << EON

* LV integers handled by tboolean-proxy
* string names such as box by tboolean-box for example

EON
}

tboolean-funcname()
{
    local funcname
    if [ -n "$LV" ]; then 
        if [[ $LV =~ ^[0-9]+ ]]; then
           funcname="tboolean-proxy"
        else
           funcname="tboolean-$LV"
        fi 
    else
        funcname=$(tboolean-testname-default)
    fi 
    echo $funcname
}


tboolean-testconfig(){ echo $TESTCONFIG ; }
tboolean-testconfig-antipattern()
{
    # This "bend-over-backwards" pattern is not a good idea, better not
    # to handle expected missing config, just raise an error.
    # Simpler is better because see and identify problems sooner/easier.

    local testconfig
    local testname
    if [ -n "$TESTCONFIG" ]; then
        testconfig=${TESTCONFIG}
    else
        testname=$(tboolean-testname-default)
        testconfig=$($testname- 2>/dev/null)
    fi 
    echo $testconfig
}

tboolean-torchconfig()
{
    local torchconfig
    if [ -n "$TORCHCONFIG" ]; then
        torchconfig=${TORCHCONFIG}
    else
        torchconfig=$(tboolean-torchconfig- 2>/dev/null)
    fi 
    echo $torchconfig
}

tboolean-dbgseqhis()
{
    local dbgseqhis
    if [ -n "$DBGSEQHIS" ]; then
        dbgseqhis=${DBGSEQHIS}
    else
        dbgseqhis=$(tboolean-seqhis "TO SR SA")
    fi 
    echo $dbgseqhis
}


tboolean-info(){ cat << EOI

$FUNCNAME
==================


BASH_VERSION         : $BASH_VERSION
TESTNAME             : $TESTNAME
TESTCONFIG           : $TESTCONFIG
TORCHCONFIG          : $TORCHCONFIG

tboolean-testname    : $(tboolean-testname)
tboolean-testconfig  : $(tboolean-testconfig)
tboolean-torchconfig : $(tboolean-torchconfig)


EOI
}














tboolean-lv-notes(){ cat << EON
$FUNCNAME
===================

What to run depends on envvars TESTNAME and LV 

EON
}

tboolean-lv()
{
   local msg="=== $FUNCNAME :"
   local funcname=$(tboolean-funcname)
   local testname=$(tboolean-testname)
   local RC
   local cmdline="$*"
   echo $msg $testname cmdline $cmdline

   if [ "${cmdline/--ip}" != "${cmdline}" ]; then
       TESTNAME=$testname tboolean-ipy- $* 
   elif [ "${cmdline/--py}" != "${cmdline}" ]; then
       TESTNAME=$testname tboolean-py- $* 
   elif [ "${cmdline/--chk}" != "${cmdline}" ]; then
       ${funcname}-
   elif [ "${cmdline/--oktest}" != "${cmdline}" ]; then
       $funcname $* --nog4propagate  
   elif [ "${cmdline/--noalign}" != "${cmdline}" ]; then
       $funcname --okg4test  $*   
   else
       $funcname --okg4test --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $*   
       RC=$?
   fi 
   echo $msg $funcname RC $RC
   return $RC
}



tboolean-pfx(){ echo ${PFX:-$TESTNAME} ; }
tboolean-cat(){ echo ${CAT:-$TESTNAME} ; }

tboolean-ipy-(){ ${OPTICKS_IPYTHON:-ipython} -i --pdb -- $(which tboolean.py) --cat $(tboolean-cat) --pfx $(tboolean-pfx) --tag="$(tboolean-tag)" $* ; }
tboolean-py-(){                               tboolean.py --cat $(tboolean-cat) --pfx $(tboolean-pfx) --tag="$(tboolean-tag)"  $* ; }
tboolean-m-(){  metadata.py --cat $(tboolean-cat) --pfx $(tboolean-pfx) --tag="$(tboolean-tag)" ; }
tboolean-g-(){  lldb -- CTestDetectorTest --test --testconfig "$TESTCONFIG" $* ; }

tboolean-eye(){ echo ${EYE:-1,0,0} ; }
tboolean-up(){  echo ${UP:-0,0,1} ; }


tboolean--(){

    #tboolean-

    local msg="=== $FUNCNAME :"
    local cmdline=$*

    local stack=2180  # default

    local testconfig=$(tboolean-testconfig)
    local torchconfig=$(tboolean-torchconfig)

    tboolean-info
    [ "$testconfig" == "" ] && echo $msg no testconfig : try ${testname}- && return 255 

    o.sh  \
            $cmdline \
            --envkey \
            --rendermode +global,+axis \
            --geocenter \
            --stack $stack \
            --eye $(tboolean-eye) \
            --up $(tboolean-up) \
            --test \
            --testconfig "$testconfig" \
            --torch \
            --torchconfig "$torchconfig" \
            --torchdbg \
            --tag $(tboolean-tag) \
            --anakey tboolean \
            --args \
            --save 


    RC=$?
    echo $FUNCNAME RC $RC

    cat << EON > /dev/null
            --dbganalytic \
            --dbgemit \
            --dumpenv \
            --strace \
            --args \
            --timemax $tmax \
            --animtimemax $tmax \
            --pfx $(tboolean-pfx) \
            --cat $(tboolean-cat) \



EON

    return $RC 
}

tboolean-tracetest()
{
    tboolean-- --tracetest $*
}

tboolean-enum(){
   local tmp=$TMP/$FUNCNAME.exe
   clang $OPTICKS_HOME/optixrap/cu/boolean-solid.cc -lstdc++ -I$OPTICKS_HOME/optickscore -o $tmp && $tmp $*
}



tboolean-pol(){ echo s ; }
tboolean-wavelength(){ echo 500 ; }
tboolean-photons(){ echo 100000 ; }
tboolean-identity(){ echo 1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000 ; }


# sheetmask : eg a cube has 6 sheets, a truncated cone has 3 sheets (2 endcaps + body)
#    0x1  : sheet 0 only
#    0x3f : sheets 0:6 

tboolean-emitconfig(){ echo "photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1" ; }  


tboolean-torchconfig-disc()
{
    local source_default=0,0,599
    local radius_default=300
    local photons_default=100000

    local source_=${1:-$source_default}
    local radius=${2:-$radius_default}
    local photons=${3:-$photons_default}

    local torch_config=(
                 type=disc
                 photons=$photons
                 mode=fixpol
                 polarization=1,1,0
                 frame=-1
                 transform=$(tboolean-identity)
                 source=${source_}
                 target=0,0,0
                 time=0.0
                 radius=${radius}
                 distance=200
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$(tboolean-wavelength)
               )
    echo "$(join _ ${torch_config[@]})" 
}


tboolean-torchconfig-discaxial()
{
    local discaxial_target=0,0,0
    local torch_config=(
                 type=discaxial
                 photons=$(tboolean-photons)
                 frame=-1
                 transform=$(tboolean-identity)
                 source=$discaxial_target
                 target=0,0,0
                 time=0.1
                 radius=100
                 distance=400
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$(tboolean-wavelength)
               )
   echo "$(join _ ${torch_config[@]})" 
}

tboolean-torchconfig-sphere()
{
    local torch_config=(
                 type=sphere
                 photons=$(tboolean-photons)
                 frame=-1
                 transform=$(tboolean-identity)
                 source=0,0,0
                 target=0,0,1
                 time=0.1
                 radius=100
                 distance=400
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$(tboolean-wavelength)
               )
   echo "$(join _ ${torch_config[@]})" 
}



tboolean-torchconfig-()
{
    tboolean-torchconfig-disc
    #tboolean-torchconfig-discaxial
}



#tboolean-material(){ echo MainH2OHale ; }
tboolean-material(){ echo GlassSchottF2 ; }
tboolean-container(){ echo Rock//perfectAbsorbSurface/Vacuum ; }
tboolean-testobject(){ echo Vacuum///GlassSchottF2 ; }




tboolean-strace()
{
    local pid=$(pidof OKG4Test)
    local cmd="strace  -o /tmp/strace.log -e open -p$pid"
    echo $cmd
    eval $cmd
}



tboolean-proxy-lvidx(){    echo ${LV:--1} ; }
tboolean-proxy-name(){  echo tboolean-proxy-$(tboolean-proxy-lvidx) ; }

tboolean-proxy-pdb(){ ${OPTICKS_IPYTHON:-ipython} --pdb $(which tboolean.py) -i -- --tag 1 --tagoffset 0 --cat $(tboolean-cat) --pfx $(tboolean-pfx) --src torch ; }
tboolean-proxy-ip(){ TESTNAME=$(tboolean-proxy-name) tboolean-ipy- $* ; } 
tboolean-proxy-p(){ TESTNAME=$(tboolean-proxy-name) tboolean-py- $* ; } 
tboolean-proxy-a(){ TESTNAME=$(tboolean-proxy-name) tboolean-ana- $* ; } 
tboolean-proxy(){ TESTNAME=$(tboolean-proxy-name) TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; RC=$? ; echo $FUNCNAME RC $RC ; return $RC ;  } 
tboolean-proxy-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-proxy--(){ cat << EOP 
import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0"
args = opticks_main(csgname="$(tboolean-proxy-name)", autoemitconfig=autoemitconfig)


# see nbox::par_posnrm_model

# 0x0 all sheets
# 0x1 -Z
# 0x2 +Z  
# 0x3 -Z and +Z
# 0x3f all 6 

#emitconfig = "photons:10000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.35,umax:0.65,vmin:0.35,vmax:0.65" 
emitconfig = "photons:10000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x2,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 

CSG.kwa = dict(poly="IM",resolution=20, verbosity=0, ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1, containerautosize=1 ) 

box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2", proxylv=$(tboolean-proxy-lvidx) )

CSG.Serialize([container, box], args )
EOP
}

tboolean-proxy-notes(){ cat << EON
$FUNCNAME
=======================

As changing the LV envvar value changes geometry it is necessary to 
for it to change the geometry and event directories to avoid presenting 
events from one geometry with another geometry. 

To see a list of the lvidx indices and names use::

   GMeshLibTest --envkey 
   
When LV envvar is defined and is an integer the tboolean-proxy functions 
are automatically invoked, rather than box etc..::

   LV=20 tboolean.sh             # simulates, visualizes, saves and analyses 
   LV=20 tboolean.sh --strace    # with monitoring of all opened files, log analysed with strace.py
   LV=20 tboolean.sh --ip        # analyse previously saved propagations

Load the event and visualize the propagation::

   LV=20 tboolean.sh --load         

For subsequent analysis of the events written by the above::

   LV=20 tboolean-proxy-p        # python analysis       
   LV=20 tboolean-proxy-ip       # ipython analysis  
   LV=20 tboolean-proxy-a        # C++ OpticksEventCompare analysis

EON
}




tboolean-boxrot-notes(){ cat << EON

Most of the tboolean- tests follow the same pattern as tboolean-boxrot. 

tboolean-boxrot--
   emits python text to stdout 

tboolean-boxrot-
   pipes the text from tboolean-boxrot-- to python which serializes the geometry 
   into a directory and emits to stdout the config including the directory path

tboolean-boxrot 
   TESTCONFIG input envvar captures the output from tboolean-boxrot- and passes this to tboolean-- 
   with launches Opticks executables with --test option. Opticks reads the TESTCONFIG and deserializes
   the python defined geometry   

EON
}

tboolean-boxrot(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; } 
tboolean-boxrot-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-boxrot--(){ cat << EOP 
import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.polyconfig import PolyConfig
from opticks.analytic.csg import CSG  
from opticks.analytic.glm import rotate  


# 0x3f is all 6 
autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0"
args = opticks_main(csgname="${FUNCNAME/--}", autoemitconfig=autoemitconfig)

emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 

CSG.kwa = dict(poly="IM",resolution=20, verbosity=0, ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1 )  # no param, container=1 switches on auto-sizing

box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2"  )

#degrees = 45
#degrees = 0.135
degrees = 1
box.transform = rotate([1,0,0,degrees])

CSG.Serialize([container, box], args )
EOP
}








tboolean-box-dbg(){ ${OPTICKS_IPYTHON:-ipython} --pdb $(which tboolean.py) -i -- --tag 1 --tagoffset 0 --det tboolean-box --src torch ; }

tboolean-box-ip(){ TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; } 
tboolean-box-p(){  TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-box-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-box(){ 
   local msg="=== $FUNCNAME :"
   $FUNCNAME- 
   local testconfig=$($FUNCNAME- 2>/dev/null)
   echo $msg testconfig $testconfig  
   TESTNAME=$FUNCNAME TESTCONFIG=$testconfig tboolean-- $* 
 } 
tboolean-box-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-box--(){ cat << EOP 
import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

# 0x3f is all 6 
autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0"
args = opticks_main(csgname="${FUNCNAME/--}", autoemitconfig=autoemitconfig)

#emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
#emitconfig = "photons:1,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 

CSG.kwa = dict(poly="IM",resolution=20, verbosity=0, ctrl=0, containerscale=3.0, containerautosize=1, emitconfig=emitconfig  )

container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1 ) 

box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2"  )

CSG.Serialize([container, box], args )
EOP
}


tboolean-box-testauto-all-sr()
{
    tboolean-
    tboolean-box --okg4 --testauto --noab --nosc 
}


tboolean-box-notes(){ cat << EON

$FUNCNAME
============================

tboolean-;tboolean-box --okg4 --testauto
    PASS
tboolean-;tboolean-box --okg4 
    PASS



It now seems necessary for containerautosize=1 to be set 
for containers with no param.


CSG.Serialize writes separate directories for each tree containing src buffers::

    [blyth@localhost tboolean-box]$ np.py 0
    /tmp/tboolean-box/0
    . :                                                 0/srcidx.npy :               (1, 4) : 6c5c9fc27d1d5e44e98016bdef9aaba1 : 20190607-2129 
    . :                                               0/srcnodes.npy :            (1, 4, 4) : e279d8f47aa14224674129cf6007768d : 20190607-2129 
    . :                                          0/srctransforms.npy :            (1, 4, 4) : ead0cc88939337cc91c7e7432bf1889c : 20190607-2129 




* FIXED notes/issues/tboolean-box-okg4-seqmat-mismatch.rst
  "TO BR SA" was always giving incorrect 1st material in G4 recording 
  (presumably a matswap?)

* this issue prompted a major refactoring of the cfg4/CRecorder
  breaking apart the monolithic into many pieces: CWriter/CDebug/...
 

EON
}




tboolean-box3-ip(){ TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; } 
tboolean-box3-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-box3-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-box3(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; } 
tboolean-box3-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-box3--(){ cat << EOP 
import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.polyconfig import PolyConfig
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1" 

CSG.kwa = dict(poly="IM",resolution=20, verbosity=0,ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1 )  # no param, container=1 switches on auto-sizing

wbox = CSG("box3", param=[400,400,300,0], emit=0,  boundary="Vacuum///Water" )
box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Water///GlassSchottF2" )

CSG.Serialize([container, wbox, box], args )
EOP
}
tboolean-box3-notes(){ cat << EON

$FUNCNAME
============================

Used this to show that the incorrect matswap was only effecting 
a BR on the 1st step, ie "TO BR SA" and not all "BR".

* see notes/issues/tboolean-box-okg4-seqmat-mismatch.rst
 

EON
}









tboolean-boxx(){ 
   local msg="=== $FUNCNAME :"
   $FUNCNAME- 
   local testconfig=$($FUNCNAME- 2>/dev/null)
   echo $msg testconfig $testconfig  
   TESTNAME=$FUNCNAME TESTCONFIG=$testconfig tboolean-- $* 
 } 
tboolean-boxx-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-boxx--(){ cat << EOP 
import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.kwa = dict(poly="IM",resolution=20, verbosity=0, ctrl=0, containerscale=3.0, containerautosize=0 )

container = CSG("box", param=[0,0,0,600], boundary='Rock//perfectAbsorbSurface/Vacuum', container=1  ) 

box = CSG("box3", param=[300,300,200,0],  boundary="Vacuum///GlassSchottF2"  )

CSG.Serialize([container, box], args )
EOP
}







tboolean-truncate-ip(){ TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; } 
tboolean-truncate-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-truncate-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-truncate(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; } 
tboolean-truncate-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-truncate--(){ cat << EOP 
import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.5,sheetmask:0x1" 

CSG.kwa = dict(poly="IM",resolution=20, verbosity=0,ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

#smallbox = CSG("box", param=[0,0,0,10], emit=0,  boundary="Vacuum///Water"  )

box = CSG("box", param=[0,0,0,200], emit=-1,  boundary="Rock//perfectSpecularSurface/Vacuum", container=1, containerautosize=1  )

CSG.Serialize([box], args )
EOP
}


tboolean-truncate-notes(){ cat << EON


$FUNCNAME
============================

Aligning truncation behaviour of infinite bounce "hall of mirrors" situation 

Box with internal perfectSpecularSurface : 
with Opticks the bouncemax prevents this going on forever, 
but there was no equivalent with G4 ... so it proceeded to occupy all machine memory 
and subsequent death ! 

Applying a step limit avoids memory death, but initally gave seqhis/seqmat zeros because
never reaching DONE.

The reason for this is that G4 handles BR with StepTooSmall turnarounds, which 
get skipped. 

* Now fixed 

* see notes/issues/cfg4-bouncemax-not-working.rst

* this truncation issue motivated development of the more efficient cfg4/CRecorder/CRec recpoi approach, 
  which makes sense of steps, picking the points for posttrack writing, 
  rather than recstp blindly storing steps "canned approach" for later examination     

* notes/issues/geant4_opticks_integration/tconcentric_pflags_mismatch_from_truncation_handling.rst


EON
}





tboolean-cone-scan(){ SCAN="0,0,100,1,0,0,0,300,10" NCSGScanTest $TMP/tboolean-cone--/1 ; }
tboolean-cone-ip(){ TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; } 
tboolean-cone-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-cone-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-cone(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; } 
tboolean-cone-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-cone--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.polyconfig import PolyConfig
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1" 

CSG.kwa = dict(poly="HY",resolution=4, verbosity=0 ,ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

container = CSG("box", param=[0,0,0,1000], emit=-1, boundary="Rock//perfectAbsorbSurface/Vacuum", container=0 )  

r2,r1 = 100,300
#r2,r1 = 300,300    ## with equal radii (a cylinder) polygonization and raytrace both yield nothing 
#r2,r1 = 300,100    ## radii swapped (upside-down cone) works

z2 = 100
z1 = -100

cone = CSG("cone", param=[r1,z1,r2,z2], boundary="Vacuum///GlassSchottF2", emit=0 )
cone.dump()

CSG.Serialize([container, cone], args )

EOP
}

tboolean-cone-notes(){ cat << EON

$FUNCNAME
=======================

tboolean-;tboolean-cone --okg4 --testauto
    PASS


* CMaker::ConvertPrimitive requires z symmetry for the cone section
  assert( z2 > z1 && z2 == -z1 );

* emit from container box gives okg4 seqhis/seqmat  match

* trying to emit from cone asserts for missing NCone specialization nnode::par_posnrm_model


EON
}




tboolean-prism-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-prism-ip(){ TESTNAME=tboolean-prism tboolean-ipy- $* ; } 
tboolean-prism-p(){ TESTNAME=tboolean-prism tboolean-py- $* ; } 
tboolean-prism(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) &&  tboolean-- $* ; } 
tboolean-prism-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-prism--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.kwa = dict(poly="IM", resolution=40, verbosity=0, ctrl=0 )
container = CSG("box", param=[0,0,0,600], boundary="Rock//perfectAbsorbSurface/Vacuum" )

prism = CSG.MakePrism(angle=45, height=200, depth=300 )
prism.boundary="Vacuum///GlassSchottF2" 
prism.dump()

CSG.Serialize([container, prism], args )

EOP
}

tboolean-prism-notes(){ cat << EON

$FUNCNAME
==========================

tboolean-;tboolean-prism --okg4 
    PASS

tboolean-;tboolean-prism --okg4 --testauto
    PASS 

Stats not great, as the prism is too small compared to container


* unlike other solids, need to manually set bbox for solids stored as a 
  set of planes in NConvexPolyhedron as cannot easily derive the bbox 
  from the set of planes

* see notes/issues/tboolean-prism-convexpolyhedron-meta-assert.rst
  adding missing G4 conversion in CMaker::ConvertConvexPolyhedron 

* see notes/issues/tboolean-prism-G4TessellatedSolid-stuck-track.rst
  4/100000 needed a boot to get un-stuck 


EON
}



tboolean-icosahedron-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-icosahedron-ip(){ TESTNAME=tboolean-icosahedron tboolean-ipy- $* ; } 
tboolean-icosahedron-p(){ TESTNAME=tboolean-icosahedron tboolean-py- $* ; } 
tboolean-icosahedron(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) &&  tboolean-- $* ; } 
tboolean-icosahedron-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-icosahedron--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.kwa = dict(poly="IM", resolution=40, verbosity=0, ctrl=0 )
container = CSG("box", param=[0,0,0,600], boundary="Rock//perfectAbsorbSurface/Vacuum" )

icos = CSG.MakeIcosahedron(300)
icos.boundary="Vacuum///GlassSchottF2" 
icos.dump()

CSG.Serialize([container, icos], args )

EOP
}

tboolean-icosahedron-notes(){ cat << EON

$FUNCNAME
==========================

tboolean-;tboolean-icosahedron --okg4 
    PASS
tboolean-;tboolean-icosahedron --okg4 --testauto
    PASS


EON
}


tboolean-cubeplanes-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-cubeplanes-ip(){ TESTNAME=tboolean-cubeplanes tboolean-ipy- $* ; } 
tboolean-cubeplanes-p(){ TESTNAME=tboolean-cubeplanes tboolean-py- $* ; } 
tboolean-cubeplanes(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) &&  tboolean-- $* ; } 
tboolean-cubeplanes-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-cubeplanes--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.kwa = dict(poly="IM", resolution=40, verbosity=0, ctrl=0, emitconfig=args.autoemitconfig )
container = CSG("box", param=[0,0,0,500], boundary="Rock//perfectAbsorbSurface/Vacuum", emit=-1 )

obj = CSG.MakeCubePlanes(200,200,200)
obj.boundary="Vacuum///GlassSchottF2" 
obj.dump()

CSG.Serialize([container, obj], args )

EOP
}

tboolean-cubeplanes-notes(){ cat << EON

$FUNCNAME
==========================

tboolean-;tboolean-cubeplanes --okg4 
    PASS

tboolean-;tboolean-cubeplanes --okg4 --testauto
    PASS


* notes/issues/tboolean-cubeplanes-many-stuck-tracks-drastic-difference.rst
  
Initially both these had drastic fail from bad G4TesselatedSolid tris 

CubePlanes is a cube that in Opticks is handled as a convexpolyhedon set of planes
and in G4 as a G4TessellatedSolid : it exists mainly as a way of testing these
solids with a shape with easy to check results.  

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/geometry/1257.html
* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/geometry/1257/1/1/1/1/1/1/1.html

Problem may be the familiar one from OpenMesh of 
specifying the verts in the wrong order.  Turned out to be tris
missing entirely.


EON
}



tboolean-trapezoid-deserialize(){ NCSGDeserializeTest $TMP/${FUNCNAME/-deserialize}-- ; }
tboolean-trapezoid-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-trapezoid-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-trapezoid(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-trapezoid-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-trapezoid--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.kwa = dict(poly="IM", resolution=40, verbosity=0, ctrl=0 )

container = CSG("box", param=[0,0,0,500], boundary="Rock//perfectAbsorbSurface/Vacuum" )
container.boundary = args.container

obj = CSG.MakeTrapezoid(z=50.02, x1=100, y1=27, x2=237.2, y2=27)
#obj = CSG.MakeTrapezoid(z=2228.5, x1=160, y1=20, x2=691.02, y2=20 )
obj.boundary="Vacuum///GlassSchottF2" 

obj.dump()

CSG.Serialize([container, obj], args)
EOP
}


tboolean-trapezoid-notes(){ cat << EON

$FUNCNAME
==========================

* notes/issues/tboolean-trapezoid-no-stuck-tracks-but-still-drastic-difference.rst

EON
}



tboolean-uniontree-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-uniontree-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-uniontree(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; }
tboolean-uniontree-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-uniontree--(){ cat << EOP 

import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.polyconfig import PolyConfig
from opticks.analytic.csg import CSG  
from opticks.analytic.treebuilder import TreeBuilder 

args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box", param=[0,0,0,1000], boundary="Rock//perfectAbsorbSurface/Vacuum" )

im = dict(poly="IM", resolution=40, verbosity=1, ctrl=0 )


sp = CSG("sphere", param=[0,0,-1,100] )   # zrange -100:100
sp2 = CSG("sphere", param=[0,0,0,200] )   # zrange -200:200

zs = CSG("zsphere", param=[0,0,0,500], param1=[100,200,0,0],param2=[0,0,0,0])  # zrange 100:200
zs.param2.view(np.uint32)[0] = 3 


r1, r2 = 300,100
z1, z2 = -100, 100

co = CSG("cone", param=[r1,z1,r2,z2])   # zrange 0:200


#prim = [sp,zs,co]    
prim = [sp,co,zs]    

#prim = [sp2, zs, co]  # works as expected
#prim = [sp, co]       # works as expected
#prim = [sp, zs ]      #  sp just touches zp, so difficult to say 
#prim = [sp2, zs]      # expected behavior


ut = TreeBuilder.uniontree(prim, name="$FUNCNAME")
ut.boundary = "Vacuum///GlassSchottF2" 
ut.meta.update(im)
ut.dump()

CSG.Serialize([container, ut], args)


EOP
}


tboolean-uniontree-notes(){ cat << EON


* CMaker objects to node->left->gtransform, curious as seems to be no transforms applied ? 

::

    320     else if(node->is_operator())
    321     {
    322         G4VSolid* left = makeSolid_r(node->left);
    323         G4VSolid* right = makeSolid_r(node->right);
    324 
    325         assert(node->left->gtransform == NULL );
    326 
    327         G4Transform3D* rtransform = ConvertTransform(node->right->gtransform->t);
    328 



prim = [sp,zs,co]    

            un    
     un          co
 sp      zs        


         __________
        /          \       zs
       +---+-----+--+  
          /_______\        co
             \_/           sp


Looks like improper "shadow" sphere surface inside the union,  
propagation intersects with improper surf between sphere and cone,
nudging sphere upwards makes a hole in the center of improper surface
 ... suspect issue with three way overlapping 
nudging downwards still get the improper surf


Can only see the shadow shape when positioned to look up at cone 
(ie looking into threeway region)

Changing order to a more natural (2-way overlapping) one, gets expected behavior

::

    prim = [sp,co,zs]   

                un    
         un          zs
     sp      co        


Does the order depency of a set of unions indicate a bug, 
or a limitation of the algorithm ... or is it just 
a result of having coincident z-faces ?

NON-CONCLUSION: 

* make sure uniontree primitives are in a sensible order 
* avoid three way overlapping where possible


Thinking about the pairwise CSG algorithm the behaviour is kinda
understandable... sp and zs are initially tangential and then after 
nudging the sp upwards creates a small intersection opening up the hole.

But then the union with the cone thru the middle should get rid of that 
surface, and open up a full cavity betweeb the zs and sp ?


TODO:

* see what happens when all coincidident z-planes are avoided

* automatic z-growing in polycone to avoid coincident surfaces 
  (this would be difficult in a general uniontree, but easy in 
  polycone as just needs 3*epsilon grow in z from the
  smaller radius part into the larger radius part)  
  ... analogous to joins in carpentry


EON
}



tboolean-parade-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-parade(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-parade-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-parade--(){ cat << EOP 

import logging, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
        
args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box", param=[0,0,0,1200], boundary=args.container, poly="MC", nx=20 )

dcs = dict(poly="DCS", nominal="7", coarse="6", threshold="1", verbosity=0)
hy = dict(poly="HY", level="4", polycfg="contiguous=1,reversed=0,numsubdiv=0,offsave=1", verbosity=0 )


CSG.boundary = args.testobject
#CSG.kwa = dcs
CSG.kwa = hy

a = CSG("sphere", param=[0,0,0,100]) 
b = CSG("zsphere", param=[0,0,0,100], param1=[-50,60,0,0]) 
c = CSG("box3",param=[100,50,70,0]) 

#d = CSG("box",param=[0,0,10,50])
d = CSG.MakeTrapezoid(z=100, x1=80, y1=100, x2=100, y2=80)


e = CSG("cylinder",param=[0,0,0,100], param1=[-100,100,0,0])
f = CSG("disc",param=[0,0,0,100], param1=[-1,1,0,0])
g = CSG("cone", param=[100,-100,50,100])

#h = CSG.MakeTrapezoid(z=100, x1=80, y1=100, x2=100, y2=80)
#i = CSG.MakeSegment(phi0=0,phi1=45,sz=100,sr=100)

h = CSG.MakeTorus(R=70, r=30)
i = CSG.MakeHyperboloid(r0=80, zf=100, z1=-100, z2=100)
j = CSG.MakeIcosahedron(scale=100.)

prims = [a,b,c,d,e,f,g,h,i,j]

h.meta.update(poly="IM", resolution=40)
i.meta.update(poly="IM", resolution=40)
j.meta.update(poly="IM", resolution=40)


nprim = len(prims)
hprim = nprim/2
ygap = 200
ysize = hprim*ygap
yy = np.repeat(np.linspace(-ysize/2,ysize/2, hprim), 2)
xx = np.tile(np.linspace(-ygap,ygap,2),hprim)
for iprim, prim in enumerate(prims):
    prim.translate = "%s,%s,0" % (xx[iprim],yy[iprim])
pass

CSG.Serialize([container] + prims, args )

EOP
}


tboolean-parade-notes(){ cat << EON

$FUNCNAME
========================

* hmm breaking the russian-doll rule, resulting in assert in NCSGList::checkMaterialConsistency
* suggests this should move to tgltf and use some simple gltf tree

EON
}











tboolean-complement-deserialize(){ NCSGDeserializeTest $TMP/${FUNCNAME/-deserialize}-- ; }
tboolean-complement-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-complement(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-complement-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-complement--(){ cat << EOP 

import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx=20 )

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50, verbosity=1, ctrl=0 )


al = CSG("sphere", param=[0,0,50,100])   # mid-right-Y, conventional difference(top-sphere,bottom-sphere)
ar = CSG("sphere", param=[0,0,-50,100])
a = CSG("difference", left=al, right=ar, translate="0,200,0" )

bl = CSG("sphere", param=[0,0,50,100])   # far-right-Y,     intersect(top-sphere, complement(bot-sphere) )      
br = CSG("sphere", param=[0,0,-50,100], complement=True)
b = CSG("intersection", left=bl, right=br, translate="0,400,0" )

cl = CSG("sphere", param=[0,0, 50,100])  # mid left Y,  conventional difference(bot-sphere,top-sphere) 
cr = CSG("sphere", param=[0,0,-50,100])
c = CSG("difference", left=cr, right=cl, translate="0,-200,0" )

dl = CSG("sphere", param=[0,0,50,100], complement=True)    #  far-left-Y  intersect( complement(top-sphere), bot-sphere )    #  bot-top
dr = CSG("sphere", param=[0,0,-50,100])
d = CSG("intersection", left=dl, right=dr, translate="0,-400,0" )

#b.analyse()
#b.dump_tbool("name")
#b.dump_NNodeTest("name")

CSG.Serialize([container, a, b, c, d ], args )

"""
Getting expected:

+ve Y : top - bottom
-ve Y : bottom - top

"""

EOP
}


tboolean-complement-notes(){ cat << EON

$FUNCNAME
============================

* again breaking Russian doll



EON
}




tboolean-zsphere0-a(){ TESTNAME=${FUNCNAME/-a} tboolean-a- $* ; } 
tboolean-zsphere0-ip(){ TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; } 
tboolean-zsphere0-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-zsphere0(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-zsphere0-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-zsphere0--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1" 

CSG.kwa = dict(poly="IM", resolution=40, verbosity=0, ctrl=0, emitconfig=emitconfig )

container = CSG("box", param=[0,0,0,1000], emit=-1 )
testobj = CSG("zsphere", param=[0,0,0,500], param1=[-200,200,0,0],param2=[0,0,0,0] )

container.boundary = "Rock//perfectAbsorbSurface/Vacuum"
testobj.boundary = "Vacuum///GlassSchottF2" 

#testobj.boundary = "Vacuum/perfectSpecularSurface//GlassSchottF2" 

CSG.Serialize([container, testobj], args )

EOP
}

tboolean-zsphere0-notes(){ cat << EON

$FUNCNAME
============================

Bog standard geometry for checking 
if --testauto can succeed to modify it for automated 
seqmap NCSGIntersect checking::

   tboolean-;tboolean-zsphere0 --okg4 --testauto 
   tboolean-;tboolean-zsphere0-a


* observe col:pol1 rec:vector "circular" shape 
  premonition prior to first bounce 
  
  * premonitions are expected/seen with col:pol2/mat2/flag2 
    but not expected for col:pol1  

  * it looks like the length of the vector is depending on
    something in the future, giving shorter vec for the circle
    of photons that is about to BR 
 



EON
}





tboolean-zsphere1-a(){ TESTNAME=${FUNCNAME/-a} tboolean-a- $* ; } 
tboolean-zsphere1-ip(){ TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; } 
tboolean-zsphere1-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-zsphere1(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-zsphere1-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-zsphere1--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1" 

CSG.kwa = dict(poly="IM", resolution=40, verbosity=0, ctrl=0, emitconfig=emitconfig )

container = CSG("box", param=[0,0,0,1000], emit=-1 )
testobj = CSG("zsphere", param=[0,0,0,500], param1=[-200,200,0,0],param2=[0,0,0,0] )
#testobj = CSG("box", param=[0,0,0,500] )
#testobj = CSG("sphere", param=[0,0,0,500] )

container.boundary = "Rock//perfectAbsorbSurface/Vacuum"
#testobj.boundary = "Vacuum///GlassSchottF2" 
testobj.boundary = "Vacuum/perfectSpecularSurface//GlassSchottF2" 

CSG.Serialize([container, testobj], args )

EOP
}


tboolean-zsphere1-notes(){ cat << EON

$FUNCNAME
============================

* discrepant tboolean-zsphere1-p
* replacing zsphere with box or sphere get agreement, so presumably not material or torch issue
* notes/issues/tboolean-zsphere1-zsphere2-discrep.rst


automated NCSGIntersect check::

   tboolean-;tboolean-zsphere1 --okg4 --testauto 
   tboolean-;tboolean-zsphere1-a --dbgseqhismap "TO:0,SR:1,SA:0"


eyeball the simulations::

   # orthographic (d-key), point photons (p-key several times), mat1 coloring (m-key several times) gives a precise view of whats happening 

   tboolean-;tboolean-zsphere1 --load 
       # endcaps as intersected : appear in expected place 
  
   tboolean-;tboolean-zsphere1 --load --vizg4
       # endcaps as intersected : appear as back to back cones, touching at apex 



EON
}




tboolean-zsphere2-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-zsphere2-ip(){ TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; } 
tboolean-zsphere2-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-zsphere2(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-zsphere2-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-zsphere2--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")
emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1" 
CSG.kwa = dict(poly="IM", resolution=40, verbosity=0, ctrl=0, emitconfig=emitconfig )

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx=20, emit=-1 )

zsphere = CSG("zsphere", param=[0,0,0,500], param1=[100,200,0,0],param2=[0,0,0,0],  boundary="$(tboolean-testobject)" )

CSG.Serialize([container, zsphere], args )

EOP
}

tboolean-zsphere2-notes(){ cat << EON

$FUNCNAME
============================

* matched tboolean-zsphere2-p   when z1:z2 100:200 
* not matched with z1:z2 0:200 or 1:200 !!
 

EON
}



tboolean-union-zsphere-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-union-zsphere-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-union-zsphere(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-union-zsphere-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-union-zsphere--(){ cat << EOP 

import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx=20 )

CSG.kwa = dict(poly="IM", resolution=50, verbosity=3, ctrl=0 )

lzs = CSG("zsphere", param=[0,0,0,500], param1=[-200,200,0,0],param2=[0,0,0,0] )
rzs = CSG("zsphere", param=[0,0,0,500], param1=[300,400,0,0] ,param2=[0,0,0,0] )

uzs = CSG("union", left=lzs, right=rzs )

obj = lzs
obj.boundary = "$(tboolean-testobject)"

CSG.Serialize([container, obj],  args )


EOP
}

tboolean-union-zsphere-notes(){ cat << EON

$FUNCNAME
============================


* FIXED : Assertion failed: (node->left->gtransform == NULL), function makeSolid_r, file /Users/blyth/opticks/cfg4/CMaker.cc, line 325.



EON
}











tboolean-difference-zsphere-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-difference-zsphere(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-difference-zsphere-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-difference-zsphere--(){ cat << EOP 

import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx=20 )

im = dict(poly="IM", resolution=50, verbosity=3, ctrl=0, seeds="0,0,0,1,0,0")

kwa = {}
kwa.update(im)


lzs = CSG("zsphere", param=[0,0,0,500], param1=[-100,100,0,0],param2=[0,0,0,0] )
rzs = CSG("zsphere", param=[0,0,0,400], param1=[-101,101,0,0] ,param2=[0,0,0,0] )

dzs = CSG("difference", left=lzs, right=rzs, boundary="$(tboolean-testobject)", **kwa )

CSG.Serialize([container, dzs], args )

"""

#. FIXED: Differencing two concentric zspheres with same zmin/zmax does not 
   produce the expected ring like shape, unless you slightly increase the 
   zmin/zmax of the one you are subtracting over the other

   * avoid common/coincident faces between the subtracted solids 

#. FIXED: IM poly: fails to find surface even radii 400 and 500, hmm NZSphere looking in +z, 
   but manual seeding doesnt find surface either, it does after fix  
   bug in the setting of manual seed directions in NImplicitMesher


"""

EOP
}








tboolean-hybrid-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-hybrid(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-hybrid-combinetest(){ lldb NOpenMeshCombineTest -- $TMP/${FUNCNAME/-combinetest}--/1 ; }
tboolean-hybrid-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-hybrid--(){ cat << EOP
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box",   name="container",  param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution=10 )

box = CSG("box", param=[0,0,0,201], boundary="$(tboolean-testobject)", level=2 )
sph = CSG("sphere", param=[100,0,0,200], boundary="$(tboolean-testobject)", level=4  )

polycfg="contiguous=1,reversed=0,numsubdiv=0,offsave=1"
obj = CSG("union", left=box, right=sph, boundary="$(tboolean-testobject)", poly="HY", level=4, verbosity=2, polycfg=polycfg  )

# only root node poly and polycfg are obeyed, and distributed to the entire tree

CSG.Serialize([container, obj ],  args )


"""
* initially with box size and sphere radius equal, the poles of the sphere
  were just touching the inside of the box : this is invisible in the ray trace but meant that 
  the polecaps tris where classified as frontier as the poles were regarded as being 
  outside the box (perhaps some epsiloning needed here) 
  ... how to handle such tangency issues without manual tweaks ? 


   (x - 100)^2 + y^2 + z^2 = 200^2       x = 201 

        y^2 + z^2 = 200^2 - 101^2

In [1]: np.sqrt( 200*200-101*101 ) 
Out[1]: 172.62386856978961
  

"""

EOP
}

tboolean-hyctrl-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-hyctrl(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-hyctrl-polytest-lldb(){ lldb NPolygonizerTest -- $TMP/${FUNCNAME/-polytest-lldb}--/1 ; }
tboolean-hyctrl-polytest(){           NPolygonizerTest    $TMP/${FUNCNAME/-polytest}--/1 ; }
tboolean-hyctrl-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-hyctrl--(){ cat << EOP
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box",   name="container",  param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution="1" )

#ctrl = "3"   # tripatch
# tripatch : works in phased and contiguous, contiguous-reversed missed edge(looks broken) 

#ctrl = "4"  # tetrahedron
#ctrl = "6"  # cube
#ctrl = "66"  
# hexpatch_inner : contiguous works, but not with reversed, showing face order sensitivity 
# hexpatch_inner : phased fails to do last flip, when reversed fails to do two flips

ctrl = "666"  # hexpatch 
# hexpatch : contiguous works until reversed=1 showing face order sensitivity
# hexpatch : phased is missing ~6 flips 

cfg0="phased=1,split=1,flip=1,numflip=0,reversed=0,maxflip=0,offsave=1"
cfg1="contiguous=1,reversed=1,numsubdiv=1,offsave=1"

box = CSG("box", param=[0,0,0,500], boundary="$(tboolean-testobject)", poly="HY", level=0, ctrl=ctrl, verbosity=4,  polycfg=cfg1 )



CSG.Serialize([container, box  ], args )

EOP
}






tboolean-boxsphere-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-boxsphere(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-boxsphere-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-boxsphere--(){ cat << EOP
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("sphere",           param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution=10 )

box = CSG("box",    param=[0,0,0,200], boundary="$(tboolean-testobject)", rotate="0,0,1,45" )
sph = CSG("sphere", param=[0,0,0,100], boundary="$(tboolean-testobject)", translate="0,0,200", scale="1,1,0.5" )

object = CSG("${1:-difference}", left=box, right=sph, boundary="$(tboolean-testobject)", poly="IM", resolution=50 )

CSG.Serialize([container, object], args )
EOP
}




tboolean-uncoincide-loadtest(){ ${FUNCNAME/-loadtest}- ; NCSGLoadTest $TMP/${FUNCNAME/-loadtest}--/1 ; }
tboolean-uncoincide-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-uncoincide(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-uncoincide-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-uncoincide--(){ cat << EOP

import logging
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = "$(tboolean-testobject)"
CSG.kwa = dict(verbosity=1, poly="IM", resolution=20 )

# container=1 metadata causes sphere/box to be auto sized to contain other trees
container = CSG("sphere",  param=[0,0,0,10], container=1, containerscale=2.0, boundary="$(tboolean-container)", poly="HY", level="5" )


a = CSG("box3", param=[400,400,100,0] )
b = CSG("box3", param=[300,300,50,0], translate="0,0,25" )

obj = a - b
obj.meta.update(uncoincide="1")  # 0:disable uncoincidence nudging

obj.translate = "0,0,100"
obj.scale = "1,1,1.5"
obj.rotate = "1,0,0,45"

CSG.Serialize([container, obj],  args )


log.info(r"""

$FUNCNAME
=========================

* FIXED : disabling uncoincide not causing speckles, its causing subtraction not to appear
* FIXED : uncoincide fails to work when there is a rotation applied to the composite obj
* FIXED : applying a non-uniform scale 0.5,1.5,1 causes speckles from some angles


SUSPECT the standard make_transformed is doing the nudge translation last
as translation last is the usual TRS order, but nudging 
needs it to be first ...

::
  
  -200,200                   200,200
        +-----------------------+
        |                150,150|
        |   +---------------+   |
        |   |               |   |
        |   |       Y       |   |
        |   |       |       |   |
        |   |       +--X    |   |
        |   |               |   |
        |   |               |   |
        |   |               |   |
        |   +---------------+   |
        |                       |
        +-----------------------+         
  -200,-200                  200,-200



     Z
     |

      50
        +---------------+--+--+--+--- <--- +Z face:1 at 50 
        |
        |
     25 +  +  +  +  +
        |
        |
        +               + + + + + +   <-- -Z face:0 at 0 
        |
        | 
    -25 +  +  +  +  +
        |
        |
        +----------------------------
     -50


Testing what obj transforms prevent uncoincide from working:

* obj.rotate = "0,0,1,45"  # works : about Z is OK
* obj.rotate = "1,0,0,180" # works
* obj.rotate = "1,0,0,360" # works
* obj.rotate = "1,0,0,45" # nope about X or Y 
* obj.rotate = "1,0,0,90" # nope

* obj.scale = "0.5,0.5,0.5"   # uniform scale works
* obj.scale = "0.5,0.5,2"   # works
* obj.scale = "1,1,1.5"     # works

ALL WORKING NOW

Fixed by doing coincidence detection and nudging 
entirely in FRAME_LOCAL of the CSG boolean sub-objects (usually difference) node.
Hmm have only tested right node having a transform.

Basically this means that only the left/right transforms are relevant, the
gtransforms are not used.

To enable this, added a frame argument to par_pos_ and sdf_ and bbox_

       obj
      /  \
     A    B 

Local coincidence testing and uncoincidence fixes 
needs to be done in obj frame, and thus be impervious to whatever
transforms are applied to obj... only the transforms
applied to A and B are relevant.

This means need an SDF route that can operate in such local
frames is needed.


""")
EOP
}





tboolean-disc-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-disc(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-disc-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-disc--(){ cat << EOP

import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = "$(tboolean-testobject)"
CSG.kwa = dict(verbosity=1,poly="IM")

delta = 1.0 

z1,z2 = -0.050,0.050
z1d,z2d = z1-delta,z2+delta


a = CSG("disc", param = [0.000,0.000,523.0,2223.000],param1 = [z1,z2,0.000,0.000])
b = CSG("disc", param = [1000.000,0.000,0.000,223.000],param1 = [z1d,z2d,0.000,0.000])

obj = a - b 

con = CSG("sphere",  param=[0,0,0,10], container=1, containerscale=2.0, boundary="$(tboolean-container)" , poly="HY", level="5" )
CSG.Serialize([con, obj], args )

EOP
}






tboolean-esr-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-esr(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-esr-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-esr--(){ cat << EOP

con_ = "$(tboolean-container)"

import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = "$(tboolean-testobject)"
CSG.kwa = dict(verbosity=1,poly="IM")

z1,z2 = -0.050,0.050

scale = 10. 

z1 *= scale
z2 *= scale

# Varying the delta that increases the thickness of the subtracted disc cylinder 
#
# delta observations for scale = 1
#
#delta = 20.    # NO pole artifact visible
#delta = 15.    # slight artifacting 
#delta = 10.    # pole artifact, disappears axially 
#delta = 1.     # pole artifact, disappears axially   (scale:10 tiny artifact)
#delta = 0.1    # pole artifact, does not disappear axially 
#delta = 0.01   # pole artifact, does not disappear axially 
#delta = 0.001  # pole artifact, does not disappear axially + RING SPECKLES, ~/opticks_refs/tboolean_esr_speckle_centered_on_pole_delta_10-3.png 
delta = 0.0001 # full metal speckle from all angles
#delta = 0.     # zero makes the subtraction appear not to work, just get a disc not an annulus 

z1d,z2d = z1-delta, z2+delta

a = CSG("cylinder", param = [0.000,0.000,0.000,2223.000],param1 = [z1 , z2,0.000,0.000])
b = CSG("cylinder", param = [0.000,0.000,0.000,2144.500],param1 = [z1d,z2d,0.000,0.000],complement = True)
ab = CSG("intersection", left=a, right=b)

obj = ab

check_3 = False
if check_3:
    c = CSG("cylinder", param = [0.000,0.000,0.000,2223.001],param1 = [-0.050,0.050,0.000,0.000])
    d = CSG("cylinder", param = [0.000,0.000,0.000,2144.500],param1 = [-20.051,20.051,0.000,0.000],complement = True)
    cd = CSG("intersection", left=c, right=d)
    cd.translate = "0,0,10"

    e = CSG("cylinder", param = [0.000,0.000,0.000,2223.001],param1 = [-0.050,0.050,0.000,0.000])
    f = CSG("cylinder", param = [0.000,0.000,0.000,2144.500],param1 = [-20.051,20.051,0.000,0.000],complement = True)
    ef = CSG("intersection", left=e, right=f)
    ef.translate = "0,0,20"

    abcd = CSG("union", left=ab, right=cd) 
    abcdef = CSG("union", left=abcd, right=ef) 

    obj = abcdef
pass



con = CSG("sphere",  param=[0,0,0,10], container=1, containerscale=2.0, boundary=con_ , poly="HY", level="5" )
CSG.Serialize([con, obj], args )


"""

Annulus shape, difference of two very thin discs : 
-----------------------------------------------------

Speckles like crazy in the hole, especially when viewed from high angle, end on 

* without increasing thickness of the subtracted, the hole doesnt show : just looks like complete disc

* speckles only appear for very small delta

* delta 0.001 shows the speckles have a circular disposition surrounding the pole

* substantially increasing *delta* thickness of the subtracted disk, gets rid of the speckle 

  * NB this is perfectly valid thing to do, IT DOES NOT CHANGE GEOMETRY, 
    are increasing the size of smth are about to subtract.. 
    there is no concern about bumping into other geometry.

  * DONE: verified the above statement by testing two and three close annuli whose
    subtracted cylinders would overlap, bizarrely this also makes the 
    pole artifact go away ???

* small bright split circle  "pole artifact" when viewed from high angles that 
  moves across the hole as the angle is changed, but disappears in the middle
  when very close to axis (algorithm special casing of axial rays ?)

* increasing *delta* yet more gets rid of the pole artifact, varying the 
  parameters suggests that the primary cause of the pole artifact 
  is the thinness of the "disc" cylinder resulting in near degeneracy of the endcaps 
  intersections

* Adopting robust quadratic did not change behavior... so presumably 
  the issue is from the endcap intersects when they are almost degenerate 

  * using a separate disc primitive or special casing cylinder intersection
    for near degenerate endcaps, seem like the next available thing to try


* THESE ISSUES MOTIVATED DEVELOPMENT OF CSG_DISC see tboolean-disc


"""

EOP
}



tboolean-rip(){ local fnpy="tboolean-${1:-sc}--" ; local py=$TMP/$fnpy.py ; $fnpy > $py ;  ${OPTICKS_IPYTHON:-ipython} -i $py ; }
# jump into ipython session with the python streamed from a bash function

tboolean-sc-loadtest(){ ${FUNCNAME/-loadtest}- ; NCSGLoadTest $TMP/${FUNCNAME/-loadtest}--/1 ; }
tboolean-sc-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-sc(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-sc-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-sc--(){ cat << EOP


import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main

from opticks.analytic.csg import CSG  
from opticks.analytic.gdml import GDML
from opticks.analytic.sc import Sc
from opticks.analytic.treebase import Tree
from opticks.analytic.treebuilder import TreeBuilder

args = opticks_main(csgname="${FUNCNAME/--}")


CSG.boundary = "$(tboolean-testobject)"
CSG.kwa = dict(verbosity=1)

# container=1 metadata causes sphere/box to be auto sized to contain other trees
container = CSG("sphere",  param=[0,0,0,10], container=1, containerscale=2.0, boundary="$(tboolean-container)", poly="HY", level="5" )

gdml = GDML.parse()
tree = Tree(gdml.world)


#gsel = "/dd/Geometry/AdDetails/lvRadialShieldUnit0x"   # thin shell cy with 6 cy holes, poly mess
#gsel = "/dd/Geometry/AdDetails/lvTopReflector0x"      # flat plate with 5 holes, no poly
#gsel = "/dd/Geometry/AdDetails/lvTopRefGap0x"          # flat plate with 5 holes, no poly 
#gsel = "/dd/Geometry/AdDetails/lvTopESR0x"            # flat plate with 9 holes, no poly, center one has coincidence speckle FIXED with disc
#gsel = "/dd/Geometry/AdDetails/lvSstTopCirRibBase0x"       # ring with 4-T slots cut out, coincidence speckle at top of T  

#gsel = "/dd/Geometry/CalibrationSources/lvLedSourceAssy0x"  # three capsules connected with 2 wires , poly works!

gsel = "/dd/Geometry/OverflowTanks/lvGdsOflTnk0x"   # mixed deep tree (union of cy-cy) 

gidx = 0 

target = tree.findnode(gsel, gidx)


orig = Sc.translate_lv(target.lv, maxcsgheight=0)
orig.analyse()
orig.dump(msg="ORIG", detailed=True)

log.info("ORIGINAL:\n"+str(orig.txt))

orig.positivize()
log.info("POSITIVIZED:\n"+str(orig.txt))

obj = TreeBuilder.balance(orig)
log.info("BALANCED:\n"+str(obj.txt))

obj.meta.update(verbosity=1)
obj.dump(msg="BALANCED", detailed=True)
#obj.dump_tboolean(name="esr")


CSG.Serialize([container, obj], args )

"""

* raytrace of balanced tree as expected
* polygonization yielding a blank 


~/opticks_refs/tboolean_esr_pole_artifact.png




"""

EOP
}






tboolean-positivize-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-positivize(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-positivize-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-positivize--(){ cat << EOP

import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main

from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = "$(tboolean-testobject)"
CSG.kwa = dict(poly="IM", resolution=40, verbosity=1 )

container = CSG("sphere",  param=[0,0,0,1000], boundary="$(tboolean-container)", poly="HY", level="4" )

lshape = "box"
#lshape = "sphere"
rshape = "sphere"

a_msg = "Positivized CSG difference, ie A - B ->  A*!B  "
a_left = CSG(lshape, param=[0,0,0,200], rotate="0,0,1,45"  )
a_right = CSG(rshape, param=[0,0,0,100],translate="0,0,200", scale="1,1,0.5" )
a = CSG.Difference(a_left, a_right, translate="-600,0,0" )
a.positivize()   
a.analyse()
log.info("A\n"+str(a.txt))


b_msg = "Standard CSG difference,  A - B "
b_left = CSG(lshape, param=[0,0,0,200], rotate="0,0,1,45" )
b_right = CSG(rshape, param=[0,0,0,100], translate="0,0,200", scale="1,1,0.5" )
b = CSG.Difference(b_left, b_right, translate="0,0,0" )
b.analyse()
log.info("B\n"+str(b.txt))


c_msg = "Standard CSG intersection with B complemented,  A*!B "
c_left = CSG(lshape, param=[0,0,0,200], rotate="0,0,1,45" )
c_right = CSG(rshape, param=[0,0,0,100], translate="0,0,200", scale="1,1,0.5" )
c_right.complement = True 
c = CSG.Intersection(c_left, c_right, translate="600,0,0" )
c.analyse()
log.info("C\n"+str(c.txt))


CSG.Serialize([container, a, b, c  ], args )

log.info(r"""

Three CSG Differences, done in three different ways
=====================================================

Polygonizations and raytraces should all look the same.

A. %s
B. %s 
C. %s

* raytrace correctly looks the same with and without positivize, it is handling the complement

* FIXED : polygonization was ignoring the complement, got little intersection box not a difference,
          fixed by getting implicit function to honour the complement
           
* FIXED : positivize causing implicit mesher to complain of out-of-bounds, as the NNode::bbox
          was not honouring the complement

""" % (a_msg, b_msg, c_msg))

EOP
}









tboolean-bsu-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bsd-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bsi-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bsu(){ TESTNAME=$FUNCNAME TESTCONFIG=$(tboolean-boxsphere- union)        tboolean-- $* ; }
tboolean-bsd(){ TESTNAME=$FUNCNAME TESTCONFIG=$(tboolean-boxsphere- difference)   tboolean-- $* ; }
tboolean-bsi(){ TESTNAME=$FUNCNAME TESTCONFIG=$(tboolean-boxsphere- intersection) tboolean-- $* ; }
tboolean-boxsphere-(){ $FUNCNAME- $* | ${OPTICKS_PYTHON:-python}  ; } 
tboolean-boxsphere--(){ cat << EOP 
import math
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box", param=[0,0,0,400], boundary="$(tboolean-container)", poly="MC", nx=20 )
  
radius = 200 
inscribe = 1.3*radius/math.sqrt(3)    #  150.1110699

box = CSG("box", param=[0,0,0,inscribe])


rtran = dict(translate="100,0,0")
sph = CSG("sphere", param=[0,0,0,radius], **rtran)

object = CSG("${1:-difference}", left=box, right=sph, boundary="$(tboolean-testobject)", poly="IM", resolution=50 )

CSG.Serialize([container, object], args )
EOP
}


tboolean-boxsphere-notes(){ cat << EON

$FUNCNAME
============================


tboolean-;tboolean-bsu --okg4 --testauto 
    PASS

tboolean-;tboolean-bsd --okg4 
    PASS-but-bizarre photon viz, as torchsource is outside container : TODO:FIX


tboolean-;tboolean-bsd --okg4 --testauto 
    PASS

tboolean-;tboolean-bsi --okg4 --testauto 
    PASS  

* bsi was initially only just, but reducing container size pumped up stats of lower freq histories and brought into line


EON
}





tboolean-segment-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-segment(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-segment-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-segment--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.prism import make_segment
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=40, verbosity=1, ctrl=0 )
container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx=20 )

phi0,phi1,sz,sr = 0,45,200,300 

if 0:
    planes, verts, bbox = make_segment(phi0,phi1,sz,sr)
    obj = CSG("segment")
    obj.planes = planes
    obj.param2[:3] = bbox[0]
    obj.param3[:3] = bbox[1]
else:
    obj = CSG.MakeSegment(phi0,phi1,sz,sr)
pass


obj.dump()

CSG.Serialize([container, obj], args )
EOP
}


tboolean-cysegment-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-cysegment(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-cysegment-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-cysegment--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.prism import make_segment
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=40, verbosity=1, ctrl=0 )
container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx=20 )

phi0,phi1,sz,sr = 0,45,202,500*1.5

# sr needs to be significantly more than the rmax are cutting 
# to make a clean segment : otherwise the outside plane
# cuts into the tube ... 
# how much depends on the deltaphi 
 
planes, verts, bbox, srcmeta = make_segment(phi0,phi1,sz,sr)
seg = CSG("segment")
seg.planes = planes
seg.param2[:3] = bbox[0]
seg.param3[:3] = bbox[1]
  
ca = CSG("cylinder", param=[0,0,0,500], param1=[-100,100,0,0] )
cb = CSG("cylinder", param=[0,0,0,400], param1=[-101,101,0,0] )
cy = ca - cb 

obj = cy*seg

obj.dump()

CSG.Serialize([container, obj], args )
EOP
}




tboolean-cyslab-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-cyslab(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-cyslab-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-cyslab--(){ cat << EOP 
import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
from opticks.analytic.gdml import Primitive  
args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx=20 )
  
ca = CSG("cylinder", param=[0,0,0,500], param1=[-100,100,0,0] )
cb = CSG("cylinder", param=[0,0,0,400], param1=[-101,101,0,0] )
cy = ca - cb 


startphi = 90
deltaphi = 90 

phi0 = startphi
phi1 = startphi+deltaphi
dist = 500+1     # make it rmax + smth 

cyseg = Primitive.deltaphi_slab_segment( cy, phi0, phi1, dist)

if 0:
    xyzw_ = lambda phi:(np.cos(phi*np.pi/180.), np.sin(phi*np.pi/180.),0,0)
    sa = CSG("slab", param=xyzw_(phi0+90),param1=[0,501,0,0] )  # normalization done in NSlab.hpp/init_slab
    sb = CSG("slab", param=xyzw_(phi1-90),param1=[0,501,0,0] )  # normalization done in NSlab.hpp/init_slab
    # flipped +-90 as was chopping in opposite side of cylinder
    cysa = cy*sa 
    cysb = cy*sb 
    cyseg = cy*sa*sb 
pass

obj = cyseg

CSG.Serialize([container, obj], args )

"""
         
         |
         |     .
         |     
         |  .
       \ | 
         +--------x
         |
         V


* slabs are defined by two planes with the same normal

* notice that phi slab slicing doesnt work when viewed precisely end on 
  from along the unbounded direction +-Z 

  Rather than the segment, see the whole cylinder.

  However its is difficult to get into that position without 
  using G/Composition/Home(H) and the axial view buttons(+Z/-Z)

  * unclear how this can be handled ?


"""


EOP
}




tboolean-undefined-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-undefined(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-undefined-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-undefined--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,4], boundary=args.container, poly="MC", nx=20 )
  
a = CSG.MakeUndefined(name="$FUNCNAME", src_type="hello")

CSG.Serialize([container, a], args )


EOP
}




tboolean-empty-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-empty-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-empty(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; } 
tboolean-empty-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-empty--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,400], boundary=args.container, poly="MC", nx=20, emit=-1, emitconfig="$(tboolean-emitconfig)" )  
CSG.Serialize([container], args )

EOP
}



tboolean-media-ip(){ TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; } 
tboolean-media-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-media-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-media-g(){ TESTNAME=${FUNCNAME/-g} TESTCONFIG=$($TESTNAME- 2>/dev/null) tboolean-g- --export --dbgsurf --dbgbnd ; } 
tboolean-media(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; } 
tboolean-media-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-media--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

omat = "Rock"
#osur = "perfectSpecularSurface"
osur = ""
isur = "perfectAbsorbSurface"
imat = "Pyrex"

box = CSG("box", param=[0,0,0,400], boundary="/".join([omat,osur,isur,imat]), poly="MC", nx=20, emit=-1, emitconfig="$(tboolean-emitconfig)" )  
CSG.Serialize( [box], args )

EOP
}


tboolean-media-notes(){ cat << EON

${FUNCNAME/-notes}
=======================

NCSG Box with boundary : Rock//perfectAbsorbSurface/Pyrex

* initially cfg4 translation of this failed, 
  as outer material Rock required containing Rock volume in G4. 
  So introduced G4 only universe wrapper NCSGList::createUniverse  
  to reconcile the boundary based Opticks geometry 
  with the volume based G4 one.

* it was also found that cannot have a bordersurface at worldvolume edge, 
  as needs pv1/pv2 pointers : again the universe wrapper helps with 
  this as what appears to be the outer volume at NCSG level actually 
  has a wrapper volume added. 

* see notes/issues/surface_review.rst while in this context improved 
  the handling of surfaces with test geometries

* note that without the perfectAbsorbSurface on the container volume
  okg4 agreement is lost as it then not possible to reconcile 
  the volume vs boundary models with Opticks ending with "MI" for
  photons crossing the cube whereas G4 ends "BT MI"

  * this is a bookeeping technicality which could be hidden, 
    but is simpler just to like with this : anyhow a real test setup 
    would use a light tight box.  


Lost agreement without the perfectAbsorbSurface::

    tboolean-;tboolean-media --okg4
    tboolean-;tboolean-media-p 
    ...

    [2017-11-10 18:19:31,578] p53503 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171110-1812 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171110-1812 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock///Pyrex
    /tmp/blyth/opticks/tboolean-media--
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000    608996.86/15 = 40599.79  (pval:0.000 prob:1.000)  
    0000     299543         0        299543.00  TO MI
    0001          0    298231        298231.00  TO BT MI
    0002     289569    290483             1.44  TO AB
    0003          0      5448          5448.00  TO BR BT MI
    0004       5102      5286             3.26  TO BR AB
    0005       5233         0          5233.00  TO BR MI
    0006        152         0           152.00  TO SC MI
    0007          0       134           134.00  TO SC BT MI


Agreement when put back the surface::
     
    [2017-11-10 18:26:33,244] p53767 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171110-1825 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171110-1825 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Pyrex
    /tmp/blyth/opticks/tboolean-media--
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  TO SA
    0001     289569    290680             2.13  TO AB
    0002        290       292             0.01  TO SC SA
    0003         82        98             1.42  TO SC AB
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  





EON
}











tboolean-sphere-m(){ TESTNAME=${FUNCNAME/-m} tboolean-m- $* ; } 
tboolean-sphere-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-sphere-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-sphere-g(){ TESTNAME=${FUNCNAME/-g} TESTCONFIG=$($TESTNAME- 2>/dev/null) tboolean-g- --export --dbgsurf ; } 
tboolean-sphere(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; } 
tboolean-sphere-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-sphere--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

#material = "GlassSchottF2"
#material = "MainH2OHale"
material = "Pyrex"

CSG.kwa = dict(poly="IM", resolution=20 , emitconfig="$(tboolean-emitconfig)" )
container = CSG("box",    param=[0,0,0,400.0], boundary="Rock//perfectAbsorbSurface/Vacuum", emit=-1 )  
sphere    = CSG("sphere", param=[0,0,0,200.0], boundary="Vacuum/perfectSpecularSurface//%s" % material, emit=0 ) 

CSG.Serialize([container, sphere], args )

EOP
}
tboolean-sphere-notes(){ cat << EON

${FUNCNAME/-notes}
=======================

* FIXED notes/issues/G4_barfs_tboolean_sphere_emitter.rst



EON
}



tboolean-torus-p(){ TESTNAME=${FUNCNAME/-p} tboolean-py- $* ; } 
tboolean-torus-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-torus()
{

    local photons=100000
    #local photons=10
 
    TESTNAME=$FUNCNAME \
    TESTCONFIG=$($FUNCNAME- 2>/dev/null) \
    TORCHCONFIG=$(tboolean-torchconfig-disc 0,0,350 150 $photons) \
    tboolean-- $* ; 
} 
tboolean-torus-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-torus--(){ cat << EOP 

import logging
log = logging.getLogger("$FUNCNAME")
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")
log.info("args.container : %r " % args.container)
log.info("args.testobject : %r " % args.testobject)

CSG.boundary = args.testobject
#CSG.kwa = dict(poly="MC", resolution="100")
CSG.kwa = dict(poly="IM", resolution=50)

emit = -1

container = CSG("box3", param=[2*150+1,2*150+1,2*50+1,0], boundary=args.container, poly="IM", nx=20, emit=emit, emitconfig="$(tboolean-emitconfig)" )
  
a = CSG.MakeTorus(R=100, r=50)

CSG.Serialize([container, a], args )
#CSG.Serialize([a], args )


EOP
}





tboolean-hyperboloid-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-hyperboloid(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-hyperboloid-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-hyperboloid--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
#CSG.kwa = dict(poly="MC", resolution="100")
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,400], boundary=args.container, poly="MC", nx=20 )
  
#a = CSG.MakeHyperboloid(r0=100, zf=100, z1=-100, z2=100)
a = CSG.MakeHyperboloid(r0=100, zf=100, z1=0, z2=100)

CSG.Serialize([container, a], args )

EOP
}




tboolean-cubic-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-cubic(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-cubic-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-cubic--(){ cat << EOP 

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,200], boundary=args.container, poly="MC", nx=20 )
  
#a = CSG.MakeCubic(A=0.0001, B=2, C=2, D=2, z1=-10, z2=10)   ## 

a = CSG.MakeCubic(A=0.0001, B=2, C=10, D=2, z1=-10, z2=10) 

#zrrs = [[-100,30],[-50,80],[50,30],[100,100]]
#a = CSG.MakeCubicBezier(zrrs)


CSG.Serialize([container, a], args )

EOP
}






tboolean-12-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-12(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-12-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-12--(){ cat << EOP 
"""
PMT Neck Modelling With Hyperboloid
=====================================

* replace "cy - to" with "hy"
* npy/NHyperboloid.cpp re formulae


* took approach of matching xx+yy of hyperboloid to the
  torus at the z extremes of the cylinder

  * hyp gives slightly fatter neck
  * hyp is drastically cheaper
  * matching at the middle is another possibility 


hyperboloid zf to hit xx+yy = ww at z=zw

                          rr0
        zf = zw * sqrt( -----------   )
                        ww - rr0

hyperboloid 
   sqrt(xx + yy) = sqrt( rr0 * (  (z/zf)^2  +  1 ) )

                 = r0 * sqrt( (z/zf)^2 + 1) 
    
torus: 
   sqrt(xx + yy) =  R - sqrt( rr - zz )


                                  [cy frame]       [to/hy frame]
      +-----------------------+    z = ch            z = 2*ch
      |\                     /|
      | \                   / |
      |  \                 /  |
      |   \               /   |
      |    \             /    |
      |     \           /     |
      +------*----|----*------+    z = -ch           z = 0 
      
"""

import math
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
from opticks.analytic.sc import Sc

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,400], boundary=args.container, poly="MC", nx=20 )


class Tor(object):
    def __init__(self, R, r):
        self.R = R
        self.r = r

    def __repr__(self):
        return "Tor r:%s R:%s " % (self.r, self.R )

    def rz(self, z):
        R = self.R
        r = self.r
        return R - math.sqrt(r*r-z*z)  
   
class Hyp(object):
    def __init__(self, r0, zf, z1, z2):
        self.r0 = r0
        self.zf = zf
        self.z1 = z1
        self.z2 = z2

    @classmethod
    def ZF(cls, r0, zw, w ):
        """ 
        :param r0: waist radius, ie radius at z=0 
        :param zw: z at which to pin the radius
        :param w: 

        hyperboloid zf param to hit radius w, at z=zw 
        """
        rr0 = r0*r0
        ww = w*w 
        return zw*math.sqrt(rr0/(ww-rr0)) 

    def __repr__(self):
        return "Hyp r0:%s zf:%s z1:%s z2:%s " % (self.r0, self.zf, self.z1, self.z2 ) 

    def rz(self, z):
        R = self.R
        r0 = self.r0
        zf = self.zf
        zs = z/zf 
        return r0*math.sqrt( zs*zs + 1 )  
 


R,r,ch,cz,cn = 97.000,52.010,23.783,-23.773,-195.227
r0 = R - r 
rr0 = r0*r0

tor = Tor(R,r)
assert tor.rz(0) == R - r 
assert tor.rz(r) == R  

# in torus/hyp frame cylinder top and bottom at

ztop, zbot = ch - cz, -ch - cz  #     (47.556, -0.010000000000001563)
rtop, rbot = tor.rz(ztop), tor.rz(zbot)

zf = Hyp.ZF( rbot, ztop, rtop )
hyp = Hyp( rbot, zf, zbot, ztop )




a = CSG("zsphere", param = [0.000,0.000,0.000,179.000],param1 = [-179.000,179.000,0.000,0.000])
a.transform = [[1.391,0.000,0.000,0.000],[0.000,1.391,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,0.000,1.000]]

b = CSG("cylinder", param = [0.000,0.000,0.000,75.951],param1 = [-ch,ch,0.000,0.000])
c = CSG("torus", param = [0.000,0.000,r,R],param1 = [0.000,0.000,0.000,0.000],complement = True)
c.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,cz,1.000]]
bc = CSG("intersection", left=b, right=c)
bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,cn,1.000]]

bc2 = CSG("hyperboloid", param = [hyp.r0,hyp.zf,hyp.z1,hyp.z2])
bc2.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,cn+cz,1.000]]


abc = CSG("union", left=a, right=bc)
abc2 = CSG("union", left=a, right=bc2)


d = CSG("cylinder", param = [0.000,0.000,0.000,45.010],param1 = [-57.510,57.510,0.000,0.000])
d.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-276.500,1.000]]

abcd = CSG("union", left=abc, right=d)
abcd2 = CSG("union", left=abc2, right=d)


e = CSG("cylinder", param = [0.000,0.000,0.000,254.000],param1 = [-92.000,92.000,0.000,0.000],complement = True)
e.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,92.000,1.000]]
abcde = CSG("intersection", left=abcd, right=e)
abcde2 = CSG("intersection", left=abcd2, right=e)

# intersection with cylinder is the profligate splitting 
#raw = abcde
#raw2 = abcde2

raw = abcd
raw2 = abcd2





raw.dump("raw")
raw2.dump("raw2")


maxcsgheight = 4
maxcsgheight2 = 5

obj = Sc.optimize_csg(raw, maxcsgheight, maxcsgheight2 ) 
obj2 = Sc.optimize_csg(raw2, maxcsgheight, maxcsgheight2 ) 

obj.dump("optimized")
obj2.dump("optimized")


obj.translate = [-300,0,0]
obj2.translate = [300,0,0]


#uobjs = [raw]
uobjs = [obj, obj2]


con = CSG("box",  param=[0,0,0,10], container=1, containerscale=2.0, boundary=args.container , poly="IM", resolution=20 )
CSG.Serialize([con]+uobjs, args )



EOP
}













tboolean-ellipsoid-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-ellipsoid(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-ellipsoid-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-ellipsoid--(){ cat << EOP 
import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
from opticks.analytic.gdml import Primitive  
args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx=20 )
  
#a = CSG.MakeEllipsoid(axes=[100,200,100] )
#a = CSG.MakeEllipsoid(axes=[100,200,100], zcut1=-50, zcut2=50 )
#a = CSG.MakeEllipsoid(axes=[100,200,100], zcut1=-50  )
a = CSG.MakeEllipsoid(axes=[100,200,100], zcut1=-50  )

a.translate = [300,300,0]


#print a
#print "scale:", a.scale

CSG.Serialize([container, a], args )

"""

* currently applying a scale to the ellipsoid will stomp on base scaling 
  of the sphere 


"""

EOP
}






tboolean-spseg-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-spseg(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-spseg-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-spseg--(){ cat << EOP 
import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
from opticks.analytic.gdml import Primitive  
args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx=20 )
  
a = CSG("sphere", param=[0,0,0,500] )
b = CSG("sphere", param=[0,0,0,490] )
d = a - b


phi0 =  0
phi1 = 90
dist = 500 + 1

s = Primitive.deltaphi_slab_segment( d, phi0, phi1, dist)

CSG.Serialize([container, s], args )

EOP
}






tboolean-sphereslab-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-sphereslab(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
tboolean-sphereslab-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-sphereslab--(){ cat << EOP 
import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution=50)

container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx=20 )
  
slab   = CSG("slab", param=[1,1,1,0],param1=[-500,100,0,0] )  # normalization done in NSlab.hpp/init_slab
sphere = CSG("sphere", param=[0,0,0,500] )

object = CSG("intersection", left=sphere, right=slab )

CSG.Serialize([container, object], args )

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


* note that without the caps enabled see nothing, because of this
  the user setting of endcap flags is now diabled : they are always 
  set to ON in NSlab.hpp



"""
EOP
}


tboolean-sphereplane-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-sphereplane(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-sphereplane-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-sphereplane--(){ cat << EOP 
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject 
CSG.kwa = dict(poly="IM", resolution=50, verbosity=1 )


container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx=20, verbosity=0 )
  
plane  = CSG("plane",  param=[0,0,1,100], complement=False )
sphere = CSG("sphere", param=[0,0,0,500] )

object = CSG("intersection", left=sphere, right=plane )

CSG.Serialize([container, object], args )

"""

With or without complement on the place get visbility wierdness, 
unbounded sub-objects such as planes are not valid CSG sub-objects within OpticksCSG 

0. Polygonization looks correct
1. only see the sphere surface from beneath the plane (ie beneath z=100)
2. only see the plane surface in shape of disc from above the plane 

"""
EOP
}

tboolean-boxplane-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-boxplane(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; }
tboolean-boxplane-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-boxplane--(){ cat << EOP 
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx=20, verbosity=0 )

plane  = CSG("plane",  param=[0,0,1,100] )
box    = CSG("box", param=[0,0,0,200]  )
object = CSG("intersection", left=plane, right=box, boundary="$(tboolean-testobject)", poly="IM", resolution=50, verbosity=1 )

CSG.Serialize([container, object], args )

"""
#. Analogous issue to tboolean-sphere-plane
"""
EOP
}



tboolean-plane-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-plane(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; }
tboolean-plane-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-plane--(){ cat << EOP 
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx=20, verbosity=0 )

bigbox = CSG("box", param=[0,0,0,999] )
plane  = CSG("plane",  param=[0,0,1,100] )
object = CSG("intersection", left=plane, right=bigbox, boundary="$(tboolean-testobject)", poly="IM", resolution=50, verbosity=1 )

CSG.Serialize([container, object], args )

"""


#. An odd one, it appears OK in polygonization and raytrace : but it is breaking the rules,
   are using an unbounded sub-object (the plane) in intersection with the bigbox.

#. Actually the wierdness is there, just you there is no viewpoint from which you can see it. 
   Reducing the size of the bigbox to 500 allows it to manifest.

#. intersecting the plane with the container, leads to coincident surfaces and a flickery mess when 
   view from beneath the plane, avoided issue by intersecting instead with a bigbox slightly 
   smaller than the container

"""

EOP
}




tboolean-cy-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-cy(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-cy-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-cy--(){ cat << EOP 
import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(verbosity=1, poly="HY", resolution=4 )

container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="HY", resolution=4, verbosity=0 )

ra = 200 
z1 = -100
z2 = 100

a = CSG("cylinder", param=[0,0,0,ra], param1=[z1,z2,0,0] )
#a = CSG("disc", param=[0,0,0,ra], param1=[-0.01,0.01,0,0] )
#a = CSG("zsphere", param=[0,0,0,ra], param1=[z1,z2,0,0] )


obj = a 

CSG.Serialize([container, obj], args )

EOP
}





#tboolean-cyd-torch-(){ tboolean-torchconfig-disc 1,1,599 ; }  ## non-axial works
#tboolean-cyd-torch-(){ tboolean-torchconfig-disc 0,0,599 300 ; }  ## FIXED: axial rays fails to intersect with the sphere dimple
tboolean-cyd-torch-(){ tboolean-torchconfig-disc 0,0,600 90 1000000 ; }  ## FIXED: axial rays fails to intersect, with the sphere dimple


tboolean-cyd-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-cyd(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) TORCHCONFIG=$($FUNCNAME-torch-) tboolean-- $* ; }
tboolean-cyd-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-cyd--(){ cat << EOP 
import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
CSG.kwa = dict(verbosity=1, poly="IM", resolution=4 )

container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="IM", resolution=4, verbosity=0 )

ra = 200 
z1 = -100
z2 = 100

a = CSG("cylinder", param=[0,0,0,ra], param1=[z1,z2,0,0] )
b = CSG("sphere", param=[0,0,z2,ra/2]  )

obj = a - b 

CSG.Serialize([container, obj], args )

"""  
                Z
                |
                
         +---.---+---.---+ (200,100) 
         |   .       .   |
         |     .   .     |   
     ----|-------^-------|---- X
         |               |   
         |               |   
         +---------------+

"""

EOP
}










tboolean-cylinder-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-cylinder(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-cylinder-(){  $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-cylinder--(){ cat << EOP 
import numpy as np
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = "$(tboolean-testobject)"
CSG.kwa = dict(verbosity=1, poly="IM", resolution="30" )

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="MC", nx=20, verbosity=0 )

z1 = -100
z2 = 100
delta = 0.001 

z1d = z1 - delta
z2d = z2 + delta

a = CSG("cylinder", param=[0,0,0,400], param1=[z1,z2,0,0] )
b = CSG("cylinder", param=[0,0,0,100], param1=[z1d,z2d,0,0] )
c = CSG("cylinder", param=[200,200,0,100], param1=[z1d,z2d,0,0] )
d = CSG("cylinder", param=[-200,-200,0,100], param1=[z1d,z2d,0,0] )

obj = a - b - c - d 


#obj.rotate = "1,1,1,45"


CSG.Serialize([container, obj], args )

"""
Failed to reproduce the ESR speckles, from tboolean-sc

"""
EOP
}






tboolean-fromstring-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-fromstring(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-fromstring-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} ; }
tboolean-fromstring--(){ cat << EOP

from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  
from opticks.analytic.gdml import Primitive

args = opticks_main(csgname="${FUNCNAME/--}")


so = Primitive.fromstring(r"""<tube aunit="deg" deltaphi="360" lunit="mm" name="AdPmtCollar0xc2c5260" rmax="106" rmin="105" startphi="0" z="12.7"/>""")

obj = so.as_ncsg() 
obj.boundary = "$(tboolean-testobject)"

container = CSG("box", param=[0,0,0,200], boundary="$(tboolean-container)", poly="IM", resolution=20 )

CSG.Serialize([container, obj], args )

EOP
}






tboolean-unbalanced-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-unbalanced(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-)  tboolean-- $* ; }
tboolean-unbalanced-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $*  ; }
tboolean-unbalanced--(){  cat << EOP 
import math, logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

 
radius = 200 
inscribe = 1.3*radius/math.sqrt(3)

lbox = CSG("box",    param=[100,100,-100,inscribe])
lsph = CSG("sphere", param=[100,100,-100,radius])
left  = CSG("difference", left=lbox, right=lsph, boundary="$(tboolean-testobject)" )

right = CSG("sphere", param=[0,0,100,radius])

object = CSG("union", left=left, right=right, boundary="$(tboolean-testobject)", poly="IM", resolution=60 )
object.dump()

container = CSG("box", param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution=20)

CSG.Serialize([container, object], args )

EOP
}





tboolean-deep-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-deep(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-deep-(){ local n=14 ; tboolean-gdml- $TMP/${FUNCNAME}$n --gsel $($FUNCNAME- $n) --gmaxdepth 1 ; }
tboolean-deep--(){  $FUNCNAME- | sed -n ${1:-1}p ; }
tboolean-deep---(){ cat << EOD
/dd/Geometry/PoolDetails/lvNearTopCover0x
/dd/Geometry/AdDetails/lvRadialShieldUnit0x
/dd/Geometry/AdDetails/lvTopESR0x
/dd/Geometry/AdDetails/lvTopRefGap0x
/dd/Geometry/AdDetails/lvTopReflector0x
/dd/Geometry/AdDetails/lvBotESR0x
/dd/Geometry/AdDetails/lvBotRefGap0x
/dd/Geometry/AdDetails/lvBotReflector0x
/dd/Geometry/AdDetails/lvSstTopCirRibBase0x
/dd/Geometry/CalibrationSources/lvLedSourceAssy0x
/dd/Geometry/CalibrationSources/lvGe68SourceAssy0x
/dd/Geometry/CalibrationSources/lvAmCCo60SourceAssy0x
/dd/Geometry/OverflowTanks/lvLsoOflTnk0x
/dd/Geometry/OverflowTanks/lvGdsOflTnk0x
/dd/Geometry/OverflowTanks/lvOflTnkContainer0x
/dd/Geometry/PoolDetails/lvTablePanel0x
/dd/Geometry/Pool/lvNearPoolIWS0x
/dd/Geometry/Pool/lvNearPoolCurtain0x
/dd/Geometry/Pool/lvNearPoolOWS0x
/dd/Geometry/Pool/lvNearPoolLiner0x
/dd/Geometry/Pool/lvNearPoolDead0x
/dd/Geometry/RadSlabs/lvNearRadSlab90x
EOD
}

tboolean-deep-notes(){ cat << EON

n = {}  
n["/dd/Geometry/PoolDetails/lvNearTopCover0x"]="1:flat lozenge"
n["/dd/Geometry/AdDetails/lvRadialShieldUnit0x"]="2:tambourine with 6 holes, potential wierdness : from inside dont see the caps, coincidence perhaps"
n["/dd/Geometry/AdDetails/lvTopESR0x"]="3:evaluative_csg tranOffset 0 numParts 1023 perfect tree height 9 exceeds current limit"
n["/dd/Geometry/AdDetails/lvBotReflector0x"]="8:  disc with 4 slots, but thats partial: evaluative_csg tranOffset 0 numParts 511 perfect tree height 8 exceeds current limit "
n["/dd/Geometry/AdDetails/lvSstTopCirRibBase0x"]="9:  cross cut cylinder, obvious coincidence speckling in the cuts"


EON
}


tboolean-0q(){  TESTNAME=$FUNCNAME TESTCONFIG="analytic=1_csgpath=$TMP/tboolean-0-_name=tboolean-0-_mode=PyCsgInBox" && tboolean--  ; }
tboolean-0-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-0(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-0-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel 0 ; }
tboolean-0-deserialize(){ VERBOSITY=0 lldb NCSGDeserializeTest -- $TMP/tboolean-0- ; }
tboolean-0-polygonize(){  VERBOSITY=0 lldb NCSGPolygonizeTest  -- $TMP/tboolean-0- ; }

tboolean-gds0-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-gds0(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-gds0-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/AD/lvGDS0x ; }

tboolean-oav-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-oav(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-oav-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/AD/lvOAV0x ; }

tboolean-iav-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-iav(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-iav-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/AD/lvIAV0x ; }

tboolean-sst-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-sst(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-sst-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/AD/lvSST0x --gmaxdepth 3 ; }





tboolean-gds-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-gds(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-gds-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; } 
tboolean-gds--(){ cat << EOP
import logging
log = logging.getLogger(__name__)
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = "Acrylic//perfectAbsorbSurface/GdDopedLS"
CSG.kwa = dict(verbosity=0, poly="IM", resolution=20)

a = CSG("cylinder", param = [0.000,0.000,0.000,1550.000],param1 = [-1535.000,1535.000,0.000,0.000])
b = CSG("cone", param = [1520.000,3070.000,75.000,3145.729],param1 = [0.000,0.000,0.000,0.000])
c = CSG("cylinder", param = [0.000,0.000,0.000,75.000],param1 = [3145.729,3159.440,0.000,0.000])
bc = CSG("union", left=b, right=c)
bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-1535.000,1.000]]

abc = CSG("union", left=a, right=bc)

# photons formed maltese cross, until upped timemax from 10ns to 20ns

#abc.transform = [[0.543,-0.840,0.000,0.000],[0.840,0.543,0.000,0.000],[0.000,0.000,1.000,0.000],[-18079.453,-799699.438,-7100.000,1.000]]
abc.transform = [[0.543,-0.840,0.000,0.000],[0.840,0.543,0.000,0.000],[0.000,0.000,1.000,0.000],[0,0,1000.,1.000]]

obj = abc

container = CSG("sphere",  param=[0,0,0,10], container=1, containerscale=2.0, boundary="$(tboolean-container)" , poly="HY", level=5 )
CSG.Serialize([container, obj], args )

EOP
}



tboolean-pmt-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-pmt(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-pmt-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/PMT/lvPmtHemi0x ; }




### trapezoid examples

tboolean-sstt-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-sstt(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-sstt-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/AdDetails/lvSstTopRadiusRib0x ; }
# contains a trapezoid as part of, thats the real skinny one 

tboolean-sstt2-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-sstt2(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-sstt2-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/AdDetails/lvSstInnVerRibBase0x ; }


## ntc: flat lozenge shape, a deep CSG tree

tboolean-ntc-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-ntc(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- $* 2>/dev/null) && tboolean--  ; }
tboolean-ntc-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/PoolDetails/lvNearTopCover0x $* ; }


tboolean-p0-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-p0(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) && tboolean--  ; }
tboolean-p0-(){ tboolean-gdml- $TMP/$FUNCNAME --gsel  /dd/Geometry/AdDetails/lvOcrGdsInIav0x ; }



tboolean-gdml-()
{      
    local csgpath=$1
    shift
    ${OPTICKS_PYTHON:-python} $(tboolean-gdml-translator) \
          --csgpath $csgpath \
          --container $(tboolean-container)  \
          --testobject $(tboolean-testobject) \
          $*
}
tboolean-gdml-translator(){ echo $(opticks-home)/analytic/translate_gdml.py ; }
tboolean-gdml-translator-vi(){ vi $(tboolean-gdml-translator); }

tboolean-gdml-check(){ tboolean-gdml- 2> /dev/null ; }
tboolean-gdml-edit(){ vi $(tboolean-gdml-translator)   ; }
tboolean-gdml-scan(){ SCAN="0,0,127.9,0,0,1,0,0.1,0.01" NCSGScanTest $TMP/tboolean-gdml-/1 ; }
tboolean-gdml-ip(){  tboolean-cd ; ${OPTICKS_IPYTHON:-ipython} tboolean_gdml.py -i ; }



tboolean-dd-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-dd(){   TESTNAME=$FUNCNAME TESTCONFIG=$(tboolean-dd- 2>/dev/null)     tboolean-- $* ; }
tboolean-dd-()
{       
    ${OPTICKS_PYTHON:-python} $(tboolean-dir)/tboolean_dd.py \
          --csgpath $TMP/$FUNCNAME \
          --container $(tboolean-container)  \
          --testobject $(tboolean-testobject)  

    # got too long for here-string  so broke out into script
}
tboolean-dd-check(){ tboolean-dd- 2> /dev/null ; }
tboolean-dd-edit(){ vi $(tboolean-dir)/tboolean_dd.py  ; }
tboolean-dd-scan(){ SCAN="0,0,127.9,0,0,1,0,0.1,0.01" NCSGScanTest $TMP/tboolean-dd-/1 ; }




tboolean-interlocked-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-interlocked(){  TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-interlocked-(){ $FUNCNAME- | ${OPTICKS_PYTHON:-python} $* ; }
tboolean-interlocked--(){ cat << EOP 
import math
from opticks.ana.main import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgname="${FUNCNAME/--}")

CSG.boundary = args.testobject
  
radius = 200 
inscribe = 1.3*radius/math.sqrt(3)

lbox = CSG("box",    param=[100,100,-100,inscribe])
lsph = CSG("sphere", param=[100,100,-100,radius])
left  = CSG("difference", left=lbox, right=lsph )

rbox = CSG("box",    param=[0,0,100,inscribe])
rsph = CSG("sphere", param=[0,0,100,radius])

tran = dict(translate="0,0,100", rotate="1,1,1,45", scale="1,1,1.5" )
right = CSG("difference", left=rbox, right=rsph, **tran)

#dcs = dict(poly="DCS", nominal=7, coarse=6, threshold=1, verbosity=0)
im = dict(poly="IM", resolution=64, verbosity=0, ctrl=0 ) #seeds = "100,100,-100,0,0,300"
obj = CSG("union", left=left, right=right, **im )
#obj.translate = "0,-300,0"
#obj.meta.update(gpuoffset="0,600,0")


#mc = dict(poly="MC", nx=20)
container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="IM", resolution=20 )

CSG.Serialize([container, obj], args )

EOP
}




