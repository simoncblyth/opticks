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

ckm-source(){ echo $BASH_SOURCE ; }
ckm-vi(){ vi $(ckm-source) ; }
ckm-env(){ echo -n ; }
ckm-usage(){ cat << EOU

CerenkovMinimal
==================

HAVE MOVED THIS TO BIN AS THIS IS NOT DIGESTIBLE BY NEWBIES


Exploring a different architecture for Opticks executables, 
revolving around the direct geometry geocache and persisted gensteps which
are identified via OPTICKS_KEY envvar.   


ckm-go
    builds and runs the CerenkovMinimal executable
ckm--
    just runs the CerenkovMinimal executable

Objectives
------------

CerenkovMinimal is really intended to feature **Minimal** usage of Opticks. 
Just enough to effect the acceleration of optical photons.  

That means:

* constrain the use of Opticks headers (even for things like logging/utilities)
  to only where it is absolutely essential to do so 

* CerenkovMinimal is NOT a place for debugging, the 2nd executable that 
  adopts the CerenkovMinimal geocache and gensteps is where debugging is done

* moving to not using PLOG in CerenkovMinimal makes a stark contrast
  between user/framework code(and logging output)

  * user code : G4cout 
  * Opticks code : LOG(info) etc... 


Hmm : this guideline on logging is broken quite a bit, lots of use of PLOG ???
As intermediate step:
  
* start placing these inside WITH_OPTICKS 
* minimize them

EOU
}
ckm-dir(){ echo $(dirname $(dirname $(ckm-source)))/examples/Geant4/CerenkovMinimal  ; }
ckm-cd(){  cd $(ckm-dir) ; }
ckm-c(){  cd $(ckm-dir) ; }


#ckm-dig(){ echo c250d41454fba7cb19f3b83815b132c2 ; }
#ckm-dig(){ echo 792496b5e2cc08bdf5258cc12e63de9f ; }
ckm-dig(){ echo 27d088654714cda61096045ff5eacc02 ; }

ckm-key(){ echo CerenkovMinimal.X4PhysicalVolume.World.$(ckm-dig) ; }
ckm-key-export(){ export OPTICKS_KEY=$(ckm-key) ;  }
ckm-indexer-test(){   OPTICKS_KEY=$(ckm-key) IndexerTest --envkey ; }


ckm-idpath(){ echo $LOCAL_BASE/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/$(ckm-dig)/1 ; }
ckm-kcd(){ cd $(ckm-idpath) ; }
ckm-ls(){  ls -l $(ckm-idpath) ; }

ckm-ip(){  ipython --pdb $(which ckm.py) -i $* ; }
ckm-ipl(){  ipython --pdb $(which ckmplot.py) -i $* ; }


ckm-info(){ cat << EOI

    ckm-dig                : $(ckm-dig) 
    ckm-key                : $(ckm-key)
    ckm-idpath, ckm-kcd    : $(ckm-idpath)
 
    ckm-ls : 
EOI
    ckm-ls


}


ckm-np(){ 
    ckm-kcd
    np.py source/evt/g4live/natural/{1,-1}/*.npy tests/CCerenkovGeneratorTest/*.npy
}


ckm-notes(){ cat << EON

--envkey 
     option makes executables sensitive to the OPTICKS_KEY envvar allowing 
     booting from the corresponding geocache 

--natural
     without this option the default is torch gensteps


EON
}



ckm-dbg(){
  if [ -n "$DEBUG" ] ; then
     case $(uname) in
        Darwin) echo lldb -- ;; 
        Linux) echo gdb --args ;;
     esac
  else
     echo -n
  fi
}



ckm-go(){ ckm-cd ; ./go.sh ; }

ckm--()
{ 
    local msg="=== $FUNCNAME :"

    opticks-
    local setup=$(opticks-prefix)/bin/opticks-setup.sh 
    [ ! -f $setup ] && echo "$msg MISSING setup script $setup : create with bash function opticks-setup-generate " && return 1   
    source $setup


    CerenkovMinimal  # NB the Opticks is embedded via G4OK : so commandline doesnt get thru 
}



ckm-export-gdml(){        G4OPTICKS_DEBUG="--dbggdmlpath /tmp/ckm.gdml" ckm-- ; }
ckm-run-bouncemax-zero(){ G4OPTICKS_DEBUG="--bouncemax 0" ckm-- ; }
ckm-run-lvsdname(){       G4OPTICKS_DEBUG="--lvsdname Det --args" ckm-- ; }

ckm-load(){      OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKTest --load --natural --envkey ;}
ckm-dump(){      OPTICKS_KEY=$(ckm-key) OpticksEventDumpTest --natural --envkey  ;}
ckm-res(){       OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OpticksResourceTest --natural --envkey ;}
ckm-okg4(){      OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --compute --envkey --embedded --save --natural  --args $*  ;}
ckm-okg4-dbg(){  DEBUG=1 ckm-okg4 $* ; } 




ckm-okg4-load(){ OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --load --envkey --embedded --natural ;}
ckm-mlib(){      OPTICKS_KEY=$(ckm-key) CMaterialLibTest --envkey  ;}
ckm-gentest(){   OPTICKS_KEY=$(ckm-key) $(ckm-dbg) CCerenkovGeneratorTest --natural --envkey ;}
ckm-okt(){       OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OpticksTest --natural --envkey ;}
ckm-viz(){       OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKTest --natural --envkey --xanalytic ; }
ckm-ggeotest(){  OPTICKS_KEY=$(ckm-key) $(ckm-dbg) GGeoTest --envkey ; }
## why is natural needed ? shouldnt that be apparent from the geocache source dir ?


ckm-gentest-notes(){ cat << EON

ckm-gentest
    CCerenkovGeneratorTest 

    1. loads geometry from geocache identified by OPTICKS_KEY envvar
    2. loads direct gensteps persisted into geocache and generates photons from them.  
    3. saved photons into geocache tests/CCerenkovGeneratorTest/so.npy 

    As shown below these photons from gensteps closely match those generated by 
    G4 normally and also by Opticks on GPU.  The matching is made possible because 
    of the CAlignEngine::SetSequenceIndex calls by the CCerenkovGenerator, 
    which causes the canned random sequences for each photon slot to be used.

    This matching is the justification for the two-executable debugging approach 
    of having the first minimally instrumented example executable paired with the 
    fully instrumented one for full on debugging. 
    

Comparing ckm-gentest photons with G4 generated ones. 
Uncomment the relevant B slot then run ckm-so::

    ckm-xx- comparing so.npy and so.npy between two dirs 
    pwd /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1
       Thu Mar 14 13:19:09 CST 2019
    a  -rw-r--r--  1 blyth  staff  14224 Mar 13 21:33 source/evt/g4live/natural/-1/so.npy
    b  -rw-r--r--  1 blyth  staff  14224 Mar 14 13:13 tests/CCerenkovGeneratorTest/so.npy
    a (221, 4, 4) 
    b (221, 4, 4) 
    max deviation 5.9604645e-08 
     deviations above cut 1e-05 num_wh 0
     deviations above cut 1e-06 num_wh 0
     deviations above cut 1e-07 num_wh 0
     deviations above cut 1e-08 num_wh 182
    0 1.4901161e-08 


Comparing bouncemax 0 Opticks GPU generated photons 
with the ckm-gentest ones, max deviation 6e-5::

    ckm-xx- comparing ox.npy and so.npy between two dirs 
    pwd /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1
       Thu Mar 14 13:26:59 CST 2019
    a  -rw-r--r--  1 blyth  staff  14224 Mar 14 13:24 source/evt/g4live/natural/1/ox.npy
    b  -rw-r--r--  1 blyth  staff  14224 Mar 14 13:13 tests/CCerenkovGeneratorTest/so.npy
    a (221, 4, 4) 
    b (221, 4, 4) 
    max deviation 6.1035156e-05 
     deviations above cut 1e-05 num_wh 39

    Observe that position and time are matched at 1e-10 level, 
    much better that direction, polarization, wavelength at 6e-5 level.


EON
}



ckm-addr2line()
{
    local addr=${1:-0x10002160e}
    PATH=/usr/bin lldb $(which CerenkovMinimal) -o "source list -a $addr"  --batch
}


ckm-pyrun-note(){ cat << EON

ckm-pyrun func args...
    generates a python script into a tmp directory from the output 
    of bash function "func-" into func.py where func is the first argument 
    to this bash function, then runs the script in ipython using the remainder 
    of the arguments

    See ckm-so for an example of usage

EON
}

ckm-pyrun(){
    local iwd=$PWD
    ckm-kcd      ## to the geocache dir 

    local func=$1
    mkdir -p $(ckm-tmp)
    local py=$(ckm-tmp)/$func.py
    $func- $* > $py 
    cat $py 
    ipython -i $py 

    cd $iwd
}

ckm-tag(){ echo 1 ; }
ckm-tmp(){   echo $TMP/ckm ; }

#############################################################


#ckm-a-dir(){   echo source/evt/g4live/natural/1 ; }  # ckm-- Opticks generated photons with bouncemax 0 to skip propagation
#ckm-a-name(){  echo ox.npy ; }         

ckm-a-dir(){   echo source/evt/g4live/natural/-1 ; }  # ckm-- standard G4 generated photons
ckm-a-name(){  echo so.npy ; }         

#############################################################

ckm-b-dir(){   echo tests/CCerenkovGeneratorTest ; }  # ckm-gentest "standalone" conversion of persisted direct gensteps into Cerenkov photons (unpropagated)
ckm-b-name(){  echo so.npy ; }

#ckm-b-dir(){   echo source/evt/g4live/natural/1 ; }  # ckm--
#ckm-b-name(){  echo ox.npy ; }

#############################################################

ckm-a(){ cd $(ckm-a-dir) ; } 
ckm-b(){ cd $(ckm-b-dir) ; } 

ckm-l(){
    local iwd=$PWD
    ckm-kcd 
    date

    echo A $(ls -l $(ckm-a-dir)/*.json)
    echo B $(ls -l $(ckm-b-dir)/*.json)

    cd $iwd
}

ckm-ls(){ 
    local iwd=$PWD
    ckm-kcd 
    date

    echo A $(ckm-a-dir)
    ls -l  $(ckm-a-dir) 
    np.py  $(ckm-a-dir)

    echo B $(ckm-b-dir)
    ls -l  $(ckm-b-dir) 
    np.py  $(ckm-b-dir)

    cd $iwd
}


ckm-xx-(){ 

    local apath=$(ckm-a-dir)/$1
    local bpath=$(ckm-b-dir)/$2

    echo \# $apath $bpath 

    cat << EOP
import numpy as np, commands
from collections import OrderedDict as odict
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=10000) 

apath = "$apath"
bpath = "$bpath"

print " $FUNCNAME comparing $1 and $2 between two dirs " 

print "pwd", commands.getoutput("pwd")
print "  ", commands.getoutput("date")
print "a ", commands.getoutput("ls -l %s" % apath)
print "b ", commands.getoutput("ls -l %s" % bpath)

a = np.load(apath)
b = np.load(bpath)

print "a %s " % repr(a.shape)
print "b %s " % repr(b.shape)

dv = np.max( np.abs(a[:,:3]-b[:,:3]), axis=(1,2) )

print "max deviation %s " % dv.max() 

cuts = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
wh = odict()

wc = 0 
for c in range(len(cuts)):
    wh[c] = np.where( dv > cuts[c] )[0] 
    if len(wh[c]) > 0 and wc == 0: wc = c
    print " deviations above cut %s num_wh %d wh[c][:10] %s  " % ( cuts[c], len(wh[c]), repr(wh[c][:10]) )
pass
print("--")
for c in range(len(cuts)):
    print " deviations above cut %s num_wh %d : showing first few " % ( cuts[c], len(wh[c]) )
    w = wh[c]
    for i in w[:2]:
        print i, dv[i], "\n",np.hstack([a[i,:3],(a[i,:3]-b[i,:3]),b[i,:3]])
    pass
pass

w = wh[wc][:10]
print(np.hstack([a[w,:3], b[w,:3], a[w,:3]-b[w,:3]]))


EOP
}

ckm-so-(){ ckm-xx-    $(ckm-a-name) $(ckm-b-name)  ; }
ckm-so(){  ckm-genrun $FUNCNAME ; }


ckm-so-notes(){ cat << EON

ckm-so-notes
================

Depending on the choice of inputs ckm-so does comparisons between any two of:

1. original full G4
2. standalone photo generation G4
3. Opticks generated photons (ckm-run-bouncemax-zero) 


Comparing Cerenkov generated photons between:

ckm-- OR ckm-go  
    CerenkovMinimal : geant4 example app, with genstep and photon collection
    via embedded Opticks with embedded commandline 
    " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0"  

    Note as "--bouncemax 0" prevents getting any hits, now have to use G4OPTICKS_DEBUG="--bouncemax 0" to
    add this to the G4Opticks embedded commandline. As done by: ckm-run-bouncemax-zero

    --bouncemax 0 
        means that photons are saved immediately after generation, with no propagation 

        * which means that GPU ox.npy can be matched with CPU so.npy 
   
    --printenabled --pindex 0
        dump kernel debug for photon 0 


    The big advantage of ckm-- is that it can look like any Geant4 example, that however 
    is also its biggest disadvantage in that this restricts it to minimally instrumented G4 
    as it does not make use of the CFG4/CRecorder : for full photon step recording : which 
    is the reason source/evt/g4live/natural/-1/ is rather empty compared
    to the fully featured source/evt/g4live/natural/1/


ckm-gentest : 
    CCerenkovGeneratorTest : genstep consuming standalone CPU generator that tries 
    (and succeeds at 1e-5 level in testing so far) to mimic the cerenkov process photons 
    via verbatim code copy 

    The motivation for arranging standalone G4 photon generation 
    in a "petri dish" is to isolate the photon simulation from everything 
    else in order to have a simple random number consumption environment 
    for alignment with Opticks.


cross exe, g4-g4 "same" sim : but with float genstep transport
----------------------------------------------------------------

Comparing photons from genstep 0, 
   source/evt/g4live/natural/-1/so.npy 
   tests/CCerenkovGeneratorTest/so.npy

* initially : small deviations at 1e-5 level mostly in wavelength
* fixed precision loss issue with wavelength and omission with time :
  bringing deviations down to an unfocussed 1e-8 level 


same ckm exe, cross sim G4/OK : with --bouncemax 0 for generation only comparison
------------------------------------------------------------------------------------------

Comparing photons from genstep 0, 
    source/evt/g4live/natural/-1/so.npy 
    source/evt/g4live/natural/1/ox.npy  

* initially same level of 1e-5 level deviations, mostly in wavelength 



EON
}

