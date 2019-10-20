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

scan-source(){ echo $BASH_SOURCE ; }
scan-vi(){ vi $(scan-source)  ; }
#scan-env(){  olocal- ; opticks- ; }
scan-env(){  echo -n ; }
scan-usage(){ cat << EOU

scan
===================

Usage
------

::

   scan-smry 0 --pfx scan-px    # summary of the scan with prefix scan-px-0
   scan-ismry 0 --pfx scan-px   # summary with ipython, giving interactive access to objects

   scan-smry 10 11 --pfx scan-ph   # summary of scans with prefix scan-ph-10 scan-ph-11

   scan-plot 0 --pfx scan-px 
   scan-plot 0 --pfx scan-pf



Scan Modes
------------

ts
    bi-simulation of test geometry  

tp
    python re-analysis of events saved by ts, 
    this is the non-ipython version of ta    


Workflow
-----------

1. set the scan mode 

2. view the commandlines with : scan-;scan-cmds-all

3. adjust the options, changing eg scan-proxy-cmd

4. check the machinery with a one or two cmdlines eg with scan-cmds(){ scan-cmds-all | head -1 ; }
   and verbose scanning:: 

       scan-;VERBOSE=1 scan--


Migration from TMP to OPTICKS_EVENT_BASE for scan events
-------------------------------------------------------------

Have been using the below for a while::

   export OPTICKS_INSTALL_PREFIX=$LOCAL_BASE/opticks
   export TMP=$OPTICKS_INSTALL_PREFIX/tmp
   export OPTICKS_EVENT_BASE=$OPTICKS_INSTALL_PREFIX/tmp

But this mingles scan result folders with a great deal of 
output from opticks-t tests that changes frequently, 
so change event base to::

   export OPTICKS_EVENT_BASE=$OPTICKS_INSTALL_PREFIX/evtbase

And move the results over::

    [blyth@localhost opticks]$ mv tmp/scan-ph-? evtbase/
    [blyth@localhost opticks]$ mv tmp/scan-ph-?? evtbase/



EOU
}



#scan-mode(){ echo ${SCAN_MODE:-ph} ; }
scan-mode(){ echo ${SCAN_MODE:-ts} ; }
#scan-mode(){ echo ${SCAN_MODE:-tp} ; }

scan-ts-args(){  scan-seq ; }
scan-tp-args(){  scan-seq ; } 
scan-ph-args(){  scan-numphoton ; } 
scan-px-args(){  scan-numphoton ; } 
scan-pf-args(){  scan-numphoton ; } 
scan-pt-args(){  scan-numphoton ; } 


scan-seq-notes(){ cat << EON
scan-seq
---------

Sequence of integers, controllable via SLI envvar 
in python slice style 

   SLI=:5 scan-cmds     # first five seq:  0 1 2 3 4 
   SLI=5: scan-cmds     # seq from 5 onwards

   SLI=5: scan--v 

EON
}

scan-seq(){  
  local start_=${1:-0}
  local stop_=${2:-39}

  if [ -n "$SLI" ] ; then 
     sstart=${SLI/:*}
     sstop=${SLI/*:}
     if [ -n "$sstart" ]; then 
        start_=$sstart
     fi
     if [ -n "$sstop" ]; then 
        sstop=$(( $sstop - 1 ))   # python one beyond convention differs from seq inclusive 
        stop_=$sstop
     fi
  fi 
  seq $start_ $stop_
}


scan-numphoton-big(){ cat << EOS | tr -d " ,"  | grep -v \#
200,000,000
400,000,000
EOS
}

scan-numphoton-fine(){ cat << EOS | tr -d " ,"  | grep -v \#
  1,000,000
 10,000,000
 20,000,000
 30,000,000
 40,000,000
 50,000,000
 60,000,000
 70,000,000
 80,000,000
 90,000,000
100,000,000
EOS
}

scan-numphoton(){ cat << EOS | tr -d " ,"  | grep -v \#
  1,000,000
 10,000,000
100,000,000
EOS
}





#scan-proxy-cmd(){   printf "env LV=%s tboolean.sh --compute --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero   \n" $1 ; }

scan-cmd-notes(){ cat << EON
$FUNCNAME
=====================

Alignment options "--align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero"
are now done by default in tboolean.sh tboolean-lv


ph
    scanning generateoverride photons
ts
    bi-simulation, using ts from opticks-tboolean-shortcuts
ta
    python analysis, using ta from opticks-tboolean-shortcuts

tv
    vizualization, using tv from opticks-tboolean-shortcuts


## to reduce logging output dramatically use "--error"

How to handle variations of a scan, eg changing RTX mode or numbers of GPUs


EON
}


scan-cat(){
   case $cat in
       cvd_0_rtx_0) echo --cvd 0 --rtx 0  ;;  
       cvd_0_rtx_1) echo --cvd 0 --rtx 1  ;;  
       cvd_0_rtx_2) echo --cvd 0 --rtx 2  ;;  
       cvd_1_rtx_0) echo --cvd 1 --rtx 0  ;;  
       cvd_1_rtx_1) echo --cvd 1 --rtx 1  ;;  
       cvd_1_rtx_2) echo --cvd 1 --rtx 2  ;;  
       cvd_01_rtx_0) echo --cvd 0,1 --rtx 0  ;;  
       cvd_01_rtx_1) echo --cvd 0,1 --rtx 1  ;;  
       cvd_01_rtx_2) echo --cvd 0,1 --rtx 2  ;;  
   esac 
}

scan-cats(){
  if [ "$(scan-pfx)" == "scan-ph-13" ] ; then 
      scan-cats-tri  
      #echo cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_2
  else
      case $(scan-mode) in 
         pt) scan-cats-tri ;;
          *) scan-cats-ana ;;  
      esac
  fi

}


scan-cats-ana(){ cat << EOC
cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_0
cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_1
EOC
}
scan-cats-tri(){ cat << EOC
cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_0
cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_1
cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_2
EOC
}






scan-cats_in_waiting(){ cat << EOC
cvd_0_rtx_0
cvd_0_rtx_1
cvd_1_rtx_0
cvd_1_rtx_1
EOC
}



scan-ph-cmd-notes(){ cat << EON

--multievent fails without --nog4propagate 

EON
}



scan-num(){  python -c "from opticks.ana.num import Num ; print(Num.String($1))" ; }

scan-rngmax-opt-notes(){ cat << EON

* this simple approach of a range of fixed sizes means
  that will almost always be using a lot more memory 
  for rng_states than is necessary 

* better to use the building block approach with each block 
  corresponding to 10M slots 


EON
}


scan-rngmax-opt(){ 
   local num_photons=${1:-0}

   local M=$(( 1000000 ))
   local M3=$(( 3*M ))
   local M10=$(( 10*M ))
   local M100=$(( 100*M ))
   local M200=$(( 200*M ))
   local M400=$(( 400*M ))

   local opt

   if [ $num_photons -gt $M400 ]; then
      echo $msg num_photons $num_photons is above the ceiling 
   elif [ $num_photons -gt $M200 ]; then
       opt="--rngmax 400"
   elif [ $num_photons -gt $M100 ]; then
       opt="--rngmax 200"
   elif [ $num_photons -gt $M10 ]; then 
       opt="--rngmax 100"
   elif [ $num_photons -gt $M3 ]; then 
       opt="--rngmax 10"
   else
       opt="--rngmax 3"
   fi
   echo $opt
}



scan-ph-note(){ bashnotes.py ${1:-$(scan-vers)} --bashcmd "scan-;scan-ph-notes" ; }
scan-ph-notes(){ cat << EON
0
   Silver:Quadro_RTX_8000 : full scan with old driver, seemed not to be able to switch on RTX
   but may have been caused by a script bug
1  
   Silver:Quadro_RTX_8000 with the 435.21 driver, fixed a script bug, but even after that it 
   seems RTX not doing anything 
2
   Silver:Quadro_RTX_8000 with 435.21 driver and WITH_LOGDOUBLE commented, reducing the f64 
   did this in two goes, with some doubling up : that might have caused 
   glitch on the first 1M point
3
   Silver:Quadro_RTX_8000, 435.21 driver, OptiX_600, WITH_LOGDOUBLE commented, all at once 
4
   Gold:TITAN_RTX 418.56:OptiX 600:WITH_LOGDOUBLE commented : NB have to flip the cvd in the cats to 1  
5
   Gold:TITAN_RTX 418.56:OptiX 600:WITH_LOGDOUBLE enabled : NB check cvd in cats is 1 to pick TITAN_RTX 
6
   Silver:Quadro_RTX_8000, 435.21 drive, OptiX_650, WITH_LOGDOUBLE commented
7
   Gold:TITAN_RTX 418.56:OptiX 600:WITH_LOGDOUBLE commented:LEGACY_ENABLED:
   reproducibility check following other okdist developments before updating driver and OptiX
8
   Gold:TITAN_RTX 435.21:OptiX 650:WITH_LOGDOUBLE commented:LEGACY_ENABLED:
   after updating driver and OptiX
9
   Gold:TITAN_RTX 435.21:OptiX 650:WITH_LOGDOUBLE enabled:LEGACY_ENABLED:
10
   Gold:TITAN_RTX 435.21:OptiX 650:WITH_LOGDOUBLE enabled:LEGACY_ENABLED:
   after removing 67.1M ceiling from the cycling of unsigned long 
11
   Silver:Quadro_RTX_8000 435.21:OptiX 650:WITH_LOGDOUBLE enabled:LEGACY_ENABLED:
   after removing 67.1M ceiling from the cycling of unsigned long 
12
   Gold:TITAN RTX repro
13
   Gold:TITAN RTX with --xtriangle and RTX 0/1/2




To check switches : OpticksSwitchesTest

EON
}



scan-px-note(){ bashnotes.py ${1:-$(scan-vers)} --bashcmd "scan-;scan-px-notes" ; }
scan-px-notes(){ cat << EON
0
   Gold:TITAN_RTX checking torchconfig based GPU generation of photons with tboolean-interlocked 
1
   Gold:TITAN_RTX checking torchconfig based GPU generation of photons with tboolean-interlocked 
   reproducibility check 
2
   Silver:Quadro_RTX_8000 checking torchconfig based GPU generation of photons with tboolean-interlocked 
   interlocked is heavy
3
   Gold:TITAN_RTX checking torchconfig based GPU generation of photons with tboolean-boxx
4
   Silver:Quadro_RTX_8000 torchconfig tboolean-boxx push to 400M
   scan-px-4/cvd_0_rtx_1  

EON
}

scan-pf-note(){ bashnotes.py ${1:-$(scan-vers)} --bashcmd "scan-;scan-pf-notes" ; }
scan-pf-notes(){ cat << EON
0
   Gold:TITAN_RTX checking torchconfig based GPU generation of photons with OKTest targeting JUNO CD center
   Former hardcoded dbghitmask TO,BT,SC,SA yields too many photons (30%) 
   so mask it more difficult with a reemission TO,BT,RE,SC,SA
1
   Silver:Quadro RTX 8000 : up to 400M
2
   Gold:TITAN_RTX with v6 geocache, with SD



EON
}

scan-pf-notes-extra(){ cat << EON

OKG4Test 239s for 1M
-------------------------

To get the G4 times for use by profile.py FromExtrapolation 
switch OKTest to OKG4Test : and use profile.py : tis a bit manual 
(next time reduce multievent 10 to ~4)

::

    ip profile.py --cat cvd_1_rtx_0_1M --pfx scan-pf-0 --tag 0
         OKG4Test run  

scan-pf-0 OKG4Test 239s for 1M::

    In [17]: ap.times("CRunAction::BeginOfRunAction")
    Out[17]: array([1206.6406, 1464.2812, 1708.2578, 1950.6406, 2191.8984, 2439.336 , 2681.1562, 2916.8828, 3153.2656, 3389.5   ], dtype=float32)

    In [18]: ap.times("CRunAction::EndOfRunAction")
    Out[18]: array([1460.2266, 1706.0625, 1948.4453, 2189.6719, 2436.9219, 2678.9219, 2914.6875, 3151.0625, 3387.3281, 3622.4453], dtype=float32)

    In [19]: ap.times("CRunAction::EndOfRunAction") - ap.times("CRunAction::BeginOfRunAction")
    Out[19]: array([253.5859, 241.7812, 240.1875, 239.0312, 245.0234, 239.5859, 233.5312, 234.1797, 234.0625, 232.9453], dtype=float32)

    In [25]: np.average( ap.times("CRunAction::EndOfRunAction") - ap.times("CRunAction::BeginOfRunAction") )
    Out[25]: 239.3914


EON
}













scan-vers(){ echo ${SCAN_VERS:-2} ; }



scan-smry(){ profilesmry.py ${1:-$(scan-vers)} ${@:2} ; }
scan-ismry(){ ipython --pdb -i -- $(which profilesmry.py)     ${1:-$(scan-vers)} ${@:2} ; }
scan-plot(){  ipython --pdb -i -- $(which profilesmryplot.py) ${1:-$(scan-vers)} ${@:2}  ; }
scan-pfx(){  echo scan-$(scan-mode)-$(scan-vers) ; }


scan-xx-cmd-notes(){ cat << "EON"
$FUNCNAME
==================

For running with more than 3M photons it is necessary to create two rngmax curandState 
files with::

    cudarap-prepare-installcache-10M
    cudarap-prepare-installcache-100M

Note that the layout of the *cat* option is parsed by ana/profilesmry.py 
to extract the number of photons and that *cat* is used as the dictionary 
key for parsed profile instances.

Running stack topdown::

    scan-<mode> 
    scan-- 
    scan-cmds-all
    scan-<mode>-cmd <num_photons> <cat>

Listing commands stack topdown::

    scan-<mode>- 
    scan-cmds-all
    scan-<mode>-cmd <num_photon> <cat>


scan-px-cmd
    the geometry used by scan-px should not use emitter sources, pick an older
    one using the default torchconfig approach to configure on GPU photon 
    generation of test sources

    TODO: make tboolean-boxx so can compare the approaches

    Use of --oktest switches to OKTest executable and hence no alignment stuff in tboolean-lv


EON
}

scan-ph-lv(){ echo box ; }
scan-ph-cmd(){   
   local num_photons=$1
   local cat=$2
   local num_abbrev=$(scan-num $num_photons)
   local cmd="ts $(scan-ph-lv) --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --production --savehit --multievent 10 --xanalytic "  ; 
   cmd="$cmd --nog4propagate $(scan-rngmax-opt $num_photons) $(scan-cat $cat)"

   if [ "$(scan-pfx)" == "scan-ph-13" ]; then 
      cmd="$cmd --xtriangle"
   fi   
   echo $cmd
}

#scan-px-lv(){ echo interlocked ; }   
scan-px-lv(){ echo boxx ; }   
scan-px-cmd(){
   local num_photons=$1
   local cat=$2
   local num_abbrev=$(scan-num $num_photons)
   local cmd="ts $(scan-px-lv) --oktest --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --production --savehit --multievent 10 --xanalytic "  ; 
   cmd="$cmd $(scan-rngmax-opt $num_photons) $(scan-cat $cat)"
   echo $cmd
}


scan-pf-cmd(){
   local num_photons=$1
   local cat=$2
   local num_abbrev=$(scan-num $num_photons)
   local cmd="OKTest --target 62590  --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --save --production --savehit --dbghitmask TO,BT,RE,SC,SA --multievent 10 --xanalytic " 
   cmd="$cmd $(scan-rngmax-opt $num_photons) $(scan-cat $cat)"
   echo $cmd
}

scan-pf-check(){  OKTest --target 62590 --generateoverride -10 --rngmax 10 --cvd 1 --rtx 1 --xanalytic ; }


scan-pt-cmd(){
   local num_photons=$1
   local cat=$2
   local num_abbrev=$(scan-num $num_photons)
   local cmd="OKTest --target 62590  --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --save --production --savehit --dbghitmask TO,BT,RE,SC,SA --multievent 10  " 
   cmd="$cmd $(scan-rngmax-opt $num_photons) $(scan-cat $cat)"
   echo $cmd
}




scan-ts-cmd(){   echo ts $1 --pfx $(scan-pfx) --generateoverride -1 --cvd 1 --rtx 1 --compute --recpoi --utaildebug --xanalytic ; }
scan-tp-cmd(){   echo tp $1 --pfx $(scan-pfx) --msli :1M  ; }
scan-tv-cmd(){   echo tv $1 ; }

scan-ph-post(){  scan.py $TMP/tboolean-$(scan-ph-lv) ; }
scan-ts-post(){  absmry.py  ; }

scan-rsync(){ 
   [ -z "$OPTICKS_EVENT_BASE" ] && echo OPTICKS_EVENT_BASE not defined && return 0  

   local src=$OPTICKS_EVENT_BASE
   local dst=P:$OPTICKS_EVENT_BASE
cat << EOC
rsync -av $src/ $dst 
EOC
}


scan-cmds-all-notes(){ cat << EOD

This is invoked from mode specific functions that de

EOD
}

scan-cmds-all(){
   local mode=$(scan-mode)
   local cmd
   local arg
   local cat 
   scan-cats | while read cat 
   do
       scan-$mode-args | while read arg
       do 
          scan-$mode-cmd $arg $cat
       done
   done
}


scan--()
{
   local cmd
   scan-cmds-all | while read cmd
   do
      local rc
      if [ -n "$VERBOSE" ]; then
          echo $cmd
          $cmd 
          rc=$?
      else
          $cmd > /dev/null 2>&1  
          rc=$?
      fi 
      printf " %20s : %40s ======= RC %3d  RC 0x%.2x \n"  $FUNCNAME "$cmd" $rc $rc  
   done
}


scan-info(){ cat << EOI

    OPTICKS_INSTALL_PREFIX : $OPTICKS_INSTALL_PREFIX
    OPTICKS_EVENT_BASE     : $OPTICKS_EVENT_BASE
    TMP                    : $TMP

    scan-vers              : $(scan-vers)

EOI
    scan-v
}



scan-ph-(){     SCAN_MODE=ph scan-cmds-all ; }
scan-ph(){      SCAN_MODE=ph scan-- ; }
scan-ph-v(){    VERBOSE=1 OpticksProfile=ERROR scan-ph ; }

scan-px-(){     SCAN_MODE=px scan-cmds-all ; }
scan-px(){      SCAN_MODE=px scan-- ; }
scan-px-v(){    VERBOSE=1 OpticksProfile=ERROR scan-px ; }

scan-pf-(){     SCAN_MODE=pf scan-cmds-all ; }
scan-pf(){      SCAN_MODE=pf scan-- ; }
scan-pf-v(){    VERBOSE=1 OpticksProfile=ERROR scan-pf ; }

scan-pt-(){     SCAN_MODE=pt scan-cmds-all ; }
scan-pt(){      SCAN_MODE=pt scan-- ; }
scan-pt-v(){    VERBOSE=1 OpticksProfile=ERROR scan-pt ; }






scan-tp-(){     SCAN_MODE=tp scan-cmds-all ; }
scan-tp(){      SCAN_MODE=tp scan-- ; }
scan-tp-v(){    VERBOSE=1 scan-tp ; }

scan-ts-(){     SCAN_MODE=ts scan-cmds-all ; }
scan-ts(){      SCAN_MODE=ts scan-- ; }
scan-ts-v(){    VERBOSE=1 scan-ts ; }


scan-pubroot(){ echo ${SCAN_PUBROOT:-/Users/blyth/simoncblyth.bitbucket.io} ; }
scan-pubdir(){ echo ${SCAN_PUBDIR:-env/presentation/ana} ; }
scan-pub(){
   local msg="# $FUNCNAME :"
   local scanid=${1:-scan-pf-0}
   local pubdir=$(scan-pubroot)/$(scan-pubdir)
   [ ! -d "$pubdir" ] && echo $msg pubdir $pubdir does not exist && return 1

   echo $msg pipe to shell  
   cat << EOC
   scp -r J:local/opticks/tmp/ana/$scanid $pubdir/
EOC

}

scan-pubrst--(){
   local scanid=${1:-scan-pf-0}
   local pubdir=$(scan-pubroot)/$(scan-pubdir)
 
   cd $(scan-pubroot)
   ls -1 $(scan-pubdir)/$scanid/*.png

}

scan-pubrst-notes(){ cat << EON

Prepare RST source for including plots in presentations.::

   scan-pubrst scan-pf-1

EON
}

scan-pubrst-(){
   local path
   local name
   local stem
   local parent
   local indent="    "
   local key 

   printf "\n" 

   if [ "$PASS" == "1" ]; then
      printf ".. comment\n\n"
   fi

   $FUNCNAME- $* | while read path ; do 
       name=$(basename $path)
       stem=${name/.png}
       parent=$(basename $(dirname $path))
       key=${parent}_${stem}

       if [ "$PASS" == "0" ]; then 
           printf "%s%s\n" "$indent" $key
           printf "%s/%s %s\n" "$indent" $path "1280px_720px"
           printf "\n" 
       elif [ "$PASS" == "1" ]; then 
           printf "%s:i:\`%s\`\n" "$indent" $key
       fi 
   done

   if [ "$PASS" == "1" ]; then
      printf "\n"
   fi
}

scan-pubrst(){
   PASS=0 $FUNCNAME- $*
   PASS=1 $FUNCNAME- $*
}


