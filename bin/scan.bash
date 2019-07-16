scan-source(){ echo $BASH_SOURCE ; }
scan-vi(){ vi $(scan-source)  ; }
scan-env(){  olocal- ; opticks- ; }
scan-usage(){ cat << EOU

scan
===================

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



EOU
}



#scan-mode(){ echo ${SCAN_MODE:-ph} ; }
scan-mode(){ echo ${SCAN_MODE:-ts} ; }
#scan-mode(){ echo ${SCAN_MODE:-tp} ; }

scan-ts-args(){  scan-seq ; }
scan-tp-args(){  scan-seq ; } 


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


scan-ph-args(){ cat << EOS | tr -d " ,"  | grep -v \#
          1
         10
        100
       1000
     10,000
    100,000
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
    analysis, using ta from opticks-tboolean-shortcuts
tv
    vizualization, using tv from opticks-tboolean-shortcuts


## to reduce logging output dramatically use "--error"

How to handle variations of a scan, eg changing RTX mode or numbers of GPUs


EON
}

scan-ph-lv(){ echo box ; }

scan-ph-cat(){
   case $cat in
       cvd_0_rtx_1) echo --cvd 0 --rtx 0  ;;  
       cvd_0_rtx_1) echo --cvd 0 --rtx 1  ;;  
       cvd_1_rtx_0) echo --cvd 1 --rtx 0  ;;  
       cvd_1_rtx_1) echo --cvd 1 --rtx 1  ;;  
       cvd_01_rtx_1) echo --cvd 0,1 --rtx 0  ;;  
   esac 
}

scan-ph-cats(){ cat << EOC
cvd_1_rtx_0
cvd_1_rtx_1
EOC
}

scan-ph-cmd(){   
   local num_photons=$1
   local cat=$2
   local cmd="ts $(scan-ph-lv) --pfx scan-ph --cat $cat --generateoverride ${num_photons} --compute --production  "  ; 
   if [ $num_photons -gt 1000000 ]; then
       cmd="$cmd --nog4propagate"  
   fi
   cmd="$cmd $(scan-ph-cat $cat)"
   echo $cmd
}

scan-ts-cmd(){   printf "ts $1            --pfx scan-ts --generateoverride -1 --cvd 1 --rtx 1 --compute\n" ; }
scan-tp-cmd(){   printf "tp $1 \n" ; }
scan-tv-cmd(){   printf "tv $1 \n" ; }

scan-ph-post(){  scan.py $TMP/tboolean-$(scan-ph-lv) ; }
scan-ts-post(){  absmry.py  ; }


scan-cmds-all(){
   local mode=$(scan-mode)
   local cmd
   local arg
   local cat 
   scan-$mode-cats | while read cat 
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


scan-ph-cmds(){ SCAN_MODE=ph scan-cmds-all ; }
scan-ph(){      SCAN_MODE=ph scan-- ; }
scan-ph-v(){    VERBOSE=1 scan-ph ; }



