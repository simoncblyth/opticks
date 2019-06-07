#!/bin/bash -l
[ "$0" == "$BASH_SOURCE" ] && sauce=0 || sauce=1

o-(){    . $(which o.sh) ; } 
o-vi(){ vi $(which o.sh) ; } 

cmdline="$*"


o-usage(){ cat << \EOU
o.sh
======

Gathering the useful parts of op.sh prior to dumping that.


EOU
}

o-cmdline-parse()
{
   local msg="=== $FUNCNAME $VERBOSE :"
   [ -n "$VERBOSE" ] && echo $msg START 

   o-cmdline-specials
   o-cmdline-binary-match
   o-cmdline-binary
}


o-binary-name-default(){ echo OKG4Test ; }
o-binary-names(){ type o-binary-name | perl -ne 'm,--(\w*)\), && print "$1\n" ' - ; } 
o-binary-name()
{
   case $1 in 
           --okg4) echo OKG4Test ;;
         --tracer) echo OTracerTest ;;
   esac 
   # no default as its important this return blank for unidentified commands
}

o-cmdline-specials()
{
   local msg="=== $FUNCNAME $VERBOSE :"
   [ -n "$VERBOSE" ] && echo $msg 

   unset OPTICKS_DBG 
   unset OPTICKS_LOAD
   unset OPTIX_API_CAPTURE

   if [ "${cmdline/--malloc}" != "${cmdline}" ]; then
       export OPTICKS_MALLOC=1
   fi
   if [ "${cmdline/--debugger}" != "${cmdline}" ]; then
       export OPTICKS_DBG=1
   fi
   if [ "${cmdline/--strace}" != "${cmdline}" ]; then
       export OPTICKS_DBG=2
   fi

   if [ "${cmdline/-D}" != "${cmdline}" ]; then 
       export OPTICKS_DBG=1
   fi

   if [ "${cmdline/--load}" != "${cmdline}" ]; then
       export OPTICKS_LOAD=1
   fi
   if [ "${cmdline/--oac}" != "${cmdline}" ]; then
       export OPTIX_API_CAPTURE=1
   fi
}

o-cmdline-binary-match()
{
    local msg="=== $FUNCNAME $VERBOSE :"
    [ -n "$VERBOSE" ] && echo $msg finding 1st argument with associated binary 

    local arg
    local bin
    unset OPTICKS_CMD

    for arg in $cmdline 
    do
       bin=$(o-binary-name $arg)
       #echo arg $arg bin $bin  
       if [ "$bin" != "" ]; then 
           export OPTICKS_CMD=$arg
           echo $msg $arg
           return 
       fi
    done

    if [ -z "$OPTICKS_CMD" ]; then
       echo $msg ERR UNSET 
    fi 
}

o-cmdline-binary()
{
   unset OPTICKS_BINARY 
   unset OPTICKS_ARGS

   local cfm=$OPTICKS_CMD
   local bin=$(o-binary-name $cfm) 
   local def=$(o-binary-name-default)

   if [ "$bin" == "" ]; then
      bin=$def
   fi 

   export OPTICKS_BINARY=$(which $bin)
   export OPTICKS_ARGS=$cmdline
}

o-runline-notes(){ cat << EON

Use "--debugger" option to set the intername envvar OPTICKS_DBG

EON
}

o-runline()
{
   local runline
   if [ "${OPTICKS_BINARY: -3}" == ".py" ]; then
      runline="python ${OPTICKS_BINARY} ${OPTICKS_ARGS} "
   elif [ "${OPTICKS_DBG}" == "1" ]; then 
      case $(uname) in
               *) runline="gdb  --args ${OPTICKS_BINARY} ${OPTICKS_ARGS} " ;;
      esac
   elif [ "${OPTICKS_DBG}" == "2" ]; then 
      runline="strace -o /tmp/strace.log -e open ${OPTICKS_BINARY} ${OPTICKS_ARGS}" 
   else
      runline="${OPTICKS_BINARY} ${OPTICKS_ARGS}" 
   fi
   echo $runline
}

o-postline()
{
   local postline
   if [ "${OPTICKS_DBG}" == "2" ]; then 
       postline="strace.py -f O_CREAT"  
   else
       postline="echo $FUNCNAME : dummy"
   fi
   echo $postline 
}


o-malloc()
{
   export MallocStackLoggingNoCompact=1   # all allocations are logged
   export MallocScribble=1     # free sets each byte of every released block to the value 0x55.
   export MallocPreScribble=1  # sets each byte of a newly allocated block to the value 0xAA
   export MallocGuardEdges=1   # adds guard pages before and after large allocations
   export MallocCheckHeapStart=1 
   export MallocCheckHeapEach=1 
}
o-unmalloc()
{
   unset MallocStackLoggingNoCompact
   unset MallocScribble
   unset MallocPreScribble
   unset MallocGuardEdges
   unset MallocCheckHeapStart
   unset MallocCheckHeapEach
}

o-main()
{
   local msg="=== $FUNCNAME :"
   o-cmdline-parse

   local runline=$(o-runline)
   local postline=$(o-postline)

   echo $msg $runline ======= PWD $PWD 
   eval $runline
   RC=$?
   echo $msg $runline ======= PWD $PWD  RC $RC 

   [ $RC -eq 0 ] && echo $postline && eval $postline 

   exit $RC
}

o-main


