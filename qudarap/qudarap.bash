qudarap-source(){   echo $BASH_SOURCE ; }
qudarap-vi(){       vi $(qudarap-source) ; }
qudarap-usage(){ cat << "EOU"
QUDARap
==========

EOU
}

qudarap-env(){      
  olocal-  
}


qudarap-idir(){ echo $(opticks-idir); }
qudarap-bdir(){ echo $(opticks-bdir)/qudarap ; }
qudarap-sdir(){ echo $(opticks-home)/qudarap ; }
qudarap-tdir(){ echo $(opticks-home)/qudarap/tests ; }

qudarap-c(){    cd $(qudarap-sdir)/$1 ; }
qudarap-cd(){   cd $(qudarap-sdir)/$1 ; }
qudarap-scd(){  cd $(qudarap-sdir); }
qudarap-tcd(){  cd $(qudarap-tdir); }
qudarap-bcd(){  cd $(qudarap-bdir); }




qudarap-prepare-installation-notes(){ cat << EON
qudarap-prepare-installation-notes
---------------------------------------

**FOLLOWING THE MOVE TO PHILOX THIS IS NO LONGER NEEDED**

If you have lots of VRAM you can increase the 
maximum possible launch size by expanding the CHUNKSIZES.  
For example the initial 10x1M,9x10M,5x20M corresponds 
to 200M RNG in 24 file chunks.
That could be expanded to 300M with::

    10x1M,9x10M,10x20M

NB you will get asserts if the change does not 
ONLY expand the chunks. 

To use different chunking delete all chunks and run again. 

EON
}



qudarap-prepare-installation()
{
   # these are the defaults anyhow
   export SCurandState__init_SEED_OFFSET=0:0
   export SCurandState__init_CHUNKSIZES=10x1M,9x10M,5x20M

   ${OPTICKS_PREFIX}/lib/QCurandStateTest

   return $? 
}

qudarap-check-installation()
{
   echo -n
}


qudarap-divider()
{
   cat << EON
===========================================================================================================
  BELOW IS FOR THE OLD MONOLITHIC IMPL : LIKELY TO BE REMOVED ONCE CHUNKED APPROCH HAS BEEN FULLY TESTED
===========================================================================================================
EON
}











qudarap-prepare-sizes-Linux-(){  echo ${OPTICKS_QUDARAP_RNGMAX:-1,3,10} ; }
qudarap-prepare-sizes-Darwin-(){ echo ${OPTICKS_QUDARAP_RNGMAX:-1,3} ; }
qudarap-prepare-sizes(){ $FUNCNAME-$(uname)- | tr "," "\n"  ; }
qudarap-rngdir(){ echo $(opticks-rngdir) ; }


qudarap-prepare-installation-old-notes(){ cat << EON
qudarap-prepare-installation-old-notes
---------------------------------------

See::

    qudarap/QCurandStateMonolithic.{hh,cc}
    sysrap/SCurandStateMonolithic.{hh,cc}

NB changing the below envvars can adjust the QCurandStateMonolithic_SPEC::

   QUDARAP_RNG_SEED 
   QUDARAP_RNG_OFFSET 

But doing this is for expert usage only because it will then 
be necessary to set QCurandStateMonolithic_SPEC correspondingly
when running Opticks executables for them to find
the customized RNG files. 

HMM : THIS AWKWARDNESS SUGGESTS SPLITTING THE size and the seed:offset config

EON
}




qudarap-prepare-installation-old()
{
   local sizes=$(qudarap-prepare-sizes)
   local size 
   local seed=${QUDARAP_RNG_SEED:-0}
   local offset=${QUDARAP_RNG_OFFSET:-0}
   for size in $sizes ; do 
       QCurandStateMonolithic_SPEC=$size:$seed:$offset  ${OPTICKS_PREFIX}/lib/QCurandStateMonolithicTest
       rc=$? ; [ $rc -ne 0 ] && return $rc
   done
   return 0 
}

qudarap-check-installation-old()
{
   local msg="=== $FUNCNAME :"
   local rc=0
   qudarap-check-rngdir-
   rc=$? ; [ $rc -ne 0 ] && return $rc

   local sizes=$(qudarap-prepare-sizes)
   local size 
   for size in $sizes ; do 
       QCurandStateMonolithic_SPEC=$size:0:0  qudarap-check-rngpath-
       rc=$? 
       [ $rc -ne 0 ] && return $rc
   done
   return $rc
}

qudarap-check-rngdir-()
{
    local rngdir=$(qudarap-rngdir)
    local rc=0
    local err=""
    [ ! -d "$rngdir" ] && rc=201 && err=" MISSING rngdir $rngdir " 
    echo $msg $rngdir $err rc $rc
    return $rc 
}

qudarap-parse(){ 
    : parse the spec envvar returning num/seed/offset 
    local defspec=1:0:0
    local spec=${QCurandStateMonolithic_SPEC:-$defspec} 
    local qty=${1:-num}

    local num
    local seed
    local offset 

    IFS=: read -r num seed offset <<< "$spec"

    [ $num -le 100 ] && num=$(( $num*1000*1000 ))

    case $qty in
       num)    echo $num ;;
       seed)   echo $seed ;;
       offset) echo $offset ;;
    esac
}
qudarap-num(){    qudarap-parse num ; }
qudarap-seed(){   qudarap-parse seed ; }
qudarap-offset(){ qudarap-parse offset ; }

qudarap-rngname(){ echo QCurandStateMonolithic_$(qudarap-num)_$(qudarap-seed)_$(qudarap-offset).bin ; }

qudarap-rngname-1(){ QCurandStateMonolithic_SPEC=1:0:0 qudarap-rngname ; }
qudarap-rngname-3(){ QCurandStateMonolithic_SPEC=3:0:0 qudarap-rngname ; }

qudarap-rngpath(){ echo $(qudarap-rngdir)/$(qudarap-rngname) ; }

qudarap-check-rngpath-()
{
    local rc=0
    local path=$(qudarap-rngpath)
    local err=""
    [ ! -f $path ] && rc=202 && err=" MISSING PATH $path " 
    echo $msg $path $err rc $rc
    return $rc
}


