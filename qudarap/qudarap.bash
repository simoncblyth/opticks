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

qudarap-prepare-sizes-Linux-(){  echo ${OPTICKS_QUDARAP_RNGMAX:-1,3,10} ; }
qudarap-prepare-sizes-Darwin-(){ echo ${OPTICKS_QUDARAP_RNGMAX:-1,3} ; }
qudarap-prepare-sizes(){ $FUNCNAME-$(uname)- | tr "," "\n"  ; }

qudarap-prepare-installation-notes(){ cat << EON
qudarap-prepare-installation-notes
-----------------------------------

See::

    qudarap/QCurandState
    sysrap/SCurandState

NB changing the below envvars can adjust the QCurandState_SPEC::

   QUDARAP_RNG_SEED 
   QUDARAP_RNG_OFFSET 

But doing this is for expert usage only because it will then 
be necessary to set QCurandState_SPEC correspondingly
when running Opticks executables for them to find
the customized curandState files. 

EON
}

qudarap-prepare-installation()
{
   local sizes=$(qudarap-prepare-sizes)
   local size 
   local seed=${QUDARAP_RNG_SEED:-0}
   local offset=${QUDARAP_RNG_OFFSET:-0}
   for size in $sizes ; do 
       QCurandState_SPEC=$size:$seed:$offset  ${OPTICKS_PREFIX}/lib/QCurandStateTest
       rc=$? ; [ $rc -ne 0 ] && return $rc
   done
   return 0 
}

qudarap-rngdir(){ echo $(opticks-rngdir) ; }

qudarap-check-installation()
{
   local msg="=== $FUNCNAME :"
   local rc=0
   qudarap-check-rngdir-
   rc=$? ; [ $rc -ne 0 ] && return $rc

   local sizes=$(qudarap-prepare-sizes)
   local size 
   for size in $sizes ; do 
       QCurandState_SPEC=$size:0:0  qudarap-check-rngpath-
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
    local defspec=1:0:0
    local spec=${QCurandState_SPEC:-$defspec} 
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
qudarap-num(){ qudarap-parse num ; }
qudarap-rngname(){ echo QCurandState_$(qudarap-num)_0_0.bin ; }
qudarap-rngname-1(){ QCurandState_SPEC=1:0:0 qudarap-rngname ; }
qudarap-rngname-3(){ QCurandState_SPEC=3:0:0 qudarap-rngname ; }
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


