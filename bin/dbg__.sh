#!/bin/bash


dbg__old()
{
    : opticks/bin/dbg__.sh

    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}

dbg__()
{
    : opticks/bin/dbg__.sh
    : eg BP="junoHit_PMT::operator delete" ipc

    local H=""
    local B=""
    local T=""
    local X=""


    if [ -n "$BP" ]; then
        H="-ex \"set breakpoint pending on\"";
        : split BP on comma delimiter preserving spaces within fields
        readarray -td, bps <<<"$BP"
        for bp in "${bps[@]}"
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" ";
    fi;

    [ -n "$BP_SEventConfig__GLOBAL" ] && B="$B -ex \"break _GLOBAL__sub_I_SEventConfig.cc\" " ;
    : SEventConfig statics are initialized from envvars by the above compiler generated function
    : setting the breakpoint allows to workout prior to when the SEventConfig OPTICKS envvars must be defined
    : inorder that they may take effect in the process of loading libSysRap.so

    [ -n "$CATCH_THROW" ] && X="-ex \"catch throw\""
    [ -z "$NORUN" ] && X="$X -ex r"

    local runline="gdb $H $B $T $X --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}





