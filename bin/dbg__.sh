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

    [ -n "$CATCH_THROW" ] && X="-ex \"catch throw\""

    if [ -z "$BP" ]; then
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        : split BP on comma delimiter preserving spaces within fields
        readarray -td, bps <<<"$BP"
        for bp in "${bps[@]}"
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T $X --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}





