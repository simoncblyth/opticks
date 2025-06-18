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

    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex \"catch throw\" -ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        : split BP on comma delimiter preserving spaces within fields
        readarray -td, bps <<<"$BP"
        for bp in "${bps[@]}"
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex \"catch throw\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}





