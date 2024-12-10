dbg__ () 
{ 
    case $(uname) in 
        Darwin)
            lldb__ $*
        ;;
        Linux)
            gdb__ $*
        ;;
    esac
}

lldb__ () 
{ 
    : ~opticks/sysrap/tests/dbg__.sh
    : macOS only - this function requires LLDB envvar to provide the path;
    : to the lldb application within the appropriate Xcode.app resources eg;
    local BINARY=$1;
    shift;
    local ARGS=$*;
    local H="$HEAD";
    local B;
    local bp;
    echo HEAD $HEAD;
    echo TAIL $TAIL;
    if [ -z "$BP" ]; then
        B="";
    else
        B="";
        for bp in $BP;
        do
            B="$B -o \"b $bp\" ";
        done;
        B="$B -o b";
        [ -n "$BX" ] && B="$B -o \"$BX\" ";
    fi;
    local T="$TAIL";
    local def_lldb=/Applications/Xcode/Xcode.app/Contents/Developer/usr/bin/lldb;
    local runline="${LLDB:-$def_lldb} -f ${BINARY} $H $B $T -- ${ARGS}";
    echo $runline;
    eval $runline
}


gdb__ () 
{ 
    : ~opticks/sysrap/tests/dbg__.sh
    :  prepares and invokes gdb - sets up breakpoints based on BP envvar containing space delimited symbols;
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




