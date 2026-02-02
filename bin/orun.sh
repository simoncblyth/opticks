#!/bin/bash
usage(){ cat << EOU
Usage: orun.sh <script_path> [VAR=VAL ...] [args ...]

Examples::

   cx
   orun.sh ./cxr_min.sh FULLSCREEN=0 MOI=AXIS:56,-54,-21271,3843 EYE=-2,0,-0.5 UP=0,1,0 ELV=^s_EMF,^GZ 

   orun.sh PRIMTAB=1 RADIAL=2 KEY=yellowgreen,grey,wheat ./cxt_min.sh pdb


This Opticks script runner captures the command line and selectively exports environment 
variables while passing other arguments directly to the script. 
The script is identified by ending with ".sh", envvars are identified by containing "=", 
all other args are passed to the script. Note that this does not require any particular
ordering for the script envvars and script args.

The advantage of running scripts thru this wrapper is that the commandline
is captured and promoted into the COMMANDLINE and TITLE envvars for
use by the script or executables that the script runs (for example writing 
the commandline into metadata or including it within plot/image titles). 

The commandline is also appended to ~/.opticks/orun_history 

EOU
}

if [ -z "$1" ]; then usage; exit 1; fi

# Initialize variables
target_script=""
env_vars=()
script_args=()

for arg in "$@"; do
    if [[ "$arg" == *.sh ]]; then
        target_script="$arg"
    elif [[ "$arg" == *"="* ]]; then
        # If it has an '=' and we haven't found the script yet, 
        # or if you want it exported regardless:
        export "$arg"
        env_vars+=("$arg")
    else
        # Everything else is a script argument (like 'pdb')
        script_args+=("$arg")
    fi
done

if [ -z "$target_script" ]; then
    echo "Error: No .sh script identified in arguments."
    exit 1
fi

# Capture the exact execution context
export COMMANDLINE="${env_vars[@]} $target_script ${script_args[@]}"
export TITLE="$COMMANDLINE"


logpath=$HOME/.opticks/orun_history
mkdir -p $(dirname $logpath)
echo "[$(date)] $COMMANDLINE" >> $logpath

exec "$target_script" "${script_args[@]}"



