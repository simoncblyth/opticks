#!/usr/bin/env bash
usage(){ cat << EOU
SProf.sh
========

If LOGROOT is not defined the invoking directory is used.


prol
    list all SProf.txt found below LOGROOT
prof
    find all SProf.txt below LOGROOT and dump time tables using SProf.py
logp
    find all OJ*.log and J*.log below LOGROOT and dump times using logparse.py
grep
    grep @ lines in all logs below LOGROOT


Usage example::

    cd /data1/blyth/tmp/j/zhenning_double_muon/detsim
    T0=2025-01-01               ~/o/bin/SProf.sh logp
    T0=2025-01-01 T1=2026-01-01 ~/o/bin/SProf.sh logp



Workflow:

0. cd to a logroot directory or set LOGROOT envvar
1. use prol and logl to narrow the T0 -> T1 time strings to just the logs of interest

   T0="2025-12-04 12:00" T1="2025-12-04 16:00" ~/o/bin/SProf.sh prol

2. use LOGROOT T0 T1 to run logp prof examining the logs and profile files


The functionality of this script formerly part of ~/j/zhenning_double_muon/detsim.sh




Examples::

    A[blyth@localhost detsim]$ pwd   # LOGROOT defaults to invoking directory
    /data1/blyth/tmp/j/zhenning_double_muon/detsim

    A[blyth@localhost detsim]$ T0="2025-12-04 16:00" T1="2025-12-04 18:00" ~/o/bin/SProf.sh proi
    Identity                                             Start(s)  Duration(s)     Sub PREL→POST(s)     Sub POST→DOWN(s)    Sub TAIL→RESET(s)    VM(MB)  RSS(MB)  Comment
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    hit/A000_QSim                                         106.466   215.560469            22.996241             1.949322           190.446287   32168.83  7971.96  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    hitlite/A000_QSim                                     106.462   170.226838            23.108319             0.484299           146.471936   32499.61  5361.86  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    hitlitemerged/A000_QSim                               107.263    23.835564            23.097717             0.181023             0.404235   87284.29  4988.84  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=6409029
    hitmerged/A000_QSim                                   106.770    30.400825            22.988498             0.543559             6.713401   88292.85  6634.07  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=6409029  # slice=1,max_slot_M=150







EOU
}

LOGROOT=${LOGROOT:-$PWD}
cd $(dirname $(realpath $BASH_SOURCE))


defarg=info
allarg=info_prof0_prof_grep_logp
arg=${1:-$defarg}

vvp(){ local v ; for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done ; }

vv=""
vv="$vv BASH_SOURCE defarg arg allarg LOGROOT PWD"



if [[ "$arg" =~ info ]]; then
    vvp
fi


if [[ "$arg" =~ ^(prof|proi|prof0|prol|logp|logl|grep)$ ]]; then

    # T0 and T1 control time window within which to look for SProf.txt and logs
    t0="120 minutes ago"
    t1="now"
    
    T0=${T0:-$t0}   # start time
    T1=${T1:-$t1}   # end time

    mt0=$(date "+%Y-%m-%d %H:%M:%S" -d "$T0")
    mt1=$(date "+%Y-%m-%d %H:%M:%S" -d "$T1")

    tt="t0 T0 mt0 t1 T1 mt1"
    if [ -n "$DUMP" ]; then
        for t in $tt ; do printf "%30s : %s\n" "$t" "${!t}" ; done
    fi

    if [ "${arg}" == "prof0" ]; then
        : unsorted list of SProf.txt modified between T0 and T1 and invoke SProf.py on them which parses SProf.txt
        find "$LOGROOT" -name SProf.txt -newermt "$mt0" ! -newermt "$mt1" -exec bash -c 'echo "→ {}" && ./SProf.py "{}"' \;
    fi  

    if [ "${arg}" == "prof" ]; then

       : like prof0 but order SProf.txt by modtime
       find "$LOGROOT" -name SProf.txt -newermt "$mt0" ! -newermt "$mt1" -printf '%T@ %p\0' | \
             sort -z -n | \
             cut -z -d ' ' -f2- | \
             xargs -0 -n1 bash -c 'echo "→ $1" && ./SProf.py "$1"' bash
    fi  


    if [ "${arg}" == "proi" ]; then
        count=0
        # Use a while loop with null-termination to handle file names safely
        while IFS= read -r -d '' line; do
            # Extract the path (removing the timestamp prefix)
            file_path="${line#* }"
            
            #echo "→ $file_path"
            
            # Pass the index as an environment variable
            SPROF_INDEX=$count ./SProf.py "$file_path"
            
            ((count++))
        done < <(find "$LOGROOT" -name SProf.txt -newermt "$mt0" ! -newermt "$mt1" -printf '%T@ %p\0' | sort -z -n)
    fi



    if [ "${arg}" == "prol" ]; then
       find "$LOGROOT" -name SProf.txt -newermt "$mt0" ! -newermt "$mt1" -printf '%T@ %p\0' | \
             sort -z -n | \
             cut -z -d ' ' -f2- | \
             xargs -0 -n1 bash -c 'ls -l "$1" ' bash
    fi  


    if [ "${arg}" == "logl" ]; then
       find "$LOGROOT" -type f \( -name 'OJ*.log' -o -name 'J*.log' \) -newermt "$mt0" ! -newermt "$mt1" -exec bash -c "echo {} " \;
    fi

    if [ "${arg}" == "logp" ]; then
       find "$LOGROOT" -type f \( -name 'OJ*.log' -o -name 'J*.log' \) -newermt "$mt0" ! -newermt "$mt1" -exec bash -c "echo {} " \;
       find "$LOGROOT" -type f \( -name 'OJ*.log' -o -name 'J*.log' \) -newermt "$mt0" ! -newermt "$mt1" -exec bash -c "echo {} && ./logparse.py {}" \;
    fi  
    if [ "${arg}" == "grep" ]; then
       find "$LOGROOT" -type f \( -name 'OJ*.log' -o -name 'J*.log' \) -newermt "$mt0" ! -newermt "$mt1" -exec bash -c "echo {} " \;
       find "$LOGROOT" -type f \( -name 'OJ*.log' -o -name 'J*.log' \) -newermt "$mt0" ! -newermt "$mt1" -exec bash -c "grep \" @ \" {} " \;
    fi  
fi



