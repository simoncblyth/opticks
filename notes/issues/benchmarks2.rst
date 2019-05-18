benchmarks2
===============


Unmoving view from middle
-------------------------------

::

    [blyth@localhost opticks]$ t geocache-bench2
    geocache-bench2 is a function
    geocache-bench2 () 
    { 
        geocache-rtxcheck $FUNCNAME $*
    }
    [blyth@localhost opticks]$ t geocache-bench2-
    geocache-bench2- is a function
    geocache-bench2- () 
    { 
        type $FUNCNAME;
        UseOptiX;
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        $dbg OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 1,0,0 --up 0,0,1 --snapconfig "steps=5,eyestartz=0,eyestopz=0" --size 5120,2880,1 --embedded $*
    }
    [blyth@localhost opticks]$ t geocache-rtxcheck
    geocache-rtxcheck is a function
    geocache-rtxcheck () 
    { 
        local name=${1:-geocache-bench};
        shift;
        local stamp=$(date +%s);
        $name- --cvd 1 --rtx 0 --runfolder $name --runstamp $stamp --runlabel "R0_TITAN_RTX" $*;
        $name- --cvd 1 --rtx 1 --runfolder $name --runstamp $stamp --runlabel "R1_TITAN_RTX" $*;
        $name- --cvd 1 --rtx 2 --runfolder $name --runstamp $stamp --runlabel "R2_TITAN_RTX" $*;
        $name- --cvd 0 --rtx 0 --runfolder $name --runstamp $stamp --runlabel "R0_TITAN_V" $*;
        $name- --cvd 0 --rtx 1 --runfolder $name --runstamp $stamp --runlabel "R1_TITAN_V" $*;
        $name- --cvd 0 --rtx 2 --runfolder $name --runstamp $stamp --runlabel "R2_TITAN_V" $*;
        $name- --cvd 0,1 --rtx 0 --runfolder $name --runstamp $stamp --runlabel "R0_TITAN_V_AND_TITAN_RTX" $*;
        bench.py $TMP/results/$name
    }
    [blyth@localhost opticks]$ 



mm2
----

* RTX factor of 2 slower

::

  
    geocache-;geocache-bench2 --xanalytic --enabledmergedmesh 2

    /tmp/blyth/location/results/geocache-bench2
     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 1,0,0 --up 0,0,1 --snapconfig steps=5,eyestartz=0,eyestopz=0 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench2 --runstamp 1558187600 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190518_215320     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.064      1.000      0.256 
                       R0_TITAN_RTX      0.112      1.732      0.444 
                         R0_TITAN_V      0.120      1.854      0.475 
                         R1_TITAN_V      0.225      3.495      0.896 
                       R1_TITAN_RTX      0.252      3.901      1.000 



