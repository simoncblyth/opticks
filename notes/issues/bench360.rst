bench360 : equirectangular 360 degree view from scintillator with all PMTs 
====================================================================================

::

    [blyth@localhost issues]$ t geocache-bench360
    geocache-bench360 is a function
    geocache-bench360 () 
    { 
        geocache-rtxcheck $FUNCNAME $*
    }
    [blyth@localhost issues]$ t geocache-bench360-
    geocache-bench360- is a function
    geocache-bench360- () 
    { 
        type $FUNCNAME;
        UseOptiX $*;
        local factor=2;
        local cameratype=2;
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        $dbg OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig "steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25" --size $(geocache-size $factor) --enabledmergedmesh 1,2,3,4,5 --cameratype $cameratype --embedded $*
    }
    [blyth@localhost issues]$ t geocache-rtxcheck
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
    [blyth@localhost issues]$ 





all mm excluding global
--------------------------

triangulated with "all" PMTs visible : RTX and geometrytriangles really shines, get to x7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    geocache-;geocache-bench360


    [blyth@localhost issues]$ bench.py /tmp/blyth/location/results/geocache-bench360
    Namespace(base='/tmp/blyth/location/results/geocache-bench360', exclude=None, include=None, metric='launchAVG', other='prelaunch000')
    /tmp/blyth/location/results/geocache-bench360
     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 5120,2880,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 1 --rtx 2 --runfolder geocache-bench360 --runstamp 1558265025 --runlabel R2_TITAN_RTX
                    20190519_192345  launchAVG      rfast      rslow      prelaunch000 
                       R2_TITAN_RTX      0.020      1.000      0.143           2.109 
                       R1_TITAN_RTX      0.069      3.414      0.487           2.859 
           R0_TITAN_V_AND_TITAN_RTX      0.078      3.861      0.550           2.537 
                         R2_TITAN_V      0.093      4.598      0.655           2.379 
                         R1_TITAN_V      0.108      5.361      0.764           2.564 
                       R0_TITAN_RTX      0.131      6.469      0.922           1.910 
                         R0_TITAN_V      0.142      7.016      1.000           1.758 


* double up the resolution, 4 times the pixels : the pattern stays the same : R2 (RTX ON with GeometryTriangles) gives x6.5 

::

    geocache-;geocache-bench360    


     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 1 --rtx 2 --runfolder geocache-bench360 --runstamp 1558266558 --runlabel R2_TITAN_RTX
                    20190519_194918  launchAVG      rfast      rslow      prelaunch000 
                       R2_TITAN_RTX      0.067      1.000      0.153           1.941 
                       R1_TITAN_RTX      0.161      2.390      0.366           1.702 
           R0_TITAN_V_AND_TITAN_RTX      0.221      3.286      0.503           2.232 
                         R2_TITAN_V      0.301      4.479      0.685           1.879 
                         R1_TITAN_V      0.334      4.967      0.760           1.227 
                       R0_TITAN_RTX      0.403      5.988      0.916           1.394 
                         R0_TITAN_V      0.440      6.536      1.000           1.380 



analytic
~~~~~~~~~~~~~~~~

::

    geocache-;geocache-bench360 --xanalytic


     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 5120,2880,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench360 --runstamp 1558265453 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic
                    20190519_193053  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.119      1.000      0.208          23.791 
                       R0_TITAN_RTX      0.204      1.711      0.356          13.766 
                         R0_TITAN_V      0.236      1.976      0.412          10.728 
                       R1_TITAN_RTX      0.438      3.668      0.764           3.503 
                         R1_TITAN_V      0.573      4.801      1.000           3.167 
    [blyth@localhost issues]$ 



::

     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench360 --runstamp 1558266955 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic
                    20190519_195555  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.375      1.000      0.230          11.814 
                       R0_TITAN_RTX      0.612      1.635      0.377           6.211 
                         R0_TITAN_V      0.750      2.004      0.462           6.010 
                       R1_TITAN_RTX      1.353      3.612      0.832           1.153 
                         R1_TITAN_V      1.625      4.339      1.000           1.027 



