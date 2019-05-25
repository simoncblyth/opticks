bench360 : equirectangular 360 degree view from scintillator with all PMTs 
====================================================================================


Making a 360 degree view : attempting to make a raytrace benchmark relevant to simulation
-------------------------------------------------------------------------------------------

* https://stackoverflow.com/questions/44082298/how-to-make-360-video-output-in-opengl

A 360-surround view from the center of the scintillator would make a 
good benchmark : when switch off the globals anyhow.

Although easy in a ray tracer doing it in OpenGL starting from vertices looks
to be far too difficult for the purpose of a benchmark : is equivalent to 
making 360 degree video by stitching together different views into texture.

See env- equirect- for some reading around about this


Ray tracing part is easy
---------------------------

::

    [blyth@localhost SDK-precompiled-samples]$ LD_LIBRARY_PATH=. ./optixTutorial -T 11
    [blyth@localhost SDK-precompiled-samples]$ pwd
    /home/blyth/local/opticks/externals/OptiX/SDK-precompiled-samples

::

     46 //
     47 // (NEW)
     48 // Environment map camera
     49 //
     50 rtDeclareVariable(float3,        eye, , );
     51 rtDeclareVariable(float3,        U, , );
     52 rtDeclareVariable(float3,        V, , );
     53 rtDeclareVariable(float3,        W, , );
     54 rtDeclareVariable(float3,        bad_color, , );
     55 rtBuffer<uchar4, 2>              output_buffer;
     56 RT_PROGRAM void env_camera()
     57 {
     58   size_t2 screen = output_buffer.size();
     59 
     60   float2 d = make_float2(launch_index) / make_float2(screen) * make_float2(2.0f * M_PIf , M_PIf) + make_float2(M_PIf, 0);
     //    map pixels onto phi,theta 

     61   float3 angle = make_float3(cos(d.x) * sin(d.y), -cos(d.y), sin(d.x) * sin(d.y));
     //                              cos(phi)*sin(theta), -cos(theta), sin(phi)*sin(theta)  
     

     62   float3 ray_origin = eye;
     63   float3 ray_direction = normalize(angle.x*normalize(U) + angle.y*normalize(V) + angle.z*normalize(W));
     64 
     65   optix::Ray ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon);
     66 
     67   PerRayData_radiance prd;
     68   prd.importance = 1.f;
     69   prd.depth = 0;
     70 
     71   rtTrace(top_object, ray, prd);
     72 
     73   output_buffer[launch_index] = make_color( prd.result );
     74 }
     75 













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


20190519 : analytic initial findings show RTX mode is hindering by factor ~2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Increase size to *geocache-size 4*::

     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench360 --runstamp 1558266955 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic
                    20190519_195555  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.375      1.000      0.230          11.814 
                       R0_TITAN_RTX      0.612      1.635      0.377           6.211 
                         R0_TITAN_V      0.750      2.004      0.462           6.010 
                       R1_TITAN_RTX      1.353      3.612      0.832           1.153 
                         R1_TITAN_V      1.625      4.339      1.000           1.027 



20190525 : Analytic after WITH_TORUS and hemi-ellipsoid fixe : RTX now helping by factor 3.6x with TITAN RTX, 1.25 with TITAN V 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RTX mode gives 20% improvement with TITAN V and factor of 3 with TITAN RTX:: 

    bench.py --name geocache-bench360 --include xanalytic --include 10240,5760,1
    [blyth@localhost geocache-bench360]$ bench.py --name geocache-bench360 --include xanalytic --include 10240,5760,1
    bench.py --name geocache-bench360 --include xanalytic --include 10240,5760,1
    Namespace(digest=None, exclude=None, include=[['xanalytic'], ['10240,5760,1']], metric='launchAVG', name='geocache-bench360', other='prelaunch000', resultsdir='$TMP/results', since=None)

    ---  GROUPCOMMAND : -  GEOFUNC : - 
     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench360 --runstamp 1558266955 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.699463ea0065185a7ffaf10d4935fc61
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1
                    20190519_195555  launchAVG      rfast      rslow      prelaunch000 
            R0_TITAN_V_AND_TITAN_RTX      0.375      1.000      0.230          11.814    : /tmp/blyth/location/results/geocache-bench360/R0_TITAN_V_AND_TITAN_RTX/20190519_195555  
                        R0_TITAN_RTX      0.612      1.635      0.377           6.211    : /tmp/blyth/location/results/geocache-bench360/R0_TITAN_RTX/20190519_195555  
                          R0_TITAN_V      0.750      2.004      0.462           6.010    : /tmp/blyth/location/results/geocache-bench360/R0_TITAN_V/20190519_195555  
                        R1_TITAN_RTX      1.353      3.612      0.832           1.153    : /tmp/blyth/location/results/geocache-bench360/R1_TITAN_RTX/20190519_195555  
                          R1_TITAN_V      1.625      4.339      1.000           1.027    : /tmp/blyth/location/results/geocache-bench360/R1_TITAN_V/20190519_195555  
                        R0/1_TITAN_V      0.462 
                      R0/1_TITAN_RTX      0.453 
                        R1/0_TITAN_V      2.166 
                      R1/0_TITAN_RTX      2.209 

    ---  GROUPCOMMAND : geocache-bench360 --xanalytic  GEOFUNC : geocache-j1808-v4 
     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench360 --runstamp 1558784420 --runlabel R1_TITAN_RTX --xanalytic
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
                    20190525_194020  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.215      1.000      0.280           2.028    : /tmp/blyth/location/results/geocache-bench360/R1_TITAN_RTX/20190525_194020  
            R0_TITAN_V_AND_TITAN_RTX      0.390      1.814      0.507           2.879    : /tmp/blyth/location/results/geocache-bench360/R0_TITAN_V_AND_TITAN_RTX/20190525_194020  
                          R1_TITAN_V      0.519      2.413      0.675           2.119    : /tmp/blyth/location/results/geocache-bench360/R1_TITAN_V/20190525_194020  
                          R0_TITAN_V      0.656      3.051      0.853           1.650    : /tmp/blyth/location/results/geocache-bench360/R0_TITAN_V/20190525_194020  
                        R0_TITAN_RTX      0.769      3.577      1.000           1.671    : /tmp/blyth/location/results/geocache-bench360/R0_TITAN_RTX/20190525_194020  
                        R0/1_TITAN_V      1.264 
                      R0/1_TITAN_RTX      3.577    <<<<< HELPING BY FACTOR 3.6 WITH ANALYTIC GEOMETRY 
                        R1/0_TITAN_V      0.791 
                      R1/0_TITAN_RTX      0.280 
    ()
    bench.py --name geocache-bench360 --include xanalytic --include 10240,5760,1




