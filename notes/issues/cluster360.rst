cluster360 : compare performance with 8 GPUs on cluster node with 2 GPUs on workstation
============================================================================================

Overview
-----------

Context :doc:`bench360`

* reasonable scaling out to 4 GPUs, but not beyond
* :doc:`multi-gpu-optix` suggests some possibilities to use gpu local buffers 


Measurements
------------

Workstation::

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

Cluster::

    [blyth@lxslc701 out]$ bench.py --name geocache-bench360
    bench.py --name geocache-bench360
    Namespace(digest=None, exclude=None, include=None, metric='launchAVG', name='geocache-bench360', other='prelaunch000', resultsdir='$OPTICKS_RESULTS_PREFIX/results', since=None)

    ---  GROUPCOMMAND : geocache-bench360 --xanalytic --nosaveppm  GEOFUNC : - 
     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 0,1,2,3 --rtx 0 --runfolder geocache-bench360 --runstamp 1558848159 --xanalytic --nosaveppm
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
                    20190526_132239  launchAVG      rfast      rslow      prelaunch000 
                      R0_cvd_0,1,2,3      0.156      1.000      0.218           2.828    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1,2,3/20190526_132239  
                        R0_cvd_0,1,2      0.204      1.311      0.285           2.404    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1,2/20190526_132239  
                          R0_cvd_0,1      0.299      1.916      0.417           1.876    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1/20190526_132239  
                R0_cvd_0,1,2,3,4,5,6      0.357      2.294      0.499           4.827    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1,2,3,4,5,6/20190526_132239  
                  R0_cvd_0,1,2,3,4,5      0.371      2.382      0.519           3.977    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1,2,3,4,5/20190526_132239  
              R0_cvd_0,1,2,3,4,5,6,7      0.452      2.903      0.632           5.165    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1,2,3,4,5,6,7/20190526_132239  
           R1_Tesla_V100-SXM2-32GB-1      0.457      2.930      0.638           0.879    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_Tesla_V100-SXM2-32GB-1/20190526_132239  
           R1_Tesla_V100-SXM2-32GB-4      0.457      2.932      0.638           0.922    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_Tesla_V100-SXM2-32GB-4/20190526_132239  
           R1_Tesla_V100-SXM2-32GB-0      0.457      2.933      0.639           1.922    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_Tesla_V100-SXM2-32GB-0/20190526_132239  
           R1_Tesla_V100-SXM2-32GB-2      0.457      2.933      0.639           0.874    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_Tesla_V100-SXM2-32GB-2/20190526_132239  
           R1_Tesla_V100-SXM2-32GB-7      0.457      2.934      0.639           0.918    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_Tesla_V100-SXM2-32GB-7/20190526_132239  
           R1_Tesla_V100-SXM2-32GB-3      0.457      2.936      0.639           0.857    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_Tesla_V100-SXM2-32GB-3/20190526_132239  
           R1_Tesla_V100-SXM2-32GB-6      0.458      2.936      0.639           0.916    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_Tesla_V100-SXM2-32GB-6/20190526_132239  
           R1_Tesla_V100-SXM2-32GB-5      0.458      2.940      0.640           0.926    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_Tesla_V100-SXM2-32GB-5/20190526_132239  
                            R0_cvd_0      0.570      3.661      0.797           1.403    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0/20190526_132239  
           R0_Tesla_V100-SXM2-32GB-0      0.571      3.664      0.798           1.519    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_Tesla_V100-SXM2-32GB-0/20190526_132239  
           R0_Tesla_V100-SXM2-32GB-7      0.571      3.665      0.798           1.455    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_Tesla_V100-SXM2-32GB-7/20190526_132239  
           R0_Tesla_V100-SXM2-32GB-6      0.571      3.665      0.798           1.425    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_Tesla_V100-SXM2-32GB-6/20190526_132239  
           R0_Tesla_V100-SXM2-32GB-2      0.571      3.667      0.799           1.427    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_Tesla_V100-SXM2-32GB-2/20190526_132239  
           R0_Tesla_V100-SXM2-32GB-4      0.571      3.667      0.799           1.456    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_Tesla_V100-SXM2-32GB-4/20190526_132239  
           R0_Tesla_V100-SXM2-32GB-3      0.572      3.669      0.799           1.402    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_Tesla_V100-SXM2-32GB-3/20190526_132239  
           R0_Tesla_V100-SXM2-32GB-5      0.573      3.675      0.800           1.439    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_Tesla_V100-SXM2-32GB-5/20190526_132239  
           R0_Tesla_V100-SXM2-32GB-1      0.573      3.678      0.801           1.436    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_Tesla_V100-SXM2-32GB-1/20190526_132239  
                    R0_cvd_0,1,2,3,4      0.716      4.592      1.000           3.364    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1,2,3,4/20190526_132239  



* RTX mode helping about the same with V100 too, 0.571/0.457 = 1.2494
* R0_TITAN_V/V100  0.656/0.571  1.1488            
* R1_TITAN_V/V100  0.519/0.457  1.1356
* V100 a little faster than TITAN_V
* good uniformity between the V100

* reasonable scaling from 1 -> 2 -> 3 -> 4  but not beyond 4::

                    0.570/0.156 = 3.653  0.570/0.570 = 1.000 
                    0.299/0.156 = 1.916  0.299/0.570 = 0.524
                    0.204/0.156 = 1.307  0.204/0.570 = 0.357
                    0.156/0.156 = 1.000  0.156/0.570 = 0.273 

::

    ---  GROUPCOMMAND : geocache-bench360 --xanalytic --nosaveppm  GEOFUNC : - 
     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 0,1,2,3 --rtx 1 --runfolder geocache-bench360 --runstamp 1558852688 --xanalytic --nosaveppm
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
                    20190526_143808  launchAVG      rfast      rslow      prelaunch000 
                      R1_cvd_0,1,2,3      0.121      1.000      0.135           3.978    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_0,1,2,3/20190526_143808  
                      R1_cvd_4,5,6,7      0.123      1.014      0.137           3.467    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_4,5,6,7/20190526_143808  

                      R0_cvd_0,1,2,3      0.152      1.249      0.169           2.861    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1,2,3/20190526_143808  
                      R0_cvd_4,5,6,7      0.152      1.250      0.169           2.948    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_4,5,6,7/20190526_143808  

                          R1_cvd_0,1      0.234      1.931      0.261           2.190    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_0,1/20190526_143808  
                          R1_cvd_2,3      0.234      1.931      0.261           1.579    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_2,3/20190526_143808  
                          R1_cvd_4,5      0.237      1.954      0.264           1.638    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_4,5/20190526_143808  
                          R1_cvd_6,7      0.239      1.966      0.266           1.596    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_6,7/20190526_143808  

                          R0_cvd_0,1      0.295      2.433      0.329           1.877    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1/20190526_143808  
                          R0_cvd_2,3      0.296      2.438      0.330           1.895    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_2,3/20190526_143808  
                          R0_cvd_4,5      0.300      2.473      0.334           1.924    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_4,5/20190526_143808  
                          R0_cvd_6,7      0.301      2.480      0.335           1.970    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_6,7/20190526_143808  

              R0_cvd_0,1,2,3,4,5,6,7      0.452      3.727      0.504           5.015    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0,1,2,3,4,5,6,7/20190526_143808  

                            R1_cvd_0      0.457      3.767      0.509           0.865    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_0/20190526_143808  
                            R1_cvd_4      0.458      3.770      0.510           0.924    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_4/20190526_143808  

                            R0_cvd_0      0.570      4.698      0.635           1.397    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_0/20190526_143808  
                            R0_cvd_4      0.572      4.712      0.637           1.460    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R0_cvd_4/20190526_143808  

              R1_cvd_0,1,2,3,4,5,6,7      0.897      7.394      1.000           9.006    : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-bench360/R1_cvd_0,1,2,3,4,5,6,7/20190526_143808  
    ()
    bench.py --name geocache-bench360


* RTX mode does help with multiple V100 too 








