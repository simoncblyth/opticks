rtxmode-performance-jumps-by-factor-3-or-4-after-flipping-with-torus-switch-off
================================================================================

context
-----------

* :doc:`review-analytic-geometry`


correlation is not causation
-------------------------------

A sudden improvement leap in RTX performance occurred close to
doing a full cleaninstall, however subsequent examination 
of commits revealed the real cause to have been the flipping 
of the the WITH_TORUS define OFF in intersect_analytic.

* https://bitbucket.org/simoncblyth/opticks/commits/a966c80ceaf1593df02192aac50d31b01fcf4b47#chg-optixrap/cu/intersect_analytic.cu
* above commit 22 May 14:01 "fixes for Opticks to work with older OptiX_511" flipped the WITH_TORUS define OFF


CONCLUSIVE EVIDENCE : SWITCING WITH_TORUS OFF CAUSED THE PERFORMANCE JUMP
---------------------------------------------------------------------------

Only difference between the below is flipping the WITH_TORUS switch in intersect_analytic.cu
to the ON position in the second.  It results in RTX times increasing by a factor of almost 4. 

::

    blyth@localhost ~]$ bench.py --include xanalytic --digest 52e --since May23
    Namespace(digest='52e', exclude=None, include='xanalytic', metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='$TMP/results', since='May23')
    since : 2019-05-23 00:00:00 

    ---
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558577742 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/52e273e4ad5423fe2fc8aa44bbf055ec/1
                    20190523_101542  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.040      1.000      0.340           0.490    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190523_101542  
                          R1_TITAN_V      0.043      1.077      0.366           0.443    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190523_101542  
            R0_TITAN_V_AND_TITAN_RTX      0.064      1.596      0.542          11.661    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190523_101542  
                          R0_TITAN_V      0.102      2.516      0.855           5.930    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190523_101542  
                        R0_TITAN_RTX      0.119      2.942      1.000           5.956    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190523_101542  

    ---
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558579793 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/52e273e4ad5423fe2fc8aa44bbf055ec/1
                    20190523_104953  launchAVG      rfast      rslow      prelaunch000 
            R0_TITAN_V_AND_TITAN_RTX      0.055      1.000      0.281          24.133    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190523_104953  
                        R0_TITAN_RTX      0.099      1.807      0.508          13.804    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190523_104953  
                          R0_TITAN_V      0.100      1.818      0.511          11.218    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190523_104953  
                          R1_TITAN_V      0.170      3.085      0.867           2.655    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190523_104953  
                        R1_TITAN_RTX      0.196      3.559      1.000           2.818    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190523_104953  

    * switch WITH_TORUS back ON : RTX times increase by almost factor 5  

    ---
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558581291 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/52e273e4ad5423fe2fc8aa44bbf055ec/1
                    20190523_111451  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.037      1.000      0.339           1.402    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190523_111451  
                          R1_TITAN_V      0.043      1.167      0.396           1.371    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190523_111451  
            R0_TITAN_V_AND_TITAN_RTX      0.059      1.601      0.543           2.378    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190523_111451  
                          R0_TITAN_V      0.090      2.474      0.839           1.517    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190523_111451  
                        R0_TITAN_RTX      0.108      2.950      1.000           1.448    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190523_111451  
    Namespace(digest='52e', exclude=None, include='xanalytic', metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='$TMP/results', since='May23')


    * switch OFF WITH_TORUS WITH_PARTLIST WITH_SOLVE : aim for no doubles used in the PTX 




counting lines of PTX with .f64 before/after WITH_TORUS
------------------------------------------------------------ 

* WITH_TORUS brings in a boatload of doubles 


::

    [blyth@localhost ~]$ grep f64 /tmp/OptiXRap_generated_intersect_analytic.cu.ptx | wc -l
    1598

    [blyth@localhost ~]$ grep f64 /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic.cu.ptx | wc -l
    337



Hmm but still 337 lines of PTX with f64 


* https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

optix-pdf::

    Where possible use floats instead of doubles.
    This also extends to the use of literals and math functions. For example, use 0.5f instead
    of 0.5 and sinf instead of sin to prevent automatic type promotion. To check for
    automatic type promotion, search the PTX files for the .f64 instruction modifier.


