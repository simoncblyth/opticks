benchmarks
==============


Things to try to speedup analytic
---------------------------------------

1. simplifiy geometry tree : DONE, NO SIGNIFICANT CHANGE
2. change accel builders : DONE, NO SIGNIFICANT CHANGE
3. review the analytic buffers 

   * especially the prismBuffer, why is it INPUT_OUTPUT 
   * its very small : just try to get rid of it 


Titan V and Titan RTX : Effect of RTX execution mode
----------------------------------------------------------------

Comparing raytrace performance of Titan V and Titan RTX 
with a modified JUNO geometry with Torus removed
from PMTs and guidetube by changing the input GDML. 
(OptiX 6.0.0 crashes when attempting to use my quartic 
root finding for the Torus.)

My benchmark metric is the average of five very high resolution 
5120x2880 ~15M pixels raytrace launch times near the JUNO 
chimney with a large number of PMTs in view.

I use three RTX mode variations:

   R0
       RTX off : ordinary software BVH traversal and intersection
   R1
       RTX on : only BVH traversal using RT Cores, intersection in software
   R2
       RTX on + intersection handled with RT Cores using GeometryTriangles (new in OptiX 6) 







Note RTX mode one has much faster prelaunch ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* any possibility of cache effects between runs ? dont think so : as delete /var/tmp/OptiXCache between runs
  BUT could be some other cache, TODO: look for dependency on order of runs


::

    OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558185347 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2 --instancemodulo 2:10
                    20190518_211547  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.022      1.000      0.393          24.104 
                         R0_TITAN_V      0.040      1.799      0.707          10.931 
                       R0_TITAN_RTX      0.041      1.858      0.730          13.423 
                         R1_TITAN_V      0.051      2.294      0.901           0.365 
                         R2_TITAN_V      0.051      2.296      0.902           0.172 
                       R2_TITAN_RTX      0.055      2.487      0.978           0.151 
                       R1_TITAN_RTX      0.057      2.544      1.000           0.369 
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558185811 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2 --instancemodulo 2:5
                    20190518_212331  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.035      1.000      0.306          23.992 
                       R0_TITAN_RTX      0.061      1.767      0.541          13.343 
                         R0_TITAN_V      0.063      1.832      0.560          10.666 
                         R2_TITAN_V      0.102      2.938      0.899           0.195 
                         R1_TITAN_V      0.102      2.940      0.899           0.389 
                       R1_TITAN_RTX      0.110      3.190      0.976           0.422 
                       R2_TITAN_RTX      0.113      3.269      1.000           0.201 
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558186475 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2 --instancemodulo 2:2
                    20190518_213435  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.057      1.000      0.251          23.973 
                       R0_TITAN_RTX      0.092      1.624      0.407          13.276 
                         R0_TITAN_V      0.105      1.851      0.464          10.858 
                         R1_TITAN_V      0.210      3.700      0.928           0.496 
                       R1_TITAN_RTX      0.227      3.987      1.000           0.529 
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558187531 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190518_215211  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.074      1.000      0.220          12.375 
                       R0_TITAN_RTX      0.120      1.611      0.354           6.397 
                         R0_TITAN_V      0.136      1.834      0.403           6.479 
                         R1_TITAN_V      0.314      4.229      0.928           0.439 
                       R1_TITAN_RTX      0.338      4.555      1.000           0.645 
    [blyth@localhost opticks]$ 









Disabling ANYHIT for the ray and geometry and geometrygroup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 2 --runfolder geocache-bench --runstamp 1558081500 --runlabel R2_TITAN_RTX
                    20190517_162500     metric      rfast      rslow 
                       R2_TITAN_RTX      0.023      1.000      0.164 
                       R1_TITAN_RTX      0.071      3.063      0.501 
           R0_TITAN_V_AND_TITAN_RTX      0.077      3.319      0.543 
                         R2_TITAN_V      0.091      3.910      0.640 
                       R0_TITAN_RTX      0.102      4.369      0.715 
                         R1_TITAN_V      0.127      5.461      0.894 
                         R0_TITAN_V      0.142      6.109      1.000 

Disabling ANYHIT for the ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 2 --runfolder geocache-bench --runstamp 1558077851 --runlabel R2_TITAN_RTX
                    20190517_152411     metric      rfast      rslow 
                       R2_TITAN_RTX      0.025      1.000      0.175 
                       R1_TITAN_RTX      0.072      2.857      0.499 
           R0_TITAN_V_AND_TITAN_RTX      0.079      3.159      0.552 
                         R2_TITAN_V      0.091      3.608      0.630 
                       R0_TITAN_RTX      0.103      4.083      0.713 
                         R1_TITAN_V      0.126      5.013      0.876 
                         R0_TITAN_V      0.144      5.726      1.000 


Reproducibilioty check of triangulated, few weeks later
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    blyth@localhost opticks]$ bench.py $TMP/results/geocache-bench
    Namespace(base='/tmp/blyth/location/results/geocache-bench', exclude=None, include=None)
    /tmp/blyth/location/results/geocache-bench
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 2 --runfolder geocache-bench --runstamp 1558074625 --runlabel R2_TITAN_RTX
                    20190517_143025     metric      rfast      rslow 
                       R2_TITAN_RTX      0.031      1.000      0.219 
                       R1_TITAN_RTX      0.060      1.909      0.419 
           R0_TITAN_V_AND_TITAN_RTX      0.081      2.563      0.562 
                       R0_TITAN_RTX      0.101      3.220      0.707 
                         R2_TITAN_V      0.118      3.760      0.825 
                         R1_TITAN_V      0.130      4.139      0.908 
                         R0_TITAN_V      0.143      4.557      1.000 


Times for triangulated geometry in seconds:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

       .        20190424_203832     metric      rfast      rslow 

                   R2_TITAN_RTX      0.037      1.000      0.250 
                   R1_TITAN_RTX      0.074      2.018      0.505 
       R0_TITAN_V_AND_TITAN_RTX      0.078      2.129      0.533 
                     R2_TITAN_V      0.100      2.722      0.682 
                   R0_TITAN_RTX      0.103      2.810      0.704 
                     R1_TITAN_V      0.116      3.149      0.789 
                     R0_TITAN_V      0.147      3.993      1.000 

Example commandline::

   OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 \
              --embedded --rtx 2 --runfolder geocache-bench --runstamp 1556109512 --runlabel R2_TITAN_RTX


Observations:

* fractions of a second for 15M pixels bodes well 
* TITAN RTX gains a factor of ~3 from R0 to R2 
* TITAN V doesnt have RT cores, but RTX mode still improves its times




volumes
~~~~~~~~~

===============   =================  ================
mm index            gui label          notes
===============   =================  ================
   0                                   global non-instanced
   1                  in0              small PMT
   2                  in1              large PMT
   3                  in2              some TT plate, that manages to be 130 volumes 
   4                  in3              support stick
   5                  in4              support temple
===============   =================  ================



modulo scaledown the 20k instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

     geocache-;geocache-gui --enabledmergedmesh 2 --instancemodulo 2:10 


combination of the fast ones : --xanalytic --enabledmergedmesh 1,3,4,5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* stays fast, and RTX helps a little

::

     geocache-;geocache-gui --enabledmergedmesh 1,3,4,5                    ## changed name of restrictmesh after generalize to accepting a command delimited list 
     geocache-;geocache-bench --xanalytic --enabledmergedmesh 1,3,4,5      ## changed name of restrictmesh after generalize to accepting a command delimited list 

::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558179690 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 1,3,4,5
                    20190518_194130     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.042      1.000      0.649 
                       R2_TITAN_RTX      0.049      1.145      0.743 
                       R1_TITAN_RTX      0.049      1.149      0.746 
                         R2_TITAN_V      0.051      1.191      0.773 
                         R1_TITAN_V      0.051      1.204      0.781 
                         R0_TITAN_V      0.061      1.447      0.939 
                       R0_TITAN_RTX      0.065      1.541      1.000 



restrict to mm5 : support temple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* really fast : and its quite a deep CSG tree 
* RTX mode helps T-rex and V

::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0 --rtx 2 --runfolder geocache-bench --runstamp 1558093822 --runlabel R2_TITAN_V --restrictmesh 5 --xanalytic
                    20190517_195022     metric      rfast      rslow 
                         R2_TITAN_V      0.003      1.000      0.162 
                         R1_TITAN_V      0.003      1.013      0.165 
                       R1_TITAN_RTX      0.003      1.126      0.183 
                       R2_TITAN_RTX      0.003      1.133      0.184 
           R0_TITAN_V_AND_TITAN_RTX      0.011      3.645      0.592 
                         R0_TITAN_V      0.016      5.566      0.904 
                       R0_TITAN_RTX      0.018      6.155      1.000 


restrict to mm4 : support sticks (just cylinders)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* RTX mode helps alot (with TITAN V too)

::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558093581 --runlabel R1_TITAN_RTX --restrictmesh 4 --xanalytic
                    20190517_194621     metric      rfast      rslow 
                       R1_TITAN_RTX      0.004      1.000      0.162 
                       R2_TITAN_RTX      0.004      1.056      0.171 
                         R1_TITAN_V      0.004      1.071      0.173 
                         R2_TITAN_V      0.004      1.072      0.173 
           R0_TITAN_V_AND_TITAN_RTX      0.013      3.317      0.536 
                         R0_TITAN_V      0.021      5.409      0.875 
                       R0_TITAN_RTX      0.024      6.185      1.000 


restrict to mm3 : TT plates, times very similar to SPMT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* RTX mode gives some speedup on T-rex

::

    OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558092977 --runlabel R0_TITAN_V_AND_TITAN_RTX --restrictmesh 3 --xanalytic
                    20190517_193617     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.018      1.000      0.523 
                       R2_TITAN_RTX      0.022      1.221      0.639 
                       R1_TITAN_RTX      0.022      1.252      0.655 
                         R0_TITAN_V      0.029      1.647      0.862 
                         R2_TITAN_V      0.031      1.727      0.904 
                         R1_TITAN_V      0.031      1.736      0.909 
                       R0_TITAN_RTX      0.034      1.911      1.000 




restrict to mm2 : 20k 20-inch PMT  with 1 in 10 modulo scaledown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* even with only 2k RTX mode not helping for 20-inchers

::

     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2  --instancemodulo 2:10   ## scaledown 1 in 10 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558185347 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2 --instancemodulo 2:10
                    20190518_211547     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.022      1.000      0.393 
                         R0_TITAN_V      0.040      1.799      0.707 
                       R0_TITAN_RTX      0.041      1.858      0.730 
                         R1_TITAN_V      0.051      2.294      0.901 
                         R2_TITAN_V      0.051      2.296      0.902 
                       R2_TITAN_RTX      0.055      2.487      0.978 
                       R1_TITAN_RTX      0.057      2.544      1.000 


* with RTX mode on, looks like the time is scaling with the number of instances of mm2 

::

    OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558185811 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2 --instancemodulo 2:5
                    20190518_212331     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.035      1.000      0.306 
                       R0_TITAN_RTX      0.061      1.767      0.541 
                         R0_TITAN_V      0.063      1.832      0.560 
                         R2_TITAN_V      0.102      2.938      0.899 
                         R1_TITAN_V      0.102      2.940      0.899 
                       R1_TITAN_RTX      0.110      3.190      0.976 
                       R2_TITAN_RTX      0.113      3.269      1.000 


::

     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2  --instancemodulo 2:2   ## scaledown 1 in 2 + skip doing R2 for xanalytic

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558186475 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2 --instancemodulo 2:2
                    20190518_213435     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.057      1.000      0.251 
                       R0_TITAN_RTX      0.092      1.624      0.407 
                         R0_TITAN_V      0.105      1.851      0.464 
                         R1_TITAN_V      0.210      3.700      0.928 
                       R1_TITAN_RTX      0.227      3.987      1.000 


restrict to mm2 : 20k 20-inch PMT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* RTX mode not helping 



     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2        ## reproducibility check 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558185148 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190518_211228     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.073      1.000      0.217 
                       R0_TITAN_RTX      0.119      1.615      0.350 
                         R0_TITAN_V      0.136      1.859      0.403 
                         R2_TITAN_V      0.314      4.274      0.927 
                         R1_TITAN_V      0.315      4.288      0.930 
                       R1_TITAN_RTX      0.338      4.610      0.999 
                       R2_TITAN_RTX      0.339      4.612      1.000 



     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558092492 --runlabel R0_TITAN_V_AND_TITAN_RTX --restrictmesh 2 --xanalytic
                    20190517_192812     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.073      1.000      0.225 
                       R0_TITAN_RTX      0.121      1.668      0.376 
                         R0_TITAN_V      0.133      1.831      0.413 
                         R2_TITAN_V      0.310      4.262      0.961 
                         R1_TITAN_V      0.311      4.273      0.963 
                       R1_TITAN_RTX      0.320      4.397      0.991 
                       R2_TITAN_RTX      0.322      4.436      1.000 

::

     geocache-;geocache-bench --xanalytic --restrictmesh 2
     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2  ## changed name of restrictmesh after generalize to accepting a command delimited list 

::

    /tmp/blyth/opticks/results/geocache-bench
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558178928 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190518_192848     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.075      1.000      0.220 
                       R0_TITAN_RTX      0.118      1.564      0.344 
                         R0_TITAN_V      0.136      1.810      0.399 
                         R2_TITAN_V      0.314      4.177      0.919 
                         R1_TITAN_V      0.314      4.178      0.920 
                       R2_TITAN_RTX      0.341      4.534      0.998 
                       R1_TITAN_RTX      0.342      4.543      1.000 



restrict to mm1 : 36k instanced small PMT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* looks really fast for 36k small PMT
* RTX mode gives some speedup on T-rex and V 


::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0 --rtx 1 --runfolder geocache-bench --runstamp 1558092010 --runlabel R1_TITAN_V --restrictmesh 1 --xanalytic
                    20190517_192010     metric      rfast      rslow 
                         R1_TITAN_V      0.018      1.000      0.502 
                         R2_TITAN_V      0.018      1.002      0.503 
                       R1_TITAN_RTX      0.021      1.131      0.568 
           R0_TITAN_V_AND_TITAN_RTX      0.021      1.135      0.570 
                       R2_TITAN_RTX      0.021      1.156      0.580 
                         R0_TITAN_V      0.032      1.766      0.887 
                       R0_TITAN_RTX      0.036      1.992      1.000 


restrict to global mm0
~~~~~~~~~~~~~~~~~~~~~~~~~~

* RTX mode not helping 

::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558091640 --runlabel R0_TITAN_V_AND_TITAN_RTX --restrictmesh 0 --xanalytic
                    20190517_191400     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.045      1.000      0.220 
                         R0_TITAN_V      0.080      1.768      0.389 
                       R0_TITAN_RTX      0.086      1.908      0.419 
                       R2_TITAN_RTX      0.201      4.456      0.980 
                       R1_TITAN_RTX      0.202      4.489      0.987 
                         R1_TITAN_V      0.205      4.548      1.000 
                         R2_TITAN_V      0.205      4.549      1.000 



combination of the slow ones : --xanalytic --enabledmergedmesh 0,2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* times are close to all 

::

    OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558180048 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 0,2
                    20190518_194728     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.099      1.000      0.194 
                       R0_TITAN_RTX      0.165      1.668      0.323 
                         R0_TITAN_V      0.185      1.878      0.363 
                       R1_TITAN_RTX      0.488      4.943      0.957 
                       R2_TITAN_RTX      0.488      4.945      0.957 
                         R2_TITAN_V      0.508      5.153      0.998 
                         R1_TITAN_V      0.510      5.166      1.000 


Reprodicibility check, after pixeltime fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* matches within 0.020

::
     geocache-;geocache-bench --xanalytic


     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558176275 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic
                    20190518_184435     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.122      1.000      0.202 
                       R0_TITAN_RTX      0.190      1.561      0.315 
                         R0_TITAN_V      0.217      1.785      0.360 
                       R2_TITAN_RTX      0.509      4.179      0.844 
                       R1_TITAN_RTX      0.513      4.217      0.852 
                         R2_TITAN_V      0.602      4.948      0.999 
                         R1_TITAN_V      0.603      4.952      1.000 



Disably ANYHIT for the ray and geometry and geometrygroup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nudges in right direction, but not by much.

::

    OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558081121 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic
                    20190517_161841     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.121      1.000      0.197 
                       R0_TITAN_RTX      0.190      1.577      0.311 
                         R0_TITAN_V      0.215      1.784      0.351 
                       R2_TITAN_RTX      0.485      4.022      0.792 
                       R1_TITAN_RTX      0.485      4.026      0.792 
                         R1_TITAN_V      0.611      5.072      0.998 
                         R2_TITAN_V      0.612      5.080      1.000 

Disably ANYHIT for the ray alone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With RT_RAY_FLAG_DISABLE_ANYHIT::

    +#if OPTIX_VERSION_MAJOR >= 6
    +  RTvisibilitymask mask = RT_VISIBILITY_ALL ;
    +  //RTrayflags      flags = RT_RAY_FLAG_NONE ;  
    +  RTrayflags      flags = RT_RAY_FLAG_DISABLE_ANYHIT ;  
    +  rtTrace(top_object, ray, prd, mask, flags);
    +#else
       rtTrace(top_object, ray, prd);
    +#endif

::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558077419 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic
                    20190517_151659     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.122      1.000      0.199 
                       R0_TITAN_RTX      0.188      1.542      0.307 
                         R0_TITAN_V      0.216      1.775      0.354 
                       R2_TITAN_RTX      0.490      4.028      0.802 
                       R1_TITAN_RTX      0.491      4.032      0.803 
                         R2_TITAN_V      0.611      5.017      0.999 
                         R1_TITAN_V      0.611      5.021      1.000 


Reproducibilioty check of analytic, few weeks later
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558076076 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic
                    20190517_145436     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.123      1.000      0.190 
                       R0_TITAN_RTX      0.190      1.547      0.294 
                         R0_TITAN_V      0.218      1.776      0.338 
                       R2_TITAN_RTX      0.523      4.261      0.810 
                       R1_TITAN_RTX      0.523      4.265      0.811 
                         R1_TITAN_V      0.645      5.256      0.999 
                         R2_TITAN_V      0.645      5.260      1.000 


Times for analytic geometry in seconsds 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

       .        20190424_204442     metric      rfast      rslow 

       R0_TITAN_V_AND_TITAN_RTX      0.122      1.000      0.188   
                   R0_TITAN_RTX      0.188      1.537      0.289 
                     R0_TITAN_V      0.219      1.790      0.337    
                   R1_TITAN_RTX      0.540      4.420      0.831     
                     R1_TITAN_V      0.650      5.319      1.000 

Example commandline::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 \
                --embedded --rtx 0 --runfolder geocache-bench --runstamp 1556109882 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic

Observations:

* cost for the exact geometry is about a factor 4 over the approximate triangulated ones
  (I'm happy that my CSG processing does not cost more that that)

* analytic really benefits from the core counts (TITAN V + TITAN RTX) 5120+4680 CUDA cores
  getting into the ballpark of triangulated geometries
  
  * i look forward to trying this benchmark on the GPU cluster nodes  
  
* RTX mode makes analytic times worse : by a factor of 2-3 

  * without using triangles, the only way the RT cores can help
    is with the BVH traversal being done in hardware : the fact 
    that timings get worse by as much as a factor of 3 suggests I should
    try some alternative OptiX acceleration/geometry setups  






With my triangles, ie no --xanalytic
-----------------------------------------

* This is with the torus-less GDML j1808 v3. 
* Note the 14.7M pixels. 
* The metric is launchAVG of five launch times.  
* OFF/ON refers to RTX execution approach
* OPTICKS_KEY OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
* commandline for the first of each group of runs is given as it was the same, the 
  differnence coming from envvars CUDA_VISIBLE_DEVICES and OPTICKS_RTX


::

    [blyth@localhost opticks]$ bench.py $LOCAL_BASE/opticks/results/geocache-bench
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --runfolder geocache-bench --runstamp 1555926978 --runlabel ON_TITAN_RTX
                    20190422_175618     metric      rfast      rslow 
                       ON_TITAN_RTX      0.056      1.000      0.391 
          OFF_TITAN_V_AND_TITAN_RTX      0.080      1.431      0.560 
                      OFF_TITAN_RTX      0.108      1.923      0.752 
                         ON_TITAN_V      0.117      2.083      0.815 
                        OFF_TITAN_V      0.143      2.557      1.000 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --runfolder geocache-bench --runstamp 1555940309 --runlabel ON_TITAN_RTX
                    20190422_213829     metric      rfast      rslow 
                       ON_TITAN_RTX      0.073      1.000      0.503 
          OFF_TITAN_V_AND_TITAN_RTX      0.081      1.109      0.557 
                         ON_TITAN_V      0.116      1.589      0.799 
                      OFF_TITAN_RTX      0.117      1.607      0.808 
                        OFF_TITAN_V      0.145      1.990      1.000 



* RTX speedup should be more by using  optix::GeometryTriangles




/usr/local/OptiX_600/SDK-src/optixGeometryTriangles
--------------------------------------------------------




Finding target volume to snap
-------------------------------

Found a good viewpoint, looking up at chimney::

    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OKTest --envkey --xanalytic --target 352851 --eye -1,-1,-1        ## analytic
    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OKTest --envkey --target 352851 --eye -1,-1,-1                    ## tri 

    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=-1 OpSnapTest --envkey --xanalytic --target 352851 --eye -1,-1,-1 


* target is 0-based 
* numbers listed in PVNames.txt from *vi* in the below are 1-based 
* 352851 is pLowerChimneyLS0x5b317e0 

GNodeLib/PVNames.txt::

    .1 lWorld0x4bc2710_PV
     2 pTopRock0x4bcd120
     3 pExpHall0x4bcd520
     4 lUpperChimney_phys0x5b308a0
     5 pUpperChimneyLS0x5b2f160
    ...

    352847 PMT_3inch_inner1_phys0x510beb0
    352848 PMT_3inch_inner2_phys0x510bf60
    352849 PMT_3inch_cntr_phys0x510c010
    352850 lLowerChimney_phys0x5b32c20
    352851 pLowerChimneyAcrylic0x5b31720
    352852 pLowerChimneyLS0x5b317e0
    352853 pLowerChimneySteel0x5b318b0
    352854 lSurftube_phys0x5b3c810
    352855 pvacSurftube0x5b3c120
    352856 lMaskVirtual_phys0x5cc1ac0



OpSnapTest
-------------

* :doc:`OpSnapTest_review`



Unless I am missing something. 

* perhaps compiling with CC 75 rather than current 70 ?
* also need to check with snap paths across more demanding geometry 

Take a look at a more demanding render over in env- rtow-



Perhaps JIT compilation killing perfermanance for TITAN RTX ?

cmake/Modules/OpticksCUDAFlags.cmake needs to handle a comma delimited COMPUTE_CAPABILITY ?::

     09 if(NOT (COMPUTE_CAPABILITY LESS 30))
     10 
     11    #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
     12    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
     13    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
     14 
     15    #list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
     16    # https://github.com/facebookresearch/Detectron/issues/185
     17 
     18    list(APPEND CUDA_NVCC_FLAGS "-O2")
     19    #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
     20    list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
     21 
     22    #list(APPEND CUDA_NVCC_FLAGS "-m64")
     23    #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
     24 
     25    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
     26    set(CUDA_VERBOSE_BUILD OFF)
     27 
     28 endif()




After Fixing Several Bugs 
-----------------------------------------------------------------

Bugs included:

* prelaunch doing launch
* mis-configured snap positions

And:

* increasing size 
* finding a region with lots of PMTs
* switch to trianglulated ( no --xanalytic )


::

    [blyth@localhost optixrap]$ t geocache-bench
    geocache-bench is a function
    geocache-bench () 
    { 
        echo "TITAN RTX";
        CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=0 $FUNCNAME-;
        CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 $FUNCNAME-;
        echo "TITAN V";
        CUDA_VISIBLE_DEVICES=0 OPTICKS_RTX=0 $FUNCNAME-;
        CUDA_VISIBLE_DEVICES=0 OPTICKS_RTX=1 $FUNCNAME-
    }


::

    geocache-bench- is a function
    geocache-bench- () 
    { 
        type $FUNCNAME;
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        $dbg OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=5,eyestartz=-1,eyestopz=-0.5" --size 5120,2880,1 --embedded $*
    }
    2019-04-21 22:53:02.945 INFO  [155128] [BOpticksKey::SetKey@45] from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    2019-04-21 22:53:11.224 INFO  [155128] [OTracer::report@157] OpTracer::snap
     trace_count              5 trace_prep        0.075119 avg  0.0150238
     trace_time         2.24857 avg   0.449713

    2019-04-21 22:53:11.224 INFO  [155128] [BTimes::dump@138] OTracer::report
                  validate000                 0.050209
                   compile000                    7e-06
                 prelaunch000                  1.59024
                    launch000                 0.132858
                    launch001                  0.10317
                    launch002                 0.102913
                    launch003                 0.105186
                    launch004                 0.101064
                    launchAVG                 0.109038
    2019-04-21 22:53:11.224 INFO  [155128] [BMeta::dump@53] Opticks OpTracer::snap
    CUDA_VISIBLE_DEVICES : 1
             OPTICKS_RTX : 0
             OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                 CMDLINE :  OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded
    2019-04-21 22:53:11.225 INFO  [155128] [OpTracer::snap@132] )
    geocache-bench- is a function

    2019-04-21 22:53:19.575 INFO  [155416] [BTimes::dump@138] OTracer::report
                  validate000                   0.0517
                   compile000                    8e-06
                 prelaunch000                  1.52944
                    launch000                 0.057163
                    launch001                 0.056131
                    launch002                 0.055519
                    launch003                 0.056188
                    launch004                 0.056055
                    launchAVG                0.0562112
    2019-04-21 22:53:19.576 INFO  [155416] [BMeta::dump@53] Opticks OpTracer::snap
    CUDA_VISIBLE_DEVICES : 1
             OPTICKS_RTX : 1
             OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                 CMDLINE :  OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded
    2019-04-21 22:53:19.576 INFO  [155416] [OpTracer::snap@132] )


    2019-04-21 22:53:28.396 INFO  [155678] [BTimes::dump@138] OTracer::report
                  validate000                 0.052362
                   compile000                    9e-06
                 prelaunch000                  1.74231
                    launch000                 0.139875
                    launch001                 0.146404
                    launch002                 0.143448
                    launch003                 0.143731
                    launch004                 0.141017
                    launchAVG                 0.142895
    2019-04-21 22:53:28.396 INFO  [155678] [BMeta::dump@53] Opticks OpTracer::snap
    CUDA_VISIBLE_DEVICES : 0
             OPTICKS_RTX : 0
             OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                 CMDLINE :  OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded
    2019-04-21 22:53:37.127 INFO  [155967] [BTimes::dump@138] OTracer::report
                  validate000                 0.051268
                   compile000                    8e-06
                 prelaunch000                  1.47854
                    launch000                 0.113385
                    launch001                 0.117253
                    launch002                 0.116381
                    launch003                 0.116277
                    launch004                 0.118571
                    launchAVG                 0.116373
    2019-04-21 22:53:37.128 INFO  [155967] [BMeta::dump@53] Opticks OpTracer::snap
    CUDA_VISIBLE_DEVICES : 0
             OPTICKS_RTX : 1
             OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                 CMDLINE :  OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded
    2019-04-21 22:53:37.128 INFO  [155967] [OpTracer::snap@132] )
    [blyth@localhost sysrap]$ 





