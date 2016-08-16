
OptiX 400 : Interop Wierdness
=================================

TODO
----

* change persisting location for interop and compute for 
  photon comparison


Debug Approaches
------------------

* need to do something technically similar to the 
  full interop simulation run, but drastically simpler :
  may be replace the generate.cu with an artificial propagation
  eg just straight line propagate initial photons

* seems only a subset of photons are afflicted, perhaps
  a problem with RNG ?

* loading of persisted evts seems working OK, 
  this just exercises OpenGL no interop needed

* check for buffer overwriting ... when using 
  OpenGL and OptiX together



OOAxisAppCheck 
~~~~~~~~~~~~~~~~


*oglrap-/AxisApp.cc*
      sets up OpenGL viz using a simple scene, frame, composition etc..

      BUT axis data is too simple for realistic debug, so add some fake
      nopstep or record data

*opticksgl-/OAxisTest.cc*
      sets up optixrap-/axisTest.cu

*opticksgl-/tests/OOAxisAppCheck.cc*
      sits in renderloop calling the axisModify via OptiX launch 
      


GGeoViewTest : Interop and Compute Mode not matching
-----------------------------------------------------------

Comparing interop with compute mode events, divergence is apparent::

   // interop
   GGeoViewTest 

   // compute 
   GGeoViewTest --compute --save
   GGeoViewTest --load


Prior to 400 there was precise digest matching agreement between 
interop and compute.  Now the compute mode looks normal but 
interop has several issues.

interop
~~~~~~~~~~

Notice some rafts of parallel slowly propagating photons.
Looking at photons in different history sequences suggests 
those ending in AB (bulk absorb) are primary mis-behavers.


tpmt-- : origin attraction and swarming
------------------------------------------

interop
~~~~~~~~~~

Small numbers of slower photons seem attracted to origin, 
photons exhibit swarming 

compute
~~~~~~~~

None of the wierdness apparent on load, and matching g4::

    tpmt-- --compute 
    tpmt-- --load

    tpmt-- --compute --tcfg4


compute mode still matching g4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:issues blyth$ tpmt.py 
    /Users/blyth/opticks/ana/tpmt.py
    writing opticks environment to /tmp/blyth/opticks/opticks_env.bash 
    INFO:__main__:tag 10 src torch det PmtInBox 
    INFO:__main__: a : PmtInBox/torch/ 10 :  20160816-1853 /tmp/blyth/opticks/evt/PmtInBox/fdomtorch/10.npy 
    INFO:__main__: b : PmtInBox/torch/-10 :  20160816-1855 /tmp/blyth/opticks/evt/PmtInBox/fdomtorch/-10.npy 
                          10:PmtInBox   -10:PmtInBox           c2 
                     8cd         67948        68252             0.68  [3 ] TO BT SA
                     7cd         21648        21369             1.81  [3 ] TO BT SD
                    8ccd          4581         4539             0.19  [4 ] TO BT BT SA
                      4d          3794         3864             0.64  [2 ] TO AB
                     86d           640          617             0.42  [3 ] TO SC SA
                     4cd           444          427             0.33  [3 ] TO BT AB
                    4ccd           350          362             0.20  [4 ] TO BT BT AB
                     8bd           283          259             1.06  [3 ] TO BR SA
                    8c6d            81           84             0.05  [4 ] TO SC BT SA
                   86ccd            51           57             0.33  [5 ] TO BT BT SC SA
                  8cbbcd            36           53             3.25  [6 ] TO BT BR BR BT SA
                     46d            40           30             1.43  [3 ] TO SC AB
                    7c6d            20           28             1.33  [4 ] TO SC BT SD
                     4bd            28           21             1.00  [3 ] TO BR AB
                8cbc6ccd             9            3             0.00  [8 ] TO BT BT SC BT BR BT SA
                    866d             8            4             0.00  [4 ] TO SC SC SA
                   8cc6d             7            7             0.00  [5 ] TO SC BT BT SA
                    86bd             6            4             0.00  [4 ] TO BR SC SA
                    8b6d             3            6             0.00  [4 ] TO SC BR SA
              cbccbbbbcd             4            0             0.00  [10] TO BT BR BR BR BR BT BT BR BT
                              100000       100000         0.91 
                          10:PmtInBox   -10:PmtInBox           c2 
                     ee4         90040        90048             0.00  [3 ] MO Py Py
                    44e4          4931         4901             0.09  [4 ] MO Py MO MO
                      44          3794         3864             0.64  [2 ] MO MO
                     444           991          927             2.14  [3 ] MO MO MO
                    ee44           101          113             0.67  [4 ] MO MO Py Py
                   444e4            52           58             0.33  [5 ] MO Py MO MO MO
                  44eee4            40           54             2.09  [6 ] MO Py Py Py MO MO
                    4444            17           14             0.29  [4 ] MO MO MO MO
                   44e44             8            7             0.00  [5 ] MO MO Py MO MO
                44ee44e4             6            3             0.00  [8 ] MO Py MO MO Py Py MO MO
                444e44e4             5            0             0.00  [8 ] MO Py MO MO Py MO MO MO
              44e4eeeee4             4            0             0.00  [10] MO Py Py Py Py Py MO Py MO MO
                  ee44e4             0            4             0.00  [6 ] MO Py MO MO Py Py
                   ee444             2            0             0.00  [5 ] MO MO MO Py Py
              44edbe44e4             2            0             0.00  [10] MO Py MO MO Py OV Vm Py MO MO
                  4444e4             0            2             0.00  [6 ] MO Py MO MO MO MO
              4ebdbe44e4             0            1             0.00  [10] MO Py MO MO Py OV Vm OV Py MO
              4e5dbe44e4             0            1             0.00  [10] MO Py MO MO Py OV Vm Bk Py MO
              eebdbe44e4             1            0             0.00  [10] MO Py MO MO Py OV Vm OV Py Py
                 44ee444             1            0             0.00  [7 ] MO MO MO Py Py MO MO
                              100000       100000         0.78 




op --cerenkov
------------------

interop
~~~~~~~~

::

   op --cerenkov

10 percent (53474) of material sequence selection with NULL label, 
and slow backwards photons. 

Same number of missers (MI) in history selection. 

compute
~~~~~~~~~

::

    op --cerenkov --compute --save 
    op --cerenkov --load 


10 percent NULL still there, no visible photons



