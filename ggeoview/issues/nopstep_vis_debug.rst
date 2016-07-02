Nopstep failing to appear : viz debugg
========================================

As usual when adding a vizualization : nothing appears and no error messages.

::

   ggv-;ggv-g4gun
         # geant4 particle gun simulation within default DYB geometry, loaded from GDML

   ggv-;ggv-g4gun --load
         # visualize the geant4 propagation, with GGeoView

Gun particles start from mid GdLS (volume 3153)::

      0.543      0.840      0.000 -18079.453 
      -0.840      0.543      0.000 -799699.438 
      0.000      0.000      1.000  -6605.000 
      0.000      0.000      0.000      1.000  

At save they look to be in the correct region::

        [2016-May-25 16:41:06.388981]:info: NumpyEvt::save (nopstep) (144,4,4) 

        (  0)  -18079.453  -799699.438   -6605.000       0.000 
        (  0)       0.000       0.000       1.000       1.000 
        (  0)       0.543      -0.840       0.000   10000.000 
        (  0)       0.000       0.000       0.000       0.000 
        (  1)  -18079.453  -799699.438   -6601.928       0.025 
        (  1)      -0.007       0.073       0.997       1.000 
        (  1)       0.543      -0.840       0.000    7611.374 

With a 10 MeV muon the range of the nopstep is very large due to some neutrinos that exit the world.::

    Rdr::upload mvn name rpos type  numbytes 9216 stride 64 offset 0 count 144 extent 2400000.000000
              m_low vec3  -735007.625 -799902.000 -2400000.000  
             m_high vec3  560972.188   548652.250  2400000.000  
       m_dimensions vec3  1295979.750 1348554.250  4800000.000  
           m_center vec3  -87017.719 -125624.875      0.000  
    m_model_to_world mat4
    2400000.000      0.000      0.000 -87017.719 
         0.000 2400000.000      0.000 -125624.875 
         0.000      0.000 2400000.000      0.000 
         0.000      0.000      0.000      1.000 

    In [19]: (a[:,0,0].min(),a[:,0,1].min(),a[:,0,2].min(),a[:,0,3].min())
    Out[19]: (-735007.62, -799902.0, -2400000.0, 0.0)

    In [20]: (a[:,0,0].max(),a[:,0,1].max(),a[:,0,2].max(),a[:,0,3].max())
    Out[20]: (560972.19, 548652.25, 2400000.0, 12151.727)


Switch to 10 MeV e+ (later 100 MeV), much more compact::

    Rdr::upload mvn name rpos type  numbytes 13568 stride 64 offset 0 count 212 extent 440.272217
              m_low vec3  -18345.670 -799975.375  -7250.606  
             m_high vec3  -18067.615 -799503.625  -6370.062  
       m_dimensions vec3     278.055    471.750    880.544  
           m_center vec3  -18206.643 -799739.500  -6810.334  



Adding a simple "dbg" point shader succeeds to find the gross problems.


