gxt_MOI_shakedown
===================


TODO : Have to use MASK=t OR MASK=non to make the simtrace intersects visible ? Why ?
---------------------------------------------------------------------------------------

::

    epsilon:g4cx blyth$ ./gxt.sh grab
    epsilon:g4cx blyth$ ./gxt.sh ana
    epsilon:g4cx blyth$ MASK=t ./gxt.sh ana


./gxt.sh ana
~~~~~~~~~~~~~~

* pv plot starts all black, zooming out see the cegs grid rectangle of gs positions and simulate pos
* mp plot stars all white, no easy way to zoom out  

MASK=t ./gxt.sh ana
~~~~~~~~~~~~~~~~~~~~~~

* pv plot immediately shows the simtrace isect of the ~7 PMTs 
* zooming out see lots more 
* also zooming out more see the genstep grid rectangle, 
  which is greatly offset from the intersects

* mp plot, blank white again but lots of key entries


gx/tests/G4CXSimtraceTest.py 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




TODO : The unfixed PMT mask is apparent
-----------------------------------------

This was fixed previously in j, but awaits the new integration SVN commits, 
to be brought to SVN. 




