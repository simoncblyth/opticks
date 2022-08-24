gxt_MOI_shakedown
===================

FIXED : No longer need MASK=t OR MASK=non to make the simtrace intersects visible 
---------------------------------------------------------------------------------------

::

    epsilon:g4cx blyth$ ./gxt.sh grab
    epsilon:g4cx blyth$ ./gxt.sh ana
    epsilon:g4cx blyth$ MASK=t ./gxt.sh ana


./gxt.sh ana
~~~~~~~~~~~~~~

* pv plot starts all black, zooming out see only the cegs grid rectangle of gs positions 
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

The genstep transform looks to be carrying the 4th column identity info::

    In [3]: t.genstep[0]
    Out[3]: 
    array([[    0.   ,     0.   ,       nan,     0.   ],
           [    0.   ,     0.   ,     0.   ,     1.   ],
           [    0.24 ,    -0.792,     0.562,     0.   ],
           [   -0.957,    -0.29 ,     0.   ,     0.   ],
           [    0.163,    -0.538,    -0.827,     0.   ],
           [-3354.313, 11057.688, 16023.353,    -0.   ]], dtype=float32)

        
Add the gs_tran 4th column fixup in ana/framegensteps.py::

     64         ## apply the 4x4 transform in rows 2: to the position in row 1 
     65         world_frame_centers = np.zeros( (len(gs), 4 ), dtype=np.float32 )
     66         for igs in range(len(gs)): 
     67             gs_pos = gs[igs,1]          ## normally origin (0,0,0,1)
     68             gs_tran = gs[igs,2:]        ## m2w with grid translation 
     69             gs_tran[:,3] = [0,0,0,1]   ## fixup 4th column, as may contain identity info
     70             world_frame_centers[igs] = np.dot( gs_pos, gs_tran )    
     71             #   world_frame_centers = m2w * grid_translation * model_frame_positon
     72         pass


* the "fixup 4th column" gets the genstep grid to correspond to the intersects and no longer need MASK=t 
  to see intersects 



TODO : The unfixed PMT mask is apparent
-----------------------------------------

This was fixed previously in j, but awaits the new integration SVN commits, 
to be brought to SVN. 




