making-raytrace-snapshots
============================

Using OTracerTest is faster, no need to propagate::

    OPTICKS_RESOURCE_LAYOUT=103 OKTest --gltf 3 -G    
           create a fresh geocahce into slot 103

    OPTICKS_RESOURCE_LAYOUT=103 OKTest --gltf 3 
           propagate test 
    
    OPTICKS_RESOURCE_LAYOUT=103 OTracerTest --gltf 3
           for just geometry this is faster

    OPTICKS_RESOURCE_LAYOUT=103 OTracerTest --size 2560,1440,1 --gltf 3
            explicit size to avoid the quarter view problem    


Note the interface is deceptive regarding bookmarks, they
are loaded but are not listed until first used.  

Where are the bookmarks::

    epsilon:~ blyth$ ll /Users/blyth/.opticks/dayabay/State/
    total 40
    drwxr-xr-x  3 blyth  staff   96 Mar 25 15:02 ..
    -rw-r--r--  1 blyth  staff  411 Jun 28 09:20 001.ini
    -rw-r--r--  1 blyth  staff  411 Jun 28 09:22 002.ini
    -rw-r--r--  1 blyth  staff  408 Jun 28 09:32 003.ini
    drwxr-xr-x  7 blyth  staff  224 Jun 28 09:33 .
    -rw-r--r--  1 blyth  staff  409 Jun 28 09:33 004.ini
    -rw-r--r--  1 blyth  staff  401 Jun 28 11:22 000.ini

    epsilon:~ blyth$ date
    Thu Jun 28 11:23:18 HKT 2018
    epsilon:~ blyth$ 
    epsilon:~ blyth$ 
    epsilon:~ blyth$ cd /Users/blyth/.opticks/dayabay/State/
    epsilon:State blyth$ cat 001.ini 
    [camera]
    far=19119.1719
    near=161.8355
    scale=161.8355
    zoom=2.5633
    [clipper]
    cutnormal=1.0000,0.0000,0.0000
    cutplane=1.0000,0.0000,0.0000,1.0000
    cutpoint=0.0000,0.0000,0.0000
    [scene]
    scenetarget=11566
    [trackball]
    orientation=1.0000,0.0000,0.0000,0.0000
    radius=1.0000
    translate=0.0000,0.0000,0.0000
    translatefactor=1000.0000
    [view]
    eye=-2.1509,-0.6663,-0.3384
    look=-1.5381,-1.6631,-1.1326
    up=0.3997,-0.4082,0.8207



