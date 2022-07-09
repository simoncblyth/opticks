gxr_shakedown
================


::

    gx
    ./gxr.sh dbg




issue 1 : output path unsustainable
-------------------------------------------------

::

    2022-07-09 22:13:47.452 INFO  [375788] [CSGOptiX::launch@718]  (width, height, depth) ( 1920,1080,1) 0.0078
    2022-07-09 22:13:47.452 ERROR [375788] [CSGOptiX::render_snap@797]  name cx-1 outpath /tmp/blyth/opticks/cx-1.jpg dt 0.00778148 topline [G4CXRenderTest] botline [    0.0078]
    2022-07-09 22:13:47.452 INFO  [375788] [CSGOptiX::snap@819]  path /tmp/blyth/opticks/cx-1.jpg
    2022-07-09 22:13:47.452 INFO  [375788] [CSGOptiX::snap@828]  path_ [/tmp/blyth/opticks/cx-1.jpg]
    2022-07-09 22:13:47.452 INFO  [375788] [CSGOptiX::snap@829]  topline G4CXRenderTestPIP  td:1 pv:2 av:2 WITH_PRD  
    NP::Write dtype <f4 ni     1080 nj  1920 nk  4 nl  -1 nm  -1 no  -1 path /tmp/blyth/opticks/isect.npy
    NP::Write dtype <f4 ni     1080 nj  1920 nk  4 nl  4 nm  -1 no  -1 path /tmp/blyth/opticks/photon.npy
    2022-07-09 22:13:52.592 INFO  [375788] [CSGOptiX::saveMeta@895] /tmp/blyth/opticks/cx-1.json
    N[blyth@localhost g4cx]$ 



