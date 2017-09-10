Viz On Headless GPU Compute Server
====================================

Objective
-----------

* find way to exercise Opticks, and test performance on a headless GPU server,
  that does not even have OpenGL libs


SnapTest : OptiX compute geometry ray trace snapshot executable
-------------------------------------------------------------------


* most functionality already exists just need to extracate


* oxrap- should have all the dependencies needed to load geometry 
  and run the ray tracer without any OpenGL graphics dependancy, 
  hmm probably need to operate at higher level than oxrap

* Same flow as ok/tests/OTracerTest.cc but with OptiX compute extracated, 
  and added ppm (or png) writing.


ok/tests/OTracerTest.cc
--------------------------

::

     39 int main(int argc, char** argv)
     40 {
     ..   ...  logging setup
     63 
     64     OKMgr ok(argc, argv, "--tracer" );
     65 
     66     ok.visualize();
     67 
     68     exit(EXIT_SUCCESS);
     69 }


OpMgr ?
---------

Actually compute only Opticks is an increasing need, not 
just for SnapTest... so make an OpMgr ? migrate ok/OKMgr 
into okop/OpMgr stripping out the viz, so okop tests using 
OpMgr can be compiled and linked on machines without OpenGL


Which pkg to put SnapTest/OpTest ?
--------------------------------------

* https://simoncblyth.bitbucket.io/opticks/docs/overview.html#project-dependencies

* okop 
* oxrap 


GUI snapping
-----------------

::

    op --j1707 --gltf 3 --tracer --target 12 --eye 0.85,0.85,0. --snap --rendermode +bb0,+in1,+in2,+in3,-global



::

    okop-snap()
    {
        ## intended to give same snap as okop-snap-gui : currently getting black screen
        op --snap --j1707 --gltf 3 --tracer --target 12 --eye 0.85,0.85,0.
        libpng-;libpng-- /tmp/snap.ppm
    }


    okop-snap-gui()
    {
        ## to make a snap, need to switch to OptiX render mode with "O" key once GUI has launched
        op  --j1707 --gltf 3 --tracer --target 12 --eye 0.85,0.85,0. --rendermode +bb0,+in1,+in2,+in3,-global
        libpng-;libpng-- /tmp/snap.ppm
    }





