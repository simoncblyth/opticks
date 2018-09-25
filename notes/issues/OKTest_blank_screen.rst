OKTest_blank_screen
======================


TODO
-------

* find a way to capture such problems into a test 
* fix the issue 


Issue : genstep/resource generalizations for direct mode, have broken legacy mode 
------------------------------------------------------------------------------------

::

    OKTest 
        # blank : no geometry appears, after usual repeated Q 

    OKTest --tracer
        # geometry appears, O: raytrace triangulated works

    OKTest --compute --save
    OKTest --load --geocenter
        # geometry appears, no propagation on pressing A, no photon history 
        # issue with new gensteps approach ?





