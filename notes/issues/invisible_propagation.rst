FIXED : Invisible Propagation 
==================================

Fix
----

Space and time Domains were not setup propertly. Attempt to 
make domain handling more robust by tieing Opticks::configureDomains
to the setting of the space domain and asserting that 
this is done within Opticks::makeEvent.


Issue
-------

Whilst doing high level refactor in ggeoview/OpticksApp the
propagation runs but the visualization fails to appear both 
from the old and new mains::

    GGeoViewTest
    OpticksAppTest  

This is an all too common occurence.

Symptoms:

* material and flag indices look normal, so the propagation actually happened OK
* furthermore the "Photon Flag Selection" works so nothing fundamentally wrong with 
  the OpenGL photon buffer 
* animation time is stuck at zero, probably a domain issue 


Save the propagation and examine with **tevt.py**::


    GGeoViewTest --save 
    OpticksAppTest --save    ## get precisely same values with both the binaries

    simon:issues blyth$ ipython -i $(which tevt.py) --  --det dayabay --tag 1
    ...

    In [5]: evt.post    ## photon buffer, which does not undergo compression/decompression look normal
    Out[5]: 
    A()sliced
    A([[ -20366.123, -798767.75 ,   -7962.852,      14.132],
           [ -19641.139, -799302.25 ,   -9119.9  ,      15.018],
           [ -18080.379, -799015.812,   -4998.   ,       8.911],
           ..., 
           [ -17409.719, -797559.812,   -8459.558,      14.606],
           [ -15950.902, -798960.812,   -7200.205,      11.719],
           [ -18439.031, -799557.938,   -7175.665,       3.542]], dtype=float32)



    In [12]: evt.rpost_(0)     ## record buffer, looks like missing spatial domain
    Out[12]: 
    A()sliced
    A([[-1000.031, -1000.031, -1000.031,     0.098],
           [-1000.031, -1000.031, -1000.031,     0.098],
           [-1000.031, -1000.031, -1000.031,     0.098],
           ..., 
           [-1000.031, -1000.031, -1000.031,     0.098],
           [-1000.031, -1000.031, -1000.031,     0.098],
           [-1000.031, -1000.031, -1000.031,     0.098]])

    In [15]: evt.rpost_(3)
    Out[15]: 
    A()sliced
    A([[-1000.031, -1000.031, -1000.031,    11.298],
           [-1000.031, -1000.031, -1000.031,    14.85 ],
           [-1000.031, -1000.031, -1000.031,     8.466],
           ..., 
           [-1000.031, -1000.031, -1000.031,    12.83 ],
           [-1000.031, -1000.031, -1000.031,    10.309],
           [    0.   ,     0.   ,     0.   ,     0.   ]])



Post propagation dump, shows incorrect space domain::

    2016-08-26 11:47:51.287 INFO  [1140577] [OPropagator::downloadEvent@391] OPropagator::downloadEvent DONE
    2016-08-26 11:47:51.287 INFO  [1140577] [OpticksDomain::dump@145] OEngineImp::saveEvt dumpDomains
     space_domain      0.0000,0.0000,0.0000,1000.0000
     time_domain       0.0000,200.0000,50.0000,0.0000
     wavelength_domain 60.0000,820.0000,20.0000,760.0000




Where space domain comes from::

    simon:optickscore blyth$ opticks-find \>setSpaceDomain
    ./cfg4/CG4.cc:    m_opticks->setSpaceDomain(ce);
    ./cfg4/CG4.cc:    m_evt->setSpaceDomain(m_opticks->getSpaceDomain());
    ./optickscore/Opticks.cc:    evt->setSpaceDomain(getSpaceDomain());   // default, will be updated in App:registerGeometry following geometry loading
    ./optickscore/OpticksEvent.cc:    m_domain->setSpaceDomain(space_domain) ; 
    ./opticksgeo/OpticksGeometry.cc:    m_opticks->setSpaceDomain( glm::vec4(ce0.x,ce0.y,ce0.z,ce0.w) );
    ./opticksgeo/OpticksHub.cc:        m_evt->setSpaceDomain(m_opticks->getSpaceDomain());


    On loading geometry gf

    [OpticksGeometry::registerGeometry@344] OpticksGeometry::registerGeometry setting opticks SpaceDomain :  x -16520 y -802110 z -7125 w 7710.56


Domains reside in too many places

* Opticks
* OpticksEvent
* Composition

 




