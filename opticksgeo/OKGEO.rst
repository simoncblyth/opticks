OKGEO : Opticks middle management
=====================================


OpticksHub
    Has accumulated a bit too much : has become OpticksOctopus with arms everywhere, 
    but extracating will take a long time, so just chip away at functionality 
    and indescriminate usage when see the opportunity 
    (eg when can directly use a constituent do so rather than going via the hub) 

    Another example, OpticksEvent handling is done entirely via the m_run visitor
    from Opticks : its more expressive to use m_run directly rather than via 
    the hub. 

    Formerly envisioned the hub as a switching ground between multiple geometry 
    instances GGeo/GScene/GGeoTest (all GGeoBase) for triangulated/analytic/test geometry
    but now are aiming at single GGeo : to cover all of these

OpticksGeometry
    GGeo holder/loader/fixer 

OpticksGen
    high level Genstep control

OpticksIdx
    Wrapper around hostside(only?) indexing functionality 

OpticksAim

OpticksGun


See Also
----------

* :doc:`../opticksgeo/OKGEO`
* :doc:`../okop/OKOP`
* :doc:`../optixrap/OXRAP`
* :doc:`../thrustrap/THRAP`


