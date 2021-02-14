GGeo Orientation : Geometry Modelling and Persisting 
=======================================================

.. contents:: Table of contents
   :depth: 3 


* :doc:`../docs/orientation`

* https://bitbucket.org/simoncblyth/opticks/src/master/ggeo/
* https://bitbucket.org/simoncblyth/opticks/src/master/ggeo/GGeo.cc


GGeo 
    top level holder of geometry libraries : GMaterialLib, GSurfaceLib, GBndLib, GNodeLib, GGeoLib

GVolume
    created from G4VPhysicalVolume by X4PhysicalVolume::convertNode

GNodeLib
    collects GVolume instances *GNodeLib::addVolume* 

GInstancer
    Does multiple traversals over full GVolume tree, identifying 
    repeated geometry and labelling the tree nodes with the repeat index (*ridx*).
    Remainder geometry that does not pass instancing cuts on numbers of repeats 
    is left with repeat index zero.

    * used from GGeo::prepareVolumes 
    * GInstancer::createInstancedMergedMeshes 


.. comment  

   class include each have === headers, so are not containing them 




Analytic Geometry Overview : GPt, GPts, GParts
===================================================


``GPt``
    minimally captures node(ndIdx) to solid(lvIdx,csgIdx) association and boundaryName, 
    can also hold placement transforms

``X4PhysicalVolume::convertNode``
    ``GPt`` structs instanciated and associated with GVolume 
    (NB ``GParts`` instanciation now deferred )

``GMergedMesh::mergeVolumeAnalytic``
    ``GPt`` instances are collected into ``GPts`` after ``GPt::setPlacement``
    with the appropriate ``GVolume`` (base relative or global) transform 
    during ``GMergedMesh::mergeVolume``.

``GPts``
    collects ``GPt`` struct and handles their persisting to/from two transport buffers 
    (ipt and plc) and one ``GItemList`` of boundary spec  

``GGeo::deferredCreateGParts``
    invoked from ``GGeo::postLoadFromCache`` or ``GGeo::postDirectTranslation``. 

``GParts``
    analytic geometry holding ``npy/NCSG`` :doc:`../npy/orientation`

``GParts::Create``
    from from the volume combined ``GPts`` and ``std::vector<NCSG*>`` from ``m_meshlib``




.. include:: GPt.hh
   :start-after: /**
   :end-before: **/

.. include:: GPts.hh
   :start-after: /**
   :end-before: **/

.. include:: GParts.hh
   :start-after: /**
   :end-before: **/

.. include:: GGeo.hh
   :start-after: /**
   :end-before: **/
 
.. include:: GGeo.cc
   :start-after: /**10
   :end-before: 10**/

 

