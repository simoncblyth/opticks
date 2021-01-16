GGeo Orientation : Geometry Modelling and Persisting 
=======================================================

* :doc:`../docs/orientation`


GGeo 
    top level holder of geometry libraries : GMaterialLib, GSurfaceLib, GBndLib, GNodeLib, GGeoLib

GVolume
    created from G4VPhysicalVolume by X4PhysicalVolume::convertNode

GNodeLib
    collects GVolume instances *GNodeLib::addVolume* 

GParts
    analytic geometry holding npy/NCSG :doc:`../npy/orientation`

GInstancer
    Does multiple traversals over full GVolume tree, identifying 
    repeated geometry and labelling the tree nodes with the repeat index (*ridx*).
    Remainder geometry that does not pass instancing cuts on numbers of repeats 
    is left with repeat index zero.

    * used from GGeo::prepareVolumes 
    * GInstancer::createInstancedMergedMeshes 



