ExtG4 Orientation : Translates Geant4->GGeo only 
==================================================

* :doc:`../docs/orientation`

* https://bitbucket.org/simoncblyth/opticks/src/master/extg4/
* https://bitbucket.org/simoncblyth/opticks/src/master/extg4/X4PhysicalVolume.cc


X4 Class names mostly correspond to Geant4 classnames 



X4PhysicalVolume 
    top level translator 

X4Solid
    converts many G4VSolid primitive shapes (sphere, box, polycone, ellipsoid, torus, ...) 
    into npy/nnode : either single nodes or trees. There is not a one-to-one correspondence, 
    with Opticks tending towards trees to avoid having to implement too many primitives in CUDA.






    







