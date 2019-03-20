OpticksResource_sledgehammer_needed
=====================================

Lots of dead code in OpticksResource.  Once gain confidence 
in the key based approach can take sledgehammer to OpticksResource,
and do away with opticksdata in current form instead providing 
a key based way to use precooked geocaches.

Note that with key based running, the old opticksdata.ini is still being read::

    2019-03-20 20:09:08.218 INFO  [180919] [G4Opticks::translateGeometry@172] ( Opticks
    2019-03-20 20:09:08.220 ERROR [180919] [BOpticksResource::init@87] layout : 0
    2019-03-20 20:09:08.220 ERROR [180919] [OpticksResource::init@256] OpticksResource::init
    2019-03-20 20:09:08.221 ERROR [180919] [OpticksResource::readOpticksEnvironment@524]  inipath /home/blyth/local/opticks/opticksdata/config/opticksdata.ini
    2019-03-20 20:09:08.221 INFO  [180919] [BOpticksResource::setupViaKey@426] 
                 BOpticksKey  : KEYSOURCE
                        spec  : CerenkovMinimal.X4PhysicalVolume.World.792496b5e2cc08bdf5258cc12e63de9f
                     exename  : CerenkovMinimal
             current_exename  : CerenkovMinimal
                       class  : X4PhysicalVolume
                     volname  : World
                      digest  : 792496b5e2cc08bdf5258cc12e63de9f
                      idname  : CerenkovMinimal_World_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2019-03-20 20:09:08.221 INFO  [180919] [BOpticksResource::setupViaKey@464]  idname CerenkovMinimal_World_g4live idfile g4ok.gltf srcdigest 792496b5e2cc08bdf5258cc12e63de9f idpath /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1
    2019-03-20 20:09:08.221 INFO  [180919] [OpticksResource::assignDetectorName@419] OpticksResource::assignDetectorName m_detector g4live
    2019-03-20 20:09:08.221 ERROR [180919] [OpticksResource::init@278] OpticksResource::init DONE
    2019-03-20 20:09:08.221 INFO  [180919] [G4Opticks::translateGeometry@174] ) Opticks






