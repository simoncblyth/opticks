g4ok_investigate_zero_hits
============================

num_hits in the NPY hits buffer is zero, but have entries in OpHitCollection ?

* Am I just missing some shovelling ?
* Ahha : or is this the difference between GPU handling of SD and G4 ??  

  * Need to add sensitive surfaces for GPU side to yield hits see :doc:`G4OK_SD_Matching` 


Context
----------

* story continues in :doc:`g4ok_hit_matching`


ckm-- ckm-go running giving zero GPU side hits
---------------------------------------------------

::

    2019-03-13 19:29:53.362 INFO  [1511869] [NMeta::write@206] write to /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/source/evt/g4live/natural/-1/so.json
    2019-03-13 19:29:53.362 ERROR [1511869] [EventAction::EndOfEventAction@42]  num_hits 0 hits 0x112e0a4d0
    2019-03-13 19:29:53.362 INFO  [1511869] [SensitiveDetector::DumpHitCollections@159]  query SD0/OpHitCollectionA hcid    0 hc 0x112c64530 hc.entries 91
    2019-03-13 19:29:53.362 INFO  [1511869] [SensitiveDetector::DumpHitCollections@159]  query SD0/OpHitCollectionB hcid    1 hc 0x112c64578 hc.entries 17
    2019-03-13 19:29:53.362 INFO  [1511869] [RunAction::EndOfRunAction@30] .
    2019-03-13 19:29:53.363 INFO  [1511869] [RunAction::EndOfRunAction@32] G4Opticks ok 0x110c24a80 opmgr 0x110e818b0
    BOpticksKey  KEYSOURCE 
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

    /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1

    2019-03-13 19:29:53.363 INFO  [1511869] [CAlignEngine::~CAlignEngine@48]  saving cursors to /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/CAlignEngine.npy
    Process 25430 exited with status = 0 (0x00000000) 



Smoking Gun : bouncemax 0 in G4OK embedded commandline
----------------------------------------------------------

G4OK bouncemax zero, historical for checking generation::

    42 const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0 --bouncemax 0"  ;


Removing "--bouncemax 0" yields 42 Opticks side hits, less than 91+17=108 from G4::

    2019-03-13 20:48:56.582 ERROR [1545649] [EventAction::EndOfEventAction@42]  num_hits 42 hits 0x136a757e0
    2019-03-13 20:48:56.582 INFO  [1545649] [SensitiveDetector::DumpHitCollections@159]  query SD0/OpHitCollectionA hcid    0 hc 0x110ccbd00 hc.entries 91
    2019-03-13 20:48:56.582 INFO  [1545649] [SensitiveDetector::DumpHitCollections@159]  query SD0/OpHitCollectionB hcid    1 hc 0x110ccbd48 hc.entries 17
    2019-03-13 20:48:56.583 INFO  [1545649] [RunAction::EndOfRunAction@30] .
    2019-03-13 20:48:56.583 INFO  [1545649] [RunAction::EndOfRunAction@32] G4Opticks ok 0x110f54ac0 opmgr 0x110f72b60



