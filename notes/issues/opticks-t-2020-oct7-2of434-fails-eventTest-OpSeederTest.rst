opticks-t-2020-oct7-2of434-fails-eventTest-OpSeederTest
==========================================================


Bandaid
--------


::

    321 /**
    322 OpticksGen::targetGenstep
    323 ---------------------------
    324 
    325 Targetted positioning and directioning of the torch requires geometry info, 
    326 which is not available within npy- so need to externally setFrameTransform
    327 based on integer frame volume index.
    328 
    329 Observe that with "--machinery" option the node_index from 
    330 GenstepNPY::getFrame (iframe.x) is staying at -1. So have 
    331 added a reset to 0 for this. See:
    332 
    333 * notes/issues/opticks-t-2020-oct7-2of434-fails-eventTest-OpSeederTest.rst
    334 
    335 **/
    336 
    337 void OpticksGen::targetGenstep( GenstepNPY* gs )
    338 {
    339     if(gs->isFrameTargetted())
    340     {    
    341         LOG(info) << "frame targetted already  " << gformat(gs->getFrameTransform()) ;
    342     }    
    343     else
    344     {   
    345         if(m_hub)
    346         {
    347             glm::ivec4& iframe = gs->getFrame();
    348             
    349             int node_index = iframe.x ;
    350             if(node_index < 0)
    351             {
    352                 LOG(fatal) << "node_index from GenstepNPY is -1, resetting to 0" ;
    353                 node_index = 0 ; 
    354             }   
    355             
    356             glm::mat4 transform = m_ggeo->getTransform( node_index );
    357             LOG(info) << "setting frame " << iframe.x << " " << gformat(transform) ;
    358             gs->setFrameTransform(transform);
    359         }   
    360         else
    361         {
    362             LOG(warning) << "SKIP AS NO GEOMETRY " ;
    363         }   
    364         
    365     }
    366 }   




Issue
------

::

    FAILS:  2   / 434   :  Wed Oct  7 16:58:51 2020   
      20 /28  Test #20 : OptiXRapTest.eventTest                        Child aborted***Exception:     5.26   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     5.29   
    [blyth@localhost opticks]$ 


Both failing from the same assert, eventTest::

    2020-10-07 17:03:42.484 INFO  [232243] [OpticksHub::loadGeometry@332] ]
    eventTest: /home/blyth/opticks/ggeo/GNodeLib.cc:387: glm::mat4 GNodeLib::getTransform(unsigned int) const: Assertion `index < num_transforms' failed.
    Aborted (core dumped)
    [blyth@localhost opticks]$ 


    2020-10-07 17:06:51.422 FATAL [237059] [Opticks::setProfileDir@562]  dir /tmp/blyth/opticks/eventTest/evt/g4live/machinery
    2020-10-07 17:06:51.422 INFO  [237059] [OpticksHub::loadGeometry@332] ]
    eventTest: /home/blyth/opticks/ggeo/GNodeLib.cc:387: glm::mat4 GNodeLib::getTransform(unsigned int) const: Assertion `index < num_transforms' failed.

    #4  0x00007ffff4e95a05 in GNodeLib::getTransform (this=0x1687d80, index=4294967295) at /home/blyth/opticks/ggeo/GNodeLib.cc:387
    #5  0x00007ffff4e906a0 in GGeo::getTransform (this=0x693d70, index=4294967295) at /home/blyth/opticks/ggeo/GGeo.cc:1441
    #6  0x00007ffff628899a in OpticksGen::targetGenstep (this=0x7145dd0, gs=0x7146180) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:337
    #7  0x00007ffff6288f65 in OpticksGen::makeFabstep (this=0x7145dd0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:392
    #8  0x00007ffff628830f in OpticksGen::makeLegacyGensteps (this=0x7145dd0, code=9) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:237
    #9  0x00007ffff6288110 in OpticksGen::initFromLegacyGensteps (this=0x7145dd0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:206
    #10 0x00007ffff6287963 in OpticksGen::init (this=0x7145dd0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:140
    #11 0x00007ffff628778a in OpticksGen::OpticksGen (this=0x7145dd0, hub=0x7fffffff9960) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:80
    #12 0x00007ffff62840fc in OpticksHub::init (this=0x7fffffff9960) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:267
    #13 0x00007ffff6283d66 in OpticksHub::OpticksHub (this=0x7fffffff9960, ok=0x7fffffff9a00) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:218
    #14 0x00000000004060d9 in main (argc=1, argv=0x7fffffffadf8) at /home/blyth/opticks/optixrap/tests/eventTest.cc:53
    (gdb) 


    (gdb) f 6
    #6  0x00007ffff5fdc99a in OpticksGen::targetGenstep (this=0x7147f90, gs=0x7148340) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:337
    337	            glm::mat4 transform = m_ggeo->getTransform( iframe.x );
    (gdb) p iframe.x
    $1 = -1

    (gdb) f 5
    #5  0x00007ffff4be46a0 in GGeo::getTransform (this=0x695e50, index=4294967295) at /home/blyth/opticks/ggeo/GGeo.cc:1441
    1441	    return m_nodelib->getTransform(index); 
    (gdb) p index
    $2 = 4294967295

    (gdb) f 4
    #4  0x00007ffff4be9a05 in GNodeLib::getTransform (this=0x1689e40, index=4294967295) at /home/blyth/opticks/ggeo/GNodeLib.cc:387
    387	    assert( index < num_transforms ); 

    (gdb) p index
    $3 = 4294967295

    (gdb) p num_transforms
    $4 = 12230
    (gdb) 




Add OpticksHubTest which has same assert when using  "--machinery" option::


    #include "Opticks.hh"
    #include "OpticksHub.hh"
    #include "OPTICKS_LOG.hh"

    int main(int argc, char** argv)
    {
        OPTICKS_LOG(argc, argv);

        Opticks ok(argc, argv);
        OpticksHub hub(&ok);       

        if(hub.getErr()) LOG(fatal) << "hub error " << hub.getErr() ; 

        return 0 ; 
    }




