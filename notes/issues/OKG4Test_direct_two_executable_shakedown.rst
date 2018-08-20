OKG4Test_direct_two_executable_shakedown
=========================================


Attempt to get iteration started without alignment : using direct key geocache + primary persisting
----------------------------------------------------------------------------------------------------

::

    ckm--(){ ckm-cd ; ./go.sh ; } 

        ## 1st executable : setup + save + bi-simulate ( but not instrumented) 
        ## 
        ##     Cerenkov minimal sets up geometry+primaries 
        ##     then persists to key geocache+gdml+primaries 
        ##

    ckm-okg4()
    {
        OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey --embedded --save
    }

        ##  2nd executable : compute, fully instrumented gorilla  
        ##
        ##      OKG4Test picks up geocache+GDML+primaries from the key geocache 
        ##      and proceeds to bi-simulate in compute mode
        ##

    ckm-okg4-load()
    {
        OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --load --envkey --embedded
    }

        ##  2nd executable again : load+viz 
        ##  
        ##      OKG4Test (with no G4) just loads event + geocache for visualization 
        ##


Immediate Issues
-----------------

* Lots of bad flags : skipped some asserts in CRecorder/CPhoton to complete

  * FIXED :doc:`OKG4Test_CPhoton_badflag` was using default G4OpBoundaryProcess when need the custom one

* CPU side indexing aborted, fails to create the phosel + recsel for G4 

  * FIXED :doc:`OKG4Test_OpticksEvent_indexPhotonsCPU_assert` by direct recording whether dynamic 
    in OpticksEvent and using that to decide on resize via setNumPhotons

* no hits from G4 : SD-LV ASSOCIATION DIDNT SURVIVE THE CACHE/GDML ??

  * :doc:`OKG4Test_no_G4_hits`

* listed under torch, need a new "primaries" source code ? 


::

    epsilon:torch blyth$ np.py 
    /private/tmp/blyth/opticks/evt/g4live/torch
           ./Opticks.npy : (33, 1, 4) 

         ./-1/report.txt : 31 
           ./-1/idom.npy : (1, 1, 4) 
           ./-1/fdom.npy : (3, 1, 4) 
             ./-1/gs.npy : (5, 6, 4) 
             ./-1/no.npy : (152, 4, 4) 

             ./-1/rx.npy : (76, 10, 2, 4) 
             ./-1/ox.npy : (76, 4, 4) 
             ./-1/ph.npy : (76, 1, 2) 

    ./-1/20180819_214856/report.txt : 31 

          ./1/report.txt : 38 
            ./1/idom.npy : (1, 1, 4) 
            ./1/fdom.npy : (3, 1, 4) 
              ./1/gs.npy : (5, 6, 4) 
              ./1/no.npy : (152, 4, 4) 

              ./1/rx.npy : (76, 10, 2, 4) 
              ./1/ox.npy : (76, 4, 4) 
              ./1/ph.npy : (76, 1, 2) 

              ./1/ps.npy : (76, 1, 4) 
              ./1/rs.npy : (76, 10, 1, 4)     
              ./1/ht.npy : (9, 4, 4) 

    ./1/20180819_214856/report.txt : 38 
    epsilon:torch blyth$ 






New SourceCode for primaries needed ?::

    139 void CG4Ctx::initEvent(const OpticksEvent* evt)
    140 {
    141     _ok_event_init = true ;
    142     _photons_per_g4event = evt->getNumPhotonsPerG4Event() ;
    143     _steps_per_photon = evt->getMaxRec() ;
    144     _record_max = evt->getNumPhotons();   // from the genstep summation
    145     _bounce_max = evt->getBounceMax();
    146 
    147     const char* typ = evt->getTyp();
    148     _gen = OpticksFlags::SourceCode(typ);
    149     assert( _gen == TORCH || _gen == G4GUN  );
    150 
    151     LOG(info) << "CG4Ctx::initEvent"
    152               << " photons_per_g4event " << _photons_per_g4event
    153               << " steps_per_photon " << _steps_per_photon
    154               << " gen " << _gen
    155               ;
    156 }





