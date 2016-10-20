Reemission Review
====================



seqhis machinery inconsistency between CRecorder and Rec
----------------------------------------------------------

::

    simon:geant4_opticks_integration blyth$ t tlaser-d
    tlaser-d () 
    { 
        tlaser-;
        tlaser-t --steppingdbg   ## dumps every event 
    }
    simon:geant4_opticks_integration blyth$ t tlaser-t
    tlaser-t () 
    { 
        tlaser-;
        tlaser-- --okg4 --compute $*
    }



CRecorder and Rec are disagreeing for the last slot at the 6 in 10k level. 
Presumably a truncation behavior difference::

    2016-10-20 11:23:58.951 INFO  [2770241] [OpticksEvent::collectPhotonHitsCPU@1924] OpticksEvent::collectPhotonHitsCPU numHits 13
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@397] CG4::postpropagate
     event_total 1
     track_total 10468
     step_total 51335
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@407]  seqhis_mismatch 6
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@421]  seqmat_mismatch 0
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@434]  debug_photon 6 (photon_id) 
        5235
        4221
        3186
        2766
        2766
         839
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@441] TO DEBUG THESE USE:  --dindex=5235,4221,3186,2766,2766,839
    2016-10-20 11:23:58.951 INFO  [2770241] [CG4::postpropagate@296] CG4::postpropagate(0) DONE



pushing out truncation, pushes out the problem 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    tlaser-t --dindex=4124,3285 --bouncemax 15 --recordmax 16 


    2016-10-20 15:27:35.934 INFO  [2830854] [CSteppingAction::report@412]  seqhis_mismatch 2
     rdr cccbcc0ccc9ccccd rec 5ccbcc0ccc9ccccd
     rdr cc6ccccacccccc5d rec 5c6ccccacccccc5d
    2016-10-20 15:27:35.934 INFO  [2830854] [CSteppingAction::report@426]  seqmat_mismatch 0
    2016-10-20 15:27:35.934 INFO  [2830854] [CSteppingAction::report@439]  debug_photon 2 (photon_id) 
        4124
        3285
    2016-10-20 15:27:35.934 INFO  [2830854] [CSteppingAction::report@446] TO DEBUG THESE USE:  --dindex=4124,3285


    tlaser-t --bouncemax 16 --recordmax 16 

    2016-10-20 15:59:31.210 INFO  [2839084] [CSteppingAction::report@412]  seqhis_mismatch 2
     rdr cccacccccc9ccccd rec 5ccacccccc9ccccd
     rdr cccc0b0ccccc6ccd rec 5ccc0b0ccccc6ccd
    2016-10-20 15:59:31.210 INFO  [2839084] [CSteppingAction::report@426]  seqmat_mismatch 0
    2016-10-20 15:59:31.210 INFO  [2839084] [CSteppingAction::report@439]  debug_photon 2 (photon_id) 
        7836
        5501



inkling 
~~~~~~~~~

Suspect the comparison if happening prior to the
rejoin being completed ... 




truncation control
~~~~~~~~~~~~~~~~~~~~

::

    409    char bouncemax[128];
    410    snprintf(bouncemax,128,
    411 "Maximum number of boundary bounces, 0:prevents any propagation leaving generated photons"
    412 "Default %d ", m_bouncemax);
    413    m_desc.add_options()
    414        ("bouncemax,b",  boost::program_options::value<int>(&m_bouncemax), bouncemax );
    415 
    416 
    417    // keeping bouncemax one less than recordmax is advantageous 
    418    // as bookeeping is then consistent between the photons and the records 
    419    // as this avoiding truncation of the records
    420 
    421    char recordmax[128];
    422    snprintf(recordmax,128,
    423 "Maximum number of photon step records per photon, 1:to minimize without breaking machinery. Default %d ", m_recordmax);
    424    m_desc.add_options()
    425        ("recordmax,r",  boost::program_options::value<int>(&m_recordmax), recordmax );
    426 




CRecorder m_seqhis 
~~~~~~~~~~~~~~~~~~

primarily from CRecorder::RecordStepPoint based on flag argument and current slot,
note that m_slot continues to increment well past the recording range. 

This means that local *slot* gets will continue to point to m_steps_per_photon - 1 


The mismatch happens prior to lastPost, so problem all from pre::


    488     if(!preSkip)
    489     {
    490        done = RecordStepPoint( pre, preFlag, preMat, m_prior_boundary_status, PRE );
    491     }
    492 
    493     if(lastPost && !done)
    494     {
    495        done = RecordStepPoint( post, postFlag, postMat, m_boundary_status, POST );
    496     }
    497 


Rec m_seqhis
~~~~~~~~~~~~~~~~

Rec::addFlagMaterial attemps to mimmick CRecorder recording based on m_slot and flag argument.
This is invoked based on saved states by Rec::sequence

Hmm the below will always end with POST even prior to lastPost or when truncated... 

::

    298     
    299     for(unsigned i=0 ; i < nstate; i++)
    300     {
    301         rc = getFlagMaterialStageDone(flag, material, stage, done, i, PRE );
    302         if(rc == OK)
    303             addFlagMaterial(flag, material) ;
    304     }
    305     
    306     rc = getFlagMaterialStageDone(flag, material, stage, done, nstate-1, POST );
    307     if(rc == OK)
    308         addFlagMaterial(flag, material) ;




How to proceed ?
------------------

* need to add DYB style reemission to CFG4 

First tack, teleport in the DsG4Scintillation code and try to get it to work::

    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.h .
    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc .
    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPhysConsOptical.h .



Adopting DYBOp into CFG4
---------------------------

Trying to passalong the primary index in CSteppingAction::setTrack
only works when one reem happens (ie there is at most one call to DsG4Scintillation::PostStepDoIt)
in between steps.  But there are often two such calls.. 

::

    208     if(m_optical)          
    209     {                      
    210          if(m_parent_id == -1) // track is a primary opticalphoton (ie not from reemission)
    211          {                 
    212              G4Track* mtrack = const_cast<G4Track*>(track);
    213 
    214              // m_primary_photon_id++ ;  // <-- starts at -1, thus giving zero-based index
    215              int primary_photon_id = m_track_id ;   // instead of minting new index, use track_id
    216 
    217              mtrack->SetParentID(primary_photon_id);      
    218 
    219              LOG(info) << "CSteppingAction::setTrack"
    220                        << " primary photon "
    221                        << " track_id " << m_track_id
    222                        << " parent_id " << m_parent_id
    223                        << " primary_photon_id " << primary_photon_id 
    224                        ;
    225 
    226          }   
    227          else
    228          {   
    229              LOG(info) << "CSteppingAction::setTrack"
    230                        << " 2ndary photon "
    231                        << " track_id " << m_track_id
    232                        << " parent_id " << m_parent_id << "<-primary" 
    233                        ;
    234          }
    235     }        
    236 }        




::

    2016-10-05 13:02:27.694 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 543 parent_id -1 primary_photon_id 543
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 542 parent_id -1 primary_photon_id 542
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 541 parent_id -1 primary_photon_id 541
    2016-10-05 13:02:27.695 INFO  [1902787] [*DsG4Scintillation::PostStepDoIt@771]  DsG4Scintillation reemit  psdi_index 49098 secondaryTime(ns) 2.57509 track_id 540 parent_id -1 scnt 2 nscnt 2
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 540 parent_id -1 primary_photon_id 540
    2016-10-05 13:02:27.695 INFO  [1902787] [*DsG4Scintillation::PostStepDoIt@771]  DsG4Scintillation reemit  psdi_index 49099 secondaryTime(ns) 2.66136 track_id 10440 parent_id 540 scnt 2 nscnt 2
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@229] CSteppingAction::setTrack 2ndary photon  track_id 10440 parent_id 540<-primary
    2016-10-05 13:02:27.695 WARN  [1902787] [OpPointFlag@266]  reaching...  NoProc
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@229] CSteppingAction::setTrack 2ndary photon  track_id 10441 parent_id 10440<-primary
    2016-10-05 13:02:27.695 WARN  [1902787] [OpPointFlag@266]  reaching...  NoProc
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 539 parent_id -1 primary_photon_id 539
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 538 parent_id -1 primary_photon_id 538


CRecorder and Rec are almost matching at 10k level : truncation difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* truncation difference for REJOIN into last slot 

::

    2016-10-05 20:42:04.769 INFO  [2023965] [CSteppingAction::report@383] CG4::postpropagate
     event_total 1
     track_total 10468
     step_total 51335
    2016-10-05 20:42:04.769 INFO  [2023965] [CSteppingAction::report@393]  seqhis_mismatch 6
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
    2016-10-05 20:42:04.769 INFO  [2023965] [CSteppingAction::report@407]  seqmat_mismatch 0




Hmm seems hijacking ParentID is not so easy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:geant4_10_02_p01 blyth$ find source -name '*.cc' -exec grep -H SetParentID {} \;
    source/error_propagation/src/G4ErrorPropagator.cc:  theG4Track->SetParentID(0);
    source/event/src/G4PrimaryTransformer.cc:    track->SetParentID(0);
    source/event/src/G4StackManager.cc:      aTrack->SetParentID(-1);
    source/processes/electromagnetic/dna/management/src/G4ITModelProcessor.cc:          GetIT(secondary)->SetParentID(trackA->GetTrackID(),
    source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:    tempSecondaryTrack->SetParentID(fpTrack->GetTrackID());
    source/processes/electromagnetic/dna/utils/src/G4DNAChemistryManager.cc:    H2OTrack -> SetParentID(theIncomingTrack->GetTrackID());
    source/processes/electromagnetic/dna/utils/src/G4DNAChemistryManager.cc:    e_aqTrack -> SetParentID(theIncomingTrack->GetTrackID());
    source/processes/electromagnetic/dna/utils/src/G4DNAChemistryManager.cc:    track -> SetParentID(parentID);
    source/processes/electromagnetic/dna/utils/src/G4DNAChemistryManager.cc:    track -> SetParentID(theIncomingTrack->GetTrackID());
    source/processes/electromagnetic/xrays/src/G4Cerenkov.cc:                aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    source/processes/electromagnetic/xrays/src/G4Scintillation.cc:                aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    source/processes/electromagnetic/xrays/src/G4VXTRenergyLoss.cc:      aSecondaryTrack->SetParentID( aTrack.GetTrackID() );
    source/processes/optical/src/G4OpWLS.cc:    aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    source/tracking/src/G4SteppingManager2.cc:         tempSecondaryTrack->SetParentID( fTrack->GetTrackID() );
    source/tracking/src/G4SteppingManager2.cc:         tempSecondaryTrack->SetParentID( fTrack->GetTrackID() );
    source/tracking/src/G4SteppingManager2.cc:            tempSecondaryTrack->SetParentID( fTrack->GetTrackID() );
    simon:geant4_10_02_p01 blyth$ 


attach primaryPhotonId ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generators create G4PrimaryVertex and add to G4Event::

    255 void CTorchSource::GeneratePrimaryVertex(G4Event *evt)
    256 {
    ...
    275     for (G4int i = 0; i < m_num; i++)
    276     {
    277         pp.position = m_posGen->GenerateOne();
    278         G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position,m_time);
    ...
    305         G4PrimaryParticle* particle = new G4PrimaryParticle(m_definition);
    ...
    ...
    379         vertex->SetPrimary(particle);
    380         evt->AddPrimaryVertex(vertex);
    ...
    384     }
    385 }


Searching for what happens to G4PrimaryVertex next reveals::

    //  g4-;g4-cls G4PrimaryTransformer

    041 // class description:
     42 //
     43 //  This class is exclusively used by G4EventManager for the conversion
     44 // from G4PrimaryVertex/G4PrimaryParticle to G4DynamicParticle/G4Track.
     45 //
     46 
     47 class G4PrimaryTransformer
     48 {

    115 void G4PrimaryTransformer::GenerateSingleTrack
    116      (G4PrimaryParticle* primaryParticle,
    117       G4double x0,G4double y0,G4double z0,G4double t0,G4double wv)
    118 {
    ...
    ...
    218     // Create G4Track object
    219     G4Track* track = new G4Track(DP,t0,G4ThreeVector(x0,y0,z0));
    220     // Set trackID and let primary particle know it
    221     trackID++;
    222     track->SetTrackID(trackID);
    223     primaryParticle->SetTrackID(trackID);
    224     // Set parentID to 0 as a primary particle
    225     track->SetParentID(0);
    226     // Set weight ( vertex weight * particle weight )
    227     track->SetWeight(wv*(primaryParticle->GetWeight()));
    228     // Store it to G4TrackVector
    229     TV.push_back( track );
    230 
    231   }
    232 }






flags borked, so flying blind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* lots of Undefined boundary status


tlaser-;tlaser-d;tlaser.py::

      A:seqhis_ana      1:laser 
              8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                  4d        0.055            553       [2 ] TO AB
          cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012            122       [4 ] TO BT BT AB
             8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                 45d        0.006             65       [3 ] TO RE AB
              4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
            8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                455d        0.003             34       [4 ] TO RE RE AB
          cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
             8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
           8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
               4cccd        0.003             25       [5 ] TO BT BT BT AB
          cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                 46d        0.002             21       [3 ] TO SC AB
          cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
            4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                           10000         1.00 
       B:seqhis_ana     -1:laser 
                   0        0.850           8498       [1 ] ?0?
                  4d        0.071            708       [2 ] TO AB
                   d        0.028            276       [1 ] TO
                400d        0.017            168       [4 ] TO ?0? ?0? AB
              40000d        0.009             92       [6 ] TO ?0? ?0? ?0? ?0? AB
                  6d        0.008             82       [2 ] TO SC
                600d        0.004             35       [4 ] TO ?0? ?0? SC
                 46d        0.003             26       [3 ] TO SC AB
              60000d        0.002             16       [6 ] TO ?0? ?0? ?0? ?0? SC
               4000d        0.002             15       [5 ] TO ?0? ?0? ?0? AB
          400000000d        0.002             15       [10] TO ?0? ?0? ?0? ?0? ?0? ?0? ?0? ?0? AB
                 40d        0.001             11       [3 ] TO ?0? AB
            4000000d        0.001              7       [8 ] TO ?0? ?0? ?0? ?0? ?0? ?0? AB
             400600d        0.001              6       [7 ] TO ?0? ?0? SC ?0? ?0? AB
               4006d        0.001              6       [5 ] TO SC ?0? ?0? AB
          600000000d        0.001              6       [10] TO ?0? ?0? ?0? ?0? ?0? ?0? ?0? ?0? SC
             400006d        0.000              4       [7 ] TO SC ?0? ?0? ?0? ?0? AB
                 66d        0.000              3       [3 ] TO SC SC
               6006d        0.000              3       [5 ] TO SC ?0? ?0? SC
               6000d        0.000              3       [5 ] TO ?0? ?0? ?0? SC
                           10000         1.00 

Regained flags with USE_CUSTOM_BOUNDARY flipping::

      A:seqhis_ana      1:laser 
              8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                  4d        0.055            553       [2 ] TO AB
          cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012            122       [4 ] TO BT BT AB
             8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                 45d        0.006             65       [3 ] TO RE AB
              4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
            8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                455d        0.003             34       [4 ] TO RE RE AB
          cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
             8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
           8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
               4cccd        0.003             25       [5 ] TO BT BT BT AB
          cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                 46d        0.002             21       [3 ] TO SC AB
          cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
            4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                           10000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.811           8110       [6 ] TO BT BT BT BT SA
                  4d        0.075            750       [2 ] TO AB
          cccc9ccccd        0.024            238       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.018            177       [7 ] TO SC BT BT BT BT SA
                4ccd        0.016            161       [4 ] TO BT BT AB
              4ccccd        0.010            101       [6 ] TO BT BT BT BT AB
             8cc6ccd        0.004             44       [7 ] TO BT BT SC BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
             89ccccd        0.003             27       [7 ] TO BT BT BT BT DR SA
                 46d        0.003             26       [3 ] TO SC AB
               4cccd        0.002             22       [5 ] TO BT BT BT AB
          cacccccc6d        0.002             22       [10] TO SC BT BT BT BT BT BT SR BT
            8ccccc6d        0.002             21       [8 ] TO SC BT BT BT BT BT SA
          cccccc6ccd        0.002             20       [10] TO BT BT SC BT BT BT BT BT BT
          cccc6ccccd        0.002             16       [10] TO BT BT BT BT SC BT BT BT BT
          ccbccccc6d        0.002             15       [10] TO SC BT BT BT BT BT BR BT BT
           4cc9ccccd        0.001             14       [9 ] TO BT BT BT BT DR BT BT AB
           cac0ccc6d        0.001             14       [9 ] TO SC BT BT BT ?0? BT SR BT
                 4cd        0.001             13       [3 ] TO BT AB
             49ccccd        0.001              9       [7 ] TO BT BT BT BT DR AB
                           10000         1.00 





live reemission photon counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STATIC buffer was expecting a certain number of photons, so currently truncates::

    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingAction@156] CSA (startEvent) event_id 9 event_total 9
    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100000 record_max 100000 STATIC 
    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100000 record_max 100000 STATIC 
    ...
    2016-10-04 11:49:42.529 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100495 record_max 100000 STATIC 
    2016-10-04 11:49:42.529 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100495 record_max 100000 STATIC 
    2016-10-04 11:49:42.532 INFO  [1669872] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1


Hmm, I wonder if all the "NOT RECORDING" are RE ?  Looks to be so


Normally with fabricated (as opposed to G4 live) gensteps, the number of photons is known ahead of time.

Reemission means cannot know photon counts ahead of time ?

* that statement is true only if you count reemits as new photons, Opticks does not do that
 
Contining the slot for reemiisions with G4 ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is necessary for easy comparisons between G4 and Opticks.

With Opticks a reemitted photon continues the lineage (buffer slot) 
of its predecessor but with G4 a fresh new particle is created ...  

Small scale less than 10k photon torch running (corresponding to a single G4 "subevt") 
looks like can effect a continuation of reemission photons using the parent_id.  

For over 10k, need to cope with finding parent "subevt" too to line up with the correct 
record number. Unless can be sure subevt dont handled in mixed order ?

::

    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     219 parent_id      -1 step_id    0 record_id     219 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     218 parent_id      -1 step_id    0 record_id     218 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     217 parent_id      -1 step_id    0 record_id     217 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     216 parent_id      -1 step_id    0 record_id     216 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     215 parent_id      -1 step_id    0 record_id     215 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [*DsG4Scintillation::PostStepDoIt@761] reemit secondaryTime(ns) 18.6468 parent_id 215
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] SC- photon_id   10454 parent_id     215 step_id    0 record_id   10454 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] -C- photon_id   10454 parent_id     215 step_id    1 record_id   10454 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] -C- photon_id   10454 parent_id     215 step_id    2 record_id   10454 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     214 parent_id      -1 step_id    0 record_id     214 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     213 parent_id      -1 step_id    0 record_id     213 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     212 parent_id      -1 step_id    0 record_id     212 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     211 parent_id      -1 step_id    0 record_id     211 record_max   10000 STATIC 
    2016-10-04 15:01:45.105 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     210 parent_id      -1 step_id    0 record_id     210 record_max   10000 STATIC 
    2016-10-04 15:01:45.105 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     209 parent_id      -1 step_id    0 record_id     209 record_max   10000 STATIC 
    2016-10-04 15:01:45.105 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     208 parent_id      -1 step_id    0 record_id     208 record_max   10000 STATIC 


will the reemit step always come immediately after its parent one...  note the reversed photon order
what about multiple reemissions 

otherwise need to record the slots for all photons in order to continue them ?

::

    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      21 parent_id      -1 step_id    0 record_id      21 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      20 parent_id      -1 step_id    0 record_id      20 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      19 parent_id      -1 step_id    0 record_id      19 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      18 parent_id      -1 step_id    0 record_id      18 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [*DsG4Scintillation::PostStepDoIt@761] reemit secondaryTime(ns) 1.48211 parent_id 17
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      17 parent_id      -1 step_id    0 record_id      17 record_max      50 event_id       0 pre     0.1 post 1.48211 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] SC- photon_id      50 parent_id      17 step_id    0 record_id      50 record_max      50 event_id       0 pre 1.48211 post 6.09097 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      16 parent_id      -1 step_id    0 record_id      16 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      15 parent_id      -1 step_id    0 record_id      15 record_max      50 event_id       0 pre     0.1 post 0.489073 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      14 parent_id      -1 step_id    0 record_id      14 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 



reemission continuation are difficult to implement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

G4 produces secondary reemission photon with large trackId, which then have task of
linking with the fixed set of photons, within the recording range. 

When the parent id of the 2ndary photon matches the last_photon_id 
is a simple RHOP and can just continue filling slots.

Similarly when grandparent id photon matches last_photon_id can
just continue.

::

    318     int last_photon_id = m_recorder->getPhotonId();
    319 
    320     RecStage_t stage = UNKNOWN ;
    321     if( parent_id == -1 )
    322     {
    323         stage = photon_id != last_photon_id  ? START : COLLECT ;
    324     }
    325     else if( parent_id >= 0 && parent_id == last_photon_id )
    326     {
    327         stage = RHOP ;
    328         photon_id = parent_id ;
    329     }
    330     else if( grandparent_id >= 0 && grandparent_id == last_photon_id )
    331     {
    332         stage = RJUMP ;
    333         photon_id = grandparent_id ;
    334     }
    335 
    336 
    337     m_recorder->setPhotonId(photon_id);
    338     m_recorder->setEventId(eid);
    339     m_recorder->setStepId(step_id);
    340     m_recorder->setParentId(parent_id);




* difficult to make the connection between the secondary and the parent/grandparent
  that the new photons are in lineage with

* how can avoid the AB ? and getting stuck in 


::


     A:seqhis_ana      1:laser 
              8ccccd        0.756            756       [6 ] TO BT BT BT BT SA
                  4d        0.063             63       [2 ] TO AB
          cccc9ccccd        0.026             26       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.021             21       [7 ] TO SC BT BT BT BT SA
             8cccc5d        0.012             12       [7 ] TO RE BT BT BT BT SA
                4ccd        0.011             11       [4 ] TO BT BT AB
              4ccccd        0.007              7       [6 ] TO BT BT BT BT AB
                 45d        0.005              5       [3 ] TO RE AB
           8cccc555d        0.005              5       [9 ] TO RE RE RE BT BT BT BT SA
             8cc6ccd        0.005              5       [7 ] TO BT BT SC BT BT SA
            4ccccc5d        0.005              5       [8 ] TO RE BT BT BT BT BT AB
            8cccc55d        0.005              5       [8 ] TO RE RE BT BT BT BT SA
                 4cd        0.003              3       [3 ] TO BT AB
                455d        0.003              3       [4 ] TO RE RE AB
             86ccccd        0.003              3       [7 ] TO BT BT BT BT SC SA
            4ccccc6d        0.003              3       [8 ] TO SC BT BT BT BT BT AB
            8cc55ccd        0.003              3       [8 ] TO BT BT RE RE BT BT SA
          cccccc6ccd        0.003              3       [10] TO BT BT SC BT BT BT BT BT BT
          cccc55555d        0.003              3       [10] TO RE RE RE RE RE BT BT BT BT
          ccc9cccc6d        0.002              2       [10] TO SC BT BT BT BT DR BT BT BT
                            1000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.817            817       [6 ] TO BT BT BT BT SA
                  4d        0.060             60       [2 ] TO AB
          cccc9ccccd        0.024             24       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.009              9       [7 ] TO SC BT BT BT BT SA
                4ccd        0.007              7       [4 ] TO BT BT AB
              45454d        0.005              5       [6 ] TO AB RE AB RE AB   
              4ccccd        0.005              5       [6 ] TO BT BT BT BT AB
          cccccc6ccd        0.005              5       [10] TO BT BT SC BT BT BT BT BT BT
            8ccccc6d        0.003              3       [8 ] TO SC BT BT BT BT BT SA
            8cccc54d        0.003              3       [8 ] TO AB RE BT BT BT BT SA
           ccc9ccccd        0.003              3       [9 ] TO BT BT BT BT DR BT BT BT
          8cccc5454d        0.003              3       [10] TO AB RE AB RE BT BT BT BT SA
               4cccd        0.003              3       [5 ] TO BT BT BT AB
                 46d        0.003              3       [3 ] TO SC AB
             86ccccd        0.003              3       [7 ] TO BT BT BT BT SC SA
             8cc6ccd        0.003              3       [7 ] TO BT BT SC BT BT SA
           8cccc654d        0.002              2       [9 ] TO AB RE SC BT BT BT BT SA
          8cbccccc6d        0.002              2       [10] TO SC BT BT BT BT BT BR BT SA
             8ccc6cd        0.002              2       [7 ] TO BT SC BT BT BT SA
          cacccccc6d        0.002              2       [10] TO SC BT BT BT BT BT BT SR BT
                            1000         1.00 


Must less RE in CG4 ? Scrubbing the AB by going back one slot and replace with RE::

       A:seqhis_ana      1:laser 
              8ccccd        0.764         763501       [6 ] TO BT BT BT BT SA
                  4d        0.056          55825       [2 ] TO AB
          cccc9ccccd        0.025          25263       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.020          19707       [7 ] TO SC BT BT BT BT SA
                4ccd        0.013          12576       [4 ] TO BT BT AB
             8cccc5d        0.011          11183       [7 ] TO RE BT BT BT BT SA
              4ccccd        0.009           8554       [6 ] TO BT BT BT BT AB
                 45d        0.008           7531       [3 ] TO RE AB
            8cccc55d        0.005           5362       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004           4109       [7 ] TO BT BT SC BT BT SA
                455d        0.004           3588       [4 ] TO RE RE AB
             86ccccd        0.003           2836       [7 ] TO BT BT BT BT SC SA
          cccccc6ccd        0.003           2674       [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d        0.003           2524       [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd        0.002           2359       [7 ] TO BT BT RE BT BT SA
          cacccccc6d        0.002           2210       [10] TO SC BT BT BT BT BT BT SR BT
                 46d        0.002           2118       [3 ] TO SC AB
          cccc6ccccd        0.002           2060       [10] TO BT BT BT BT SC BT BT BT BT
               4cccd        0.002           1940       [5 ] TO BT BT BT AB
             89ccccd        0.002           1880       [7 ] TO BT BT BT BT DR SA
                         1000000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.814         813976       [6 ] TO BT BT BT BT SA
                  4d        0.048          48056       [2 ] TO AB
          cccc9ccccd        0.026          26149       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019          18604       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012          11614       [4 ] TO BT BT AB
                 8cd        0.010          10193       [3 ] TO BT SA
              4ccccd        0.009           8755       [6 ] TO BT BT BT BT AB
             8cc6ccd        0.004           4157       [7 ] TO BT BT SC BT BT SA
                  8d        0.004           3614       [2 ] TO SA
               8cccd        0.003           2746       [5 ] TO BT BT BT SA
             86ccccd        0.003           2696       [7 ] TO BT BT BT BT SC SA
                8c5d        0.002           2454       [4 ] TO RE BT SA
                455d        0.002           2354       [4 ] TO RE RE AB
                 45d        0.002           2306       [3 ] TO RE AB
               4cccd        0.002           2244       [5 ] TO BT BT BT AB
             89ccccd        0.002           2241       [7 ] TO BT BT BT BT DR SA
          cacccccc6d        0.002           2172       [10] TO SC BT BT BT BT BT BT SR BT
                 4cd        0.002           1967       [3 ] TO BT AB
          cccccc6ccd        0.002           1931       [10] TO BT BT SC BT BT BT BT BT BT
            8ccccc6d        0.002           1787       [8 ] TO SC BT BT BT BT BT SA
                         1000000         1.00 



REEMISSIONPROB is not a standard G4 property
----------------------------------------------

::

       +X horizontal tlaser from middle of DYB AD

       A: opticks, has reemission treatment aiming to match DYB NuWa DetSim 
                   (it is handled as a subset of BULK_ABSORB that confers rebirth)

       B: almost stock Geant4 10.2, no reemission treatment -> hence more absorption
                   (stock G4 is just absorbing, and the REEMISSIONPROB is ignored)


       A:seqhis_ana      1:laser 
              8ccccd        0.764         763501       [6 ] TO BT BT BT BT SA
                  4d        0.056          55825       [2 ] TO AB
          cccc9ccccd        0.025          25263       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.020          19707       [7 ] TO SC BT BT BT BT SA
                4ccd        0.013          12576       [4 ] TO BT BT AB
             8cccc5d        0.011          11183       [7 ] TO RE BT BT BT BT SA
              4ccccd        0.009           8554       [6 ] TO BT BT BT BT AB
                 45d        0.008           7531       [3 ] TO RE AB
            8cccc55d        0.005           5362       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004           4109       [7 ] TO BT BT SC BT BT SA
                455d        0.004           3588       [4 ] TO RE RE AB
             86ccccd        0.003           2836       [7 ] TO BT BT BT BT SC SA
          cccccc6ccd        0.003           2674       [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d        0.003           2524       [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd        0.002           2359       [7 ] TO BT BT RE BT BT SA
          cacccccc6d        0.002           2210       [10] TO SC BT BT BT BT BT BT SR BT
                 46d        0.002           2118       [3 ] TO SC AB
          cccc6ccccd        0.002           2060       [10] TO BT BT BT BT SC BT BT BT BT
               4cccd        0.002           1940       [5 ] TO BT BT BT AB
             89ccccd        0.002           1880       [7 ] TO BT BT BT BT DR SA
                         1000000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.813         813472       [6 ] TO BT BT BT BT SA
                  4d        0.072          71523       [2 ] TO AB
          cccc9ccccd        0.027          27170       [10] TO BT BT BT BT DR BT BT BT BT
                4ccd        0.017          17386       [4 ] TO BT BT AB
             8cccc6d        0.015          15107       [7 ] TO SC BT BT BT BT SA
              4ccccd        0.009           8842       [6 ] TO BT BT BT BT AB
          cacccccc6d        0.004           3577       [10] TO SC BT BT BT BT BT BT SR BT
             8cc6ccd        0.003           3466       [7 ] TO BT BT SC BT BT SA
                 46d        0.003           2515       [3 ] TO SC AB
             86ccccd        0.002           2476       [7 ] TO BT BT BT BT SC SA
           cac0ccc6d        0.002           2356       [9 ] TO SC BT BT BT ?0? BT SR BT
          cccccc6ccd        0.002           2157       [10] TO BT BT SC BT BT BT BT BT BT
             89ccccd        0.002           2127       [7 ] TO BT BT BT BT DR SA
               4cccd        0.002           1977       [5 ] TO BT BT BT AB
          cccc6ccccd        0.002           1949       [10] TO BT BT BT BT SC BT BT BT BT
            8ccccc6d        0.002           1515       [8 ] TO SC BT BT BT BT BT SA
          ccbccccc6d        0.001           1429       [10] TO SC BT BT BT BT BT BR BT BT
           4cc9ccccd        0.001           1215       [9 ] TO BT BT BT BT DR BT BT AB
                 4cd        0.001           1077       [3 ] TO BT AB
               4cc6d        0.001            802       [5 ] TO SC BT BT AB
                         1000000         1.00 



/usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.h::

    /// NB unlike stock G4  DsG4Scintillation::IsApplicable is true for opticalphoton
    ///    --> this is needed in order to handle the reemission of optical photons

    300 inline
    301 G4bool DsG4Scintillation::IsApplicable(const G4ParticleDefinition& aParticleType)
    302 {
    303         if (aParticleType.GetParticleName() == "opticalphoton"){
    304            return true;
    305         } else {
    306            return true;
    307         }
    308 }

    ///    NB the untrue comment, presumably inherited from stock G4 
    ///
    137         G4bool IsApplicable(const G4ParticleDefinition& aParticleType);
    138         // Returns true -> 'is applicable', for any particle type except
    139         // for an 'opticalphoton' 



/usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc::

    099 DsG4Scintillation::DsG4Scintillation(const G4String& processName,
    100                                      G4ProcessType type)
    101     : G4VRestDiscreteProcess(processName, type)
    102     , doReemission(true)
    103     , doBothProcess(true)
    104     , fPhotonWeight(1.0)
    105     , fApplyPreQE(false)
    106     , fPreQE(1.)
    107     , m_noop(false)
    108 {
    109     SetProcessSubType(fScintillation);
    110     fTrackSecondariesFirst = false;



    170 G4VParticleChange*
    171 DsG4Scintillation::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    172 
    173 // This routine is called for each tracking step of a charged particle
    174 // in a scintillator. A Poisson/Gauss-distributed number of photons is 
    175 // generated according to the scintillation yield formula, distributed 
    176 // evenly along the track segment and uniformly into 4pi.
    177 
    178 {
    179     aParticleChange.Initialize(aTrack);
    ...
    187     G4String pname="";
    188     G4ThreeVector vertpos;
    189     G4double vertenergy=0.0;
    190     G4double reem_d=0.0;
    191     G4bool flagReemission= false;

    193     if (aTrack.GetDefinition() == G4OpticalPhoton::OpticalPhoton()) 
            {
    194         G4Track *track=aStep.GetTrack();
    197 
    198         const G4VProcess* process = track->GetCreatorProcess();
    199         if(process) pname = process->GetProcessName();

    ///         flagReemission is set only for opticalphotons that are 
    ///         about to be bulk absorbed(fStopAndKill and !fGeomBoundary)
    ///
    ///           doBothProcess = true :  reemission for optical photons generated by both scintillation and Cerenkov processes         
    ///           doBothProcess = false : reemission for optical photons generated by Cerenkov process only 
    ///

    200 
    204         if(doBothProcess) 
               {
    205             flagReemission= doReemission
    206                 && aTrack.GetTrackStatus() == fStopAndKill
    207                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary;
    208         }
    209         else
                {
    210             flagReemission= doReemission
    211                 && aTrack.GetTrackStatus() == fStopAndKill
    212                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary
    213                 && pname=="Cerenkov";
    214         }
    218         if (!flagReemission) 
                {
    ///          -> give up the ghost and get absorbed
    219              return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    220         }
    221     }
    223     G4double TotalEnergyDeposit = aStep.GetTotalEnergyDeposit();
    228     if (TotalEnergyDeposit <= 0.0 && !flagReemission) {
    229         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    230     }
    ...
    246     if (aParticleName == "opticalphoton") {
    247       FastTimeConstant = "ReemissionFASTTIMECONSTANT";
    248       SlowTimeConstant = "ReemissionSLOWTIMECONSTANT";
    249       strYieldRatio = "ReemissionYIELDRATIO";
    250     }
    251     else if(aParticleName == "gamma" || aParticleName == "e+" || aParticleName == "e-") {
    252       FastTimeConstant = "GammaFASTTIMECONSTANT";
    ...
            }

    273     const G4MaterialPropertyVector* Fast_Intensity  = aMaterialPropertiesTable->GetProperty("FASTCOMPONENT");
    275     const G4MaterialPropertyVector* Slow_Intensity  = aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");
    277     const G4MaterialPropertyVector* Reemission_Prob = aMaterialPropertiesTable->GetProperty("REEMISSIONPROB");
    ...
    283     if (!Fast_Intensity && !Slow_Intensity )
    284         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    ...
    286     G4int nscnt = 1;
    287     if (Fast_Intensity && Slow_Intensity) nscnt = 2;
    ...
    291     G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
    292     G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
    293 
    294     G4ThreeVector x0 = pPreStepPoint->GetPosition();
    295     G4ThreeVector p0 = aStep.GetDeltaPosition().unit();
    296     G4double      t0 = pPreStepPoint->GetGlobalTime();
    297 
    298     //Replace NumPhotons by NumTracks
    299     G4int NumTracks=0;
    300     G4double weight=1.0;
    301     if (flagReemission) 
            {
    ...
    305         if ( Reemission_Prob == 0) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    307         G4double p_reemission= Reemission_Prob->GetProperty(aTrack.GetKineticEnergy());
    309         if (G4UniformRand() >= p_reemission) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    ////
    ////        above line reemission has a chance to not happen, otherwise we create a single secondary...
    ///         conferring reemission "rebirth"
    ////

    311         NumTracks= 1;
    312         weight= aTrack.GetWeight();
    316     else {
    317         //////////////////////////////////// Birks' law ////////////////////////





