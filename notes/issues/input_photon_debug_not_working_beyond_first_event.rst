input_photon_debug_not_working_beyond_first_event
==================================================



issue
--------

Using input_photons beyond 1st event crashes, with garbled evt. 



review input photon mechanics
----------------------------------

Q: what creates the holder gensteps with input photons ? 


::

    epsilon:opticks blyth$ opticks-f MakeInputPhotonGenstep
    ./sysrap/SEvt.hh:    static quad6 MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr ); 
    ./sysrap/SEvt.cc:    MakeInputPhotonGenstep and m2w (model-2-world) 
    ./sysrap/SEvt.cc:        addGenstep(MakeInputPhotonGenstep(input_photon, frame)); 
    ./sysrap/SEvt.cc:SEvt::MakeInputPhotonGenstep
    ./sysrap/SEvt.cc:quad6 SEvt::MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr )
    epsilon:opticks blyth$ 



    0716 /**
     717 SEvt::MakeInputPhotonGenstep
     718 -----------------------------
     719 
     720 Now called from SEvt::addFrameGenstep (formerly from SEvt::setFrame)
     721 Note that the only thing taken from the *input_photon* is the 
     722 number of photons so this can work with either local or 
     723 transformed *input_photon*. 
     724 
     725 The m2w transform from the frame is copied into the genstep.  
     726 HMM: is that actually used ? Because the frame is also persisted. 
     727 
     728 **/
     729 
     730 quad6 SEvt::MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr )
     731 {
     732     LOG(LEVEL) << " input_photon " << NP::Brief(input_photon) ;
     733 
     734     quad6 ipgs ;
     735     ipgs.zero();
     736     ipgs.set_gentype( OpticksGenstep_INPUT_PHOTON );
     737     ipgs.set_numphoton(  input_photon->shape[0]  );
     738     fr.m2w.write(ipgs); // copy fr.m2w into ipgs.q2,q3,q4,q5 
     739     return ipgs ;
     740 }






    345 /**
     346 SEvt::getInputPhoton
     347 ---------------------
     348 
     349 Returns the transformed input photon if present. 
     350 For the transformed photons to  be present it is necessary to have called SEvt::setFrame
     351 That is done from on high by G4CXOpticks::setupFrame which gets invoked by G4CXOpticks::setGeometry
     352 
     353 The frame and corresponding transform used can be controlled by several envvars, 
     354 see CSGFoundry::getFrameE. Possible envvars include:
     355 
     356 +------------------------------+----------------------------+
     357 | envvar                       | Examples                   |
     358 +==============================+============================+
     359 | INST                         |                            |
     360 +------------------------------+----------------------------+
     361 | MOI                          | Hama:0:1000 NNVT:0:1000    |          
     362 +------------------------------+----------------------------+
     363 | OPTICKS_INPUT_PHOTON_FRAME   |                            |
     364 +------------------------------+----------------------------+
     365 
     366 
     367 **/
     368 NP* SEvt::getInputPhoton() const {  return input_photon_transformed ? input_photon_transformed : input_photon  ; }
     369 bool SEvt::hasInputPhotonTransformed() const { return input_photon_transformed != nullptr ; }
     370 
     371 
     372 /**
     373 SEvt::gatherInputPhoton
     374 -------------------------
     375 
     376 To avoid issues with inphoton and saving 2nd events, 
     377 treat the inphoton more like other arrays by having a distinct
     378 inphoton copy for each event. 
     379 
     380 **/
     381 
     382 NP* SEvt::gatherInputPhoton() const
     383 {
     384     NP* ip = getInputPhoton();
     385     NP* ipc = ip ? ip->copy() : nullptr ;
     386     return ipc ;
     387 }





::

    275 void QEvent::setInputPhoton()
    276 {   
    277     LOG(LEVEL);  
    278     input_photon = sev->gatherInputPhoton(); // makes a copy 
    279     checkInputPhoton(); 
    280     narrow_input_photon = input_photon->ebyte == 8 ? NP::MakeNarrow(input_photon) : input_photon ;
    281 
    282     /* 
    283     if( input_photon == nullptr ) 
    284     {
    285         NP* ip = sev->getInputPhoton() ; 
    286         input_photon = ip ? ip->copy() : nullptr ; 
    287         LOG(info) << " input_photon " << ( input_photon ? input_photon->sstr() : "-" ) ;  
    288         checkInputPhoton(); 
    289         narrow_input_photon = input_photon->ebyte == 8 ? NP::MakeNarrow(input_photon) : input_photon ; 
    290         LOG(info) << " narrow_input_photon " << ( narrow_input_photon ? narrow_input_photon->sstr() : "-" ) ;  
    291 
    292         // THIS KLUDGING : DONT MAKE MUCH SENSE
    293         // SEvt HOLDS ON TO THE INPUT_PHOTONS ALL THIS 
    294         // SHOULD DO IS CHECK THEM AND UPLOAD THEM 
    295 
    296     }
    297     */
    298 
    299     
    300     int numph = input_photon->shape[0] ;
    301     setNumPhoton( numph ); 
    302     QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)narrow_input_photon->bytes(), numph );
    303 }   


