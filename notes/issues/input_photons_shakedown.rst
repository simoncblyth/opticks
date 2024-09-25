input_photons_shakedown.rst
==============================


Questions
-----------

1. can follow the Opticks side of input photons getting to GPU, but what about Geant4 ? 

   * its done in various ways depending on access to G4Event 


Scripts
----------

Scripts to generate and plot input photons, run during installation only (unless add new types)::

    ~/o/ana/input_photons.sh
    ~/o/ana/input_photons_plt.sh


Envvars
----------


OPTICKS_RUNNING_MODE
   pick between SRM_DEFAULT/TORCH/INPUT_PHOTON/INPUT_GENTEP/GUN running 
   
   Q: what more specifically does this do ? 

OPTICKS_INPUT_PHOTON
    pick the array of input photons to use, eg RainXZ_Z230_10k_f8.npy
 
OPTICKS_INPUT_PHOTON_FRAME
    pick the frame in which the input photons should be introduced, 
    eg using MOI style specification "NNVT:0:1000"




Sysrap sources
----------------

::

    P[blyth@localhost sysrap]$ grep -l INPUT_PHOTON *.*
    OpticksGenstep.h
    SEvent.cc
    SEventConfig.cc
    SEventConfig.hh
    SEvt.cc
    SEvt.hh
    SGenerate.h
    SRM.h
    P[blyth@localhost sysrap]$ 



SEventConfig
~~~~~~~~~~~~~~

::

    178     static const char* InputPhoton();
    179     static const char* InputPhotonFrame();


SEvt.cc
~~~~~~~~~

::

     311 const char* SEvt::INPUT_GENSTEP_DIR = spath::Resolve("${SEvt__INPUT_GENSTEP_DIR:-$HOME/.opticks/InputGensteps}") ;
     312 const char* SEvt::INPUT_PHOTON_DIR = spath::Resolve("${SEvt__INPUT_PHOTON_DIR:-$HOME/.opticks/InputPhotons}") ;


     372 /**
     373 SEvt::LoadInputPhoton
     374 ----------------------
     375 
     376 This is invoked by SEvt::initInputPhoton which is invoked by SEvt::init at instanciation.
     377 
     378 Resolving the input string to a path is done in one of two ways:
     379 
     380 1. if the string starts with a letter A-Za-z eg "inphoton.npy" or "RandomSpherical10.npy" 
     381    it is assumed to be the name of a .npy file within the default SEvt__INPUT_PHOTON_DIR 
     382    of $HOME/.opticks/InputPhotons. 
     383 
     384    Create such files with ana/input_photons.py  
     385 
     386 2. if the string does not start with a letter eg /path/to/some/dir/file.npy or $TOKEN/path/to/file.npy 
     387    it is passed unchanged to  spath::Resolve
     388 
     389 **/
     390 
     391 NP* SEvt::LoadInputPhoton() // static 
     392 {
     393     const char* spec = SEventConfig::InputPhoton();
     394     return spec ? LoadInputPhoton(spec) : nullptr ;
     395 }
     396 NP* SEvt::LoadInputPhoton(const char* spec)
     397 {
     398     const char* path = ResolveInputArray( spec, INPUT_PHOTON_DIR );
     399     NP* a = LoadInputArray(path);
     400     assert( a->has_shape(-1,4,4) );
     401     return a ;
     402 }


     435 /**
     436 SEvt::initInputPhoton
     437 -----------------------
     438 
     439 This is invoked by SEvt::init on instanciating the SEvt instance  
     440 The default "SEventConfig::InputPhoton()" is nullptr meaning no input photons.
     441 This can be changed by setting an envvar in the script that runs the executable, eg::
     442 
     443    export OPTICKS_INPUT_PHOTON=CubeCorners.npy
     444    export OPTICKS_INPUT_PHOTON=$HOME/somedir/path/to/inphoton.npy
     445  
     446 Or within the code of the executable, typically in the main prior to SEvt instanciation, 
     447 using eg::
     448 
     449    SEventConfig::SetInputPhoton("CubeCorners.npy")
     450    SEventConfig::SetInputPhoton("$HOME/somedir/path/to/inphoton.npy")
     451 
     452 When non-null it is resolved into a path and the array loaded at SEvt instanciation
     453 by SEvt::LoadInputPhoton
     454 
     455 **/
     456 
     457 void SEvt::initInputPhoton()
     458 {
     459     NP* ip = LoadInputPhoton() ;
     460     setInputPhoton(ip);
     461 }
     462 
     463 void SEvt::setInputPhoton(NP* p)
     464 {
     465     if(p == nullptr) return ;
     466     input_photon = p ;
     467     bool input_photon_expect = input_photon->has_shape(-1,4,4) ;
     468     if(!input_photon_expect) std::raise(SIGINT) ;
     469     assert( input_photon_expect );
     470 
     471     int numphoton = input_photon->shape[0] ;
     472     bool numphoton_expect = numphoton > 0 ;
     473     if(!numphoton_expect) std::raise(SIGINT) ;
     474     assert( numphoton_expect  );
     475 }
     476 

     482 /**
     483 SEvt::getInputPhoton_
     484 ----------------------
     485 
     486 This variant always provides the untransformed input photons.
     487 That will be nullptr unless OPTICKS_INPUT_PHOTON is defined. 
     488 
     489 **/
     490 NP* SEvt::getInputPhoton_() const { return input_photon ; }
     491 bool SEvt::hasInputPhoton() const { return input_photon != nullptr ; }
     492 


     494 /**
     495 SEvt::getInputPhoton
     496 ---------------------
     497 
     498 Returns the transformed input photon if present. 
     499 For the transformed photons to  be present it is necessary to have called SEvt::setFrame
     500 That is done from on high by G4CXOpticks::setupFrame which gets invoked by G4CXOpticks::setGeometry
     501 
     502 The frame and corresponding transform used can be controlled by several envvars, 
     503 see CSGFoundry::getFrameE. Possible envvars include:
     504 
     505 +------------------------------+----------------------------+
     506 | envvar                       | Examples                   |
     507 +==============================+============================+
     508 | INST                         |                            |
     509 +------------------------------+----------------------------+
     510 | MOI                          | Hama:0:1000 NNVT:0:1000    |          
     511 +------------------------------+----------------------------+
     512 | OPTICKS_INPUT_PHOTON_FRAME   |                            |
     513 +------------------------------+----------------------------+
     514 
     515 
     516 **/
     517 NP* SEvt::getInputPhoton() const {  return input_photon_transformed ? input_photon_transformed : input_photon  ; }
     518 bool SEvt::hasInputPhotonTransformed() const { return input_photon_transformed != nullptr ; }
     519 


     634 /**
     635 SEvt::setFrame
     636 ------------------
     637 
     638 As it is necessary to have the geometry to provide the frame this 
     639 is now split from eg initInputPhotons.  
     640 
     641 **simtrace running**
     642     MakeCenterExtentGensteps based on the given frame. 
     643 
     644 **simulate inputphoton running**
     645     MakeInputPhotonGenstep and m2w (model-2-world) 
     646     transforms the photons using the frame transform
     647 
     648 Formerly(?) for simtrace and input photon running with or without a transform 
     649 it was necessary to call this for every event due to the former call to addInputGenstep, 
     650 but now that the genstep setup is moved to SEvt::beginOfEvent it is only needed 
     651 to call this for each frame, usually once only. 
     652 
     653 **/
     654 
     655 
     656 void SEvt::setFrame(const sframe& fr )
     657 {
     658     frame = fr ;
     659     transformInputPhoton();
     660 }


     672 /**
     673 SEvt::transformInputPhoton
     674 ---------------------------
     675 
     676 **/
     677 
     678 void SEvt::transformInputPhoton()
     679 {
     680     bool proceed = SEventConfig::IsRGModeSimulate() && hasInputPhoton() ;
     681     LOG(LEVEL) << " proceed " << ( proceed ? "YES" : "NO " ) ;
     682     if(!proceed) return ;
     683 
     684     bool normalize = true ;  // normalize mom and pol after doing the transform 
     685 
     686     NP* ipt = frame.transform_photon_m2w( input_photon, normalize );
     687 
     688     if(transformInputPhoton_WIDE)  // see notes/issues/G4ParticleChange_CheckIt_warnings.rst
     689     {
     690         input_photon_transformed = ipt ;
     691     }
     692     else
     693     {
     694         input_photon_transformed = ipt->ebyte == 8 ? NP::MakeNarrow(ipt) : ipt ;
     695         // narrow here to prevent immediate A:B difference with Geant4 seeing double precision 
     696         // and Opticks float precision 
     697     }
     698 }


QEvent
~~~~~~~

::

     401 void QEvent::setInputPhoton()
     402 {
     403     LOG_IF(info, LIFECYCLE) ;
     404     LOG(LEVEL);
     405     input_photon = sev->gatherInputPhoton();
     406     checkInputPhoton();
     407 
     408     int numph = input_photon->shape[0] ;
     409     setNumPhoton( numph );
     410     QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)input_photon->bytes(), numph );
     411 
     412     // HMM: there is a getter ... 
     413     //delete input_photon ; 
     414     //input_photon = nullptr ;  
     415 }
     416 





CSG/CSGFoundry.cc
~~~~~~~~~~~~~~~~~~~~


::

    3553 CSGFoundry::getFrameE
    3554 -----------------------
    3555 
    3556 The frame and corresponding transform used can be controlled by several envvars, 
    3557 see CSGFoundry::getFrameE. Possible envvars include:
    3558 
    3559 +------------------------------+----------------------------+
    3560 | envvar                       | Examples                   |
    3561 +==============================+============================+
    3562 | INST                         |                            |
    3563 +------------------------------+----------------------------+
    3564 | MOI                          | Hama:0:1000 NNVT:0:1000    |          
    3565 +------------------------------+----------------------------+
    3566 | OPTICKS_INPUT_PHOTON_FRAME   |                            |
    3567 +------------------------------+----------------------------+
    3568 
    3569 
    3570 The sframe::set_ekv records into frame metadata the envvar key and value 
    3571 that picked the frame. 
    3572 






Geant4 handling of input photons ? 
----------------------------------------

G4CXApp.h which is used from the raindrop example uses
U4VPrimaryGenerator::GeneratePrimaries_From_Photons profiting 
from direct access to G4Event::

    219 void G4CXApp::GeneratePrimaries(G4Event* event)
    220 {
    221     G4int eventID = event->GetEventID();
    222 
    223     LOG(LEVEL) << "[ SEventConfig::RunningModeLabel " << SEventConfig::RunningModeLabel() << " eventID " << eventID ;
    224     SEvt* sev = SEvt::Get_ECPU();
    225     assert(sev);
    226 
    227     if(SEventConfig::IsRunningModeGun())
    228     {
    229         LOG(fatal) << " THIS MODE NEEDS WORK ON U4PHYSICS " ;
    230         std::raise(SIGINT);
    231         fGun->GeneratePrimaryVertex(event) ;
    232     }
    233     else if(SEventConfig::IsRunningModeTorch())
    234     {
    235         int idx_arg = eventID ;
    236         NP* gs = SEvent::MakeTorchGenstep(idx_arg) ;
    237         NP* ph = SGenerate::GeneratePhotons(gs);
    238         U4VPrimaryGenerator::GeneratePrimaries_From_Photons(event, ph);
    239         delete ph ;
    240 
    241         SEvent::SetGENSTEP(gs);  // picked up by 
    242     }
    243     else if(SEventConfig::IsRunningModeInputPhoton())
    244     {
    245         NP* ph = sev->getInputPhoton();
    246         U4VPrimaryGenerator::GeneratePrimaries_From_Photons(event, ph) ;
    247     }
    248     else if(SEventConfig::IsRunningModeInputGenstep())
    249     {
    250         LOG(fatal) << "General InputGensteps with Geant4 not implemented, use eg cxs_min.sh to do that with Opticks " ;
    251         std::raise(SIGINT);
    252     }
    253     LOG(LEVEL) << "] " << " eventID " << eventID  ;
    254 }




JUNOSW+Opticks and InputPhotons only with special "opticks" arg configuring use of Simulation/GenTools/src/GtOpticksTool.cc
---------------------------------------------------------------------------------------------------------------------------------


Within JUNOSW+Opticks there is no direct access to G4Event have to 
go via the mutate interface and HepMC::GenEvent::

    189 bool GtOpticksTool::mutate(HepMC::GenEvent& event)

::

    P[blyth@localhost junosw]$ find . -name '*.cc' -exec grep -H InputPhoton {} \;
    ./Simulation/GenTools/src/GtOpticksTool.cc:    ret = SEvt::HasInputPhoton() ; 
    ./Simulation/GenTools/src/GtOpticksTool.cc:       << SEvt::DescHasInputPhoton()
    ./Simulation/GenTools/src/GtOpticksTool.cc:GtOpticksTool::getInputPhoton
    ./Simulation/GenTools/src/GtOpticksTool.cc:NP* GtOpticksTool::getInputPhoton() const { return m_input_photon ;  }
    ./Simulation/GenTools/src/GtOpticksTool.cc:        m_input_photon = SEvt::GetInputPhoton() ; 
    ./Simulation/GenTools/src/GtOpticksTool.cc:            << " deferred SEvt::GetInputPhoton "
    P[blyth@localhost junosw]$ 


::

    099 /**
    100 GtOpticksTool::getInputPhoton
    101 -------------------------------
    102 
    103 The m_input_photon is set by GtOpticksTool::mutate 
    104 as it is too soon at initialization time because the frame targetting 
    105 requires an Opticks CSGFoundry geometry. 
    106 
    107 **/
    108 NP* GtOpticksTool::getInputPhoton() const { return m_input_photon ;  }
    109 


    188 #ifdef WITH_G4CXOPTICKS
    189 bool GtOpticksTool::mutate(HepMC::GenEvent& event)
    190 {
    191     int event_number = event.event_number() ; // is this 0-based ? 
    192     if(m_input_photon == nullptr)
    193     {
    194         m_input_photon = SEvt::GetInputPhoton() ;
    195         std::cerr
    196             << "GtOpticksTool::mutate"
    197             << " event_number " << event_number
    198             << " deferred SEvt::GetInputPhoton "
    199             << " " << SEvt::Brief()
    200             << " m_input_photon " << ( m_input_photon ? m_input_photon->sstr() : "-" )
    201             << std::endl
    202             ;
    203     }
    204 
    205     int numPhotons = m_input_photon ? m_input_photon->shape[0] : 0 ;
    206     //LOG(info)
    207     std::cerr
    208         << "GtOpticksTool::mutate"
    209         << " event_number " << event_number
    210         << " numPhotons " << numPhotons
    211         << std::endl
    212         ;
    213 
    214     for(int idx = 0; idx < numPhotons ; ++idx) add_optical_photon(event, idx);
    215     return true;
    216 }
    217 #else


    1727 NP* SEvt::GetInputPhoton(int idx) {  return Exists(idx) ? Get(idx)->getInputPhoton() : nullptr ; }
    1728 
    1729 void SEvt::SetInputPhoton(NP* p)
    1730 {
    1731     if(Exists(0)) Get(0)->setInputPhoton(p) ;
    1732     if(Exists(1)) Get(1)->setInputPhoton(p) ;
    1733 }
    1734 bool SEvt::HasInputPhoton(int idx)
    1735 {
    1736     return Exists(idx) ? Get(idx)->hasInputPhoton() : false ;
    1737 }
    1738 bool SEvt::HasInputPhoton()
    1739 {
    1740     return HasInputPhoton(EGPU) || HasInputPhoton(ECPU) ;
    1741 }
    1742 
    1743 NP* SEvt::GetInputPhoton() // static 
    1744 {
    1745     NP* ip = nullptr ;
    1746     if(ip == nullptr && HasInputPhoton(EGPU)) ip = GetInputPhoton(EGPU) ;
    1747     if(ip == nullptr && HasInputPhoton(ECPU)) ip = GetInputPhoton(ECPU) ;
    1748     return ip ;
    1749 }



    517 NP* SEvt::getInputPhoton() const {  return input_photon_transformed ? input_photon_transformed : input_photon  ; }




Issue 1 : after switching on input photon in jok-tds get fail
------------------------------------------------------------------

Switch on input photons with::

    119 jok-tds-input-photon()
    120 {
    121     type $FUNCNAME
    122     export OPTICKS_RUNNING_MODE=SRM_INPUT_PHOTON
    123     export OPTICKS_INPUT_PHOTON=RainXZ_Z230_10k_f8.npy
    124     export OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000
    125 }

    248    #local gun=1    ## long time defalt is the base "gun"
    249    local gun=0     ## tryout input photons
    250 
    251    local GUN=${GUN:-$gun}
    252    case $GUN in
    253      0) trgs="$trgs opticks" ;;
    254      1) trgs="$trgs $gun1" ;;
    255      2) trgs="$trgs $gun2"  ;;
    256      3) trgs="$trgs $gun3"  ;;
    257    esac
    258 
    259    if [ "$GUN" == "0" ]; then
    260        jok-tds-input-photon
    261    fi


::

    jok-tds-gdb
    ...

    junotoptask:DetSim0Svc.dumpOpticks  INFO: DetSim0Svc::initializeOpticks m_opticksMode 1 WITH_G4CXOPTICKS 
    junotoptask:DetSim0Svc.initialize  INFO: Register AnaMgr FixLightVelAnaMgr
    junotoptask:SniperProfiling.initialize  INFO: 
    GtOpticksTool::configure WITH_G4CXOPTICKS : ERROR : something is missing 
    SEvt::DescHasInputPhoton()   SEventConfig::IntegrationMode 1 SEvt::HasInputPhoton(EGPU) 0 SEvt::HasInputPhoton(ECPU) 0
    SEvt::Brief
    SEvt::Brief  SEvt::Exists(0) N SEvt::Exists(1) N
     SEvt::Get(0)->brief() -
     SEvt::Get(1)->brief() -

    SEvt::DescInputPhoton(EGPU)-SEvt::DescInputPhoton(ECPU)-


    GtOpticksTool::configure_FAIL_NOTES
    =====================================

    GtOpticksTool integrates junosw with Opticks input photon 
    machinery including the frame targetting functionality using 
    the Opticks translated Geant4 geometry.  

    Getting this to work requires:

    1. compilation WITH_G4CXOPTICKS
    2. SEvt::Exists true, this typically requires 
       an opticksNode greater than zero, configure with 
       the tut_detsim.py option "--opticks-mode 1/2/3"  
    3. OPTICKS_INPUT_PHOTONS envvar identifying an 
       existing .npy file containing the photons

    To disable use of GtOpticksTool input photons simply replace 
    the "opticks" argument on the tut_detsim.py commandline 
    with for example "gun". 


    junotoptask:GenTools.initialize  INFO: configure tool "ok" failed
    junotoptaskalgorithms.initialize ERROR: junotoptask:GenTools initialize failed
    [2024-09-24 21:54:31,821] p77377 {/data/blyth/junotop/junosw/InstallArea/python/Tutorial/JUNOApplication.py:201} INFO - ]JU




Looks like SEvt not instanciated. Investigate by backing off : add SEvt debug and switch off input photon running 
to see where the SEvt gets instanciated.::

    jok-; SEvt=INFO BP=SEvt::SEvt GUN=1 jok-tds-gdb




