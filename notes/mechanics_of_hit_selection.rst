mechanics_of_hit_selection
=============================


The selection is done by sysrap/sphoton.h:sphoton_selector::

    397 struct sphoton_selector
    398 {
    399     unsigned hitmask ;
    400     sphoton_selector(unsigned hitmask_) : hitmask(hitmask_) {};
    401     SPHOTON_METHOD bool operator() (const sphoton& p) const { return ( p.flagmask & hitmask ) == hitmask  ; }   // require all bits of the mask to be     set 
    402 };
    403 

So, to understand the mechanics of hit selection look at the below and related files::

    epsilon:opticks blyth$ opticks-fl sphoton_selector
    ./sysrap/SU.cu
    ./sysrap/tests/sphoton_test.cc
    ./sysrap/sphoton.h
    ./sysrap/SEvt.hh
    ./sysrap/SU.hh
    ./sysrap/SEvt.cc
    ./qudarap/QEvent.hh

I highlight the most important ones to understand below. 
Use "export QEvent=INFO" for example to up the logging of the relevant classes/structs.  

qudarap/QEvent.cc::

     59 QEvent::QEvent()
     60     :
     61     sev(SEvt::Get()),
     62     selector(sev ? sev->selector : nullptr),
     63     evt(sev ? sev->evt : nullptr),
     64     d_evt(QU::device_alloc<sevent>(1,"QEvent::QEvent/sevent")),
     65     gs(nullptr),

sysrap/SEvt.cc instanciation of SEvt instanciates the selector using SEventConfig::HitMask()::

      44 SEvt::SEvt()
      45     :
      46     cfgrc(SEventConfig::Initialize()),  // config depends on SEventConfig::SetEventMode OR OPTICKS_EVENTMODE envvar 
      47     index(MISSING_INDEX),
      48     reldir(DEFAULT_RELDIR),
      49     selector(new sphoton_selector(SEventConfig::HitMask())),
      50     evt(new sevent),
      51     dbg(new sdebug),

sysrap/SEventConfig.hh,cc shows you can change the default hitmask of "SD" using envvar OPTICKS_HIT_MASK::

    76     static constexpr const char* kHitMask    = "OPTICKS_HIT_MASK" ;

    33 const char* SEventConfig::_HitMaskDefault = "SD" ;
    ...
    66 unsigned SEventConfig::_HitMask  = OpticksPhoton::GetHitMask(SSys::getenvvar(kHitMask, _HitMaskDefault )) ;


qudarap/QEvent.cc QEvent::gatherHit is where the hits array is obtained from the photon array::

    520 /**
    521 QEvent::gatherHit
    522 ------------------
    523 
    524 1. count *evt.num_hit* passing the photon *selector* 
    525 2. allocate *evt.hit* GPU buffer
    526 3. copy_if from *evt.photon* to *evt.hit* using the photon *selector*
    527 4. host allocate the NP hits array
    528 5. copy hits from device to the host NP hits array 
    529 6. free *evt.hit* on device
    530 7. return NP hits array to caller, who becomes owner of the array 
    531 
    532 Note that the device hits array is allocated and freed for each launch.  
    533 This is due to the expectation that the number of hits will vary greatly from launch to launch 
    534 unlike the number of photons which is expected to be rather similar for most launches other than 
    535 remainder last launches. 
    536 
    537 The alternative to this dynamic "busy" handling of hits would be to reuse a fixed hits buffer
    538 sized to max_photons : that however seems unpalatable due it always doubling up GPU memory for 
    539 photons and hits.  
    540 
    541 hitmask metadata was formerly placed on the hit array, 
    542 subsequently moved to domain_meta as domain should 
    543 always be present, unlike hits. 
    544 
    545 **/
    546 
    547 NP* QEvent::gatherHit() const
    548 {
    549     // hasHit at this juncture is misleadingly always false, 
    550     // because the hits array is derived by *gatherHit_* which  selects from the photons 
    551 
    552     bool has_photon = hasPhoton();
    553 
    554     LOG_IF(LEVEL, !has_photon) << " gatherHit called when there is no photon array " ;
    555     if(!has_photon) return nullptr ;
    556 
    557     assert( evt->photon );
    558     assert( evt->num_photon );
    559     evt->num_hit = SU::count_if_sphoton( evt->photon, evt->num_photon, *selector );
    560 
    561     LOG(LEVEL)
    562          << " evt.photon " << evt->photon
    563          << " evt.num_photon " << evt->num_photon
    564          << " evt.num_hit " << evt->num_hit
    565          << " selector.hitmask " << selector->hitmask
    566          << " SEventConfig::HitMask " << SEventConfig::HitMask()
    567          << " SEventConfig::HitMaskLabel " << SEventConfig::HitMaskLabel()
    568          ;
    569 
    570     NP* hit = evt->num_hit > 0 ? gatherHit_() : nullptr ;
    571 
    572     return hit ;
    573 }
    574 
    575 NP* QEvent::gatherHit_() const
    576 {
    577     evt->hit = QU::device_alloc<sphoton>( evt->num_hit, "QEvent::gatherHit_:sphoton" );
    578 
    579     SU::copy_if_device_to_device_presized_sphoton( evt->hit, evt->photon, evt->num_photon,  *selector );
    580 
    581     NP* hit = NP::Make<float>( evt->num_hit, 4, 4 );
    582 
    583     QU::copy_device_to_host<sphoton>( (sphoton*)hit->bytes(), evt->hit, evt->num_hit );
    584 
    585     QU::device_free<sphoton>( evt->hit );
    586 
    587     evt->hit = nullptr ;
    588     LOG(LEVEL) << " hit.sstr " << hit->sstr() ;
    589 
    590     return hit ;
    591 }
    592 



