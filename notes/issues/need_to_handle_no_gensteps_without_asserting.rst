need_to_handle_no_gensteps_without_asserting
================================================


Ami reports::

    On Oct 18, 2022, at 1:13 AM, Amirreza Hashemi <amirezahashemi@gmail.com> wrote:

    Hi Simon,

    Thanks very much for the response. For our set of problems, we largely deal with scintillation, I think I could apply to my example code (example_pet_opticks/src/mySenstitiveDetector.cc) but I end up getting a new error which is 

    2022-10-17 20:01:42.870 ERROR [1131237] [QSim::simulate@296]  QEvent::setGenstep ERROR : have event but no gensteps collected : will skip cx.simulate 


    2022-10-17 20:01:42.870 ERROR [1131237] [SEvt::gather@1417] gather_done already skip gather 
    EndOfEventAction: num_hits: 0
    EventNo:          7
    EventAction::EndOfEventAction Event:   7
    example_pet: /home/ami/Documents/codes/new_opticks/test/opticks/qudarap/QEvent.cu:202: void QEvent_count_genstep_photons_and_fill_seed_buffer(sevent*): Assertion `evt->seed && evt->num_seed > 0' failed.
    Aborted (core dumped)



Hmm : this is probably related to recent change of allowing empty arrays.
Formerly empty arrays yielded nullptr.  


::

     281 /**
     282 QSim::simulate
     283 ---------------
     284 
     285 Canonically invoked from G4CXOpticks::simulate
     286 Collected genstep are uploaded and the CSGOptiX kernel is launched to generate and propagate. 
     287 
     288 **/
     289 
     290 double QSim::simulate()
     291 {
     292    LOG_IF(error, event == nullptr) << " event null " << desc()  ;
     293    if( event == nullptr ) std::raise(SIGINT) ;
     294    if( event == nullptr ) return -1. ;
     295 
     296    int rc = event->setGenstep() ;
     297    LOG_IF(error, rc != 0) << " QEvent::setGenstep ERROR : have event but no gensteps collected : will skip cx.simulate " ;
     298    double dt = rc == 0 && cx != nullptr ? cx->simulate() : -1. ;
     299    return dt ;
     300 }
     301 


::

    147 int QEvent::setGenstep()  // onto device
    148 {
    149     NP* gs = SEvt::GatherGenstep(); // TODO: review memory handling  
    150     SEvt::Clear();   // clear the quad6 vector, ready to collect more genstep
    151     LOG_IF(fatal, gs == nullptr ) << "Must SEvt::AddGenstep before calling QEvent::setGenstep " ;
    152     //if(gs == nullptr) std::raise(SIGINT); 
    153     return gs == nullptr ? -1 : setGenstep(gs) ;
    154 }
    155 




