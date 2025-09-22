CSGOptiXService_impl_notes
============================

Currently just placeholder to check nanobind.
Now that the binding is working, need to add
a more realistic test impl.


Start from cxs_min.sh test which uses::

     161 int CSGOptiX::SimulateMain() // static
     162 {
     163     SProf::Add("CSGOptiX__SimulateMain_HEAD");
     164     SEventConfig::SetRGModeSimulate();
     165     CSGFoundry* fd = CSGFoundry::Load();
     166     CSGOptiX* cx = CSGOptiX::Create(fd) ;
     167     for(int i=0 ; i < SEventConfig::NumEvent() ; i++) cx->simulate(i);
     168     SProf::UnsetTag();
     169     SProf::Add("CSGOptiX__SimulateMain_TAIL");
     170     SProf::Write("run_meta.txt", true ); // append:true
     171     cx->write_Ctx_log();
     172     delete cx ;
     173     return 0 ;
     174 }


Need to pull out where the gensteps are coming from and where the hits
are going and expose those in the API.

::

     716 double CSGOptiX::simulate(int eventID)
     717 {
     718     SProf::SetTag(eventID, "A%0.3d_" ) ;
     719     assert(sim);
     720     bool reset = true ;   // reset:true calls SEvt::endOfEvent for cleanup after simulate
     721     double dt = sim->simulate(eventID, reset) ; // (QSim)
     722     return dt ;
     723 }




::

     408 double QSim::simulate(int eventID, bool reset_)
     409 {
     423     sev->beginOfEvent(eventID);  // set SEvt index and tees up frame gensteps for simtrace and input photon simulate running
     424 
     425     NP* igs = sev->makeGenstepArrayFromVector();
     426 
     427     MaybeSaveIGS(eventID, igs);
     428 
     429     std::vector<sslice> igs_slice ;
     430     SGenstep::GetGenstepSlices( igs_slice, igs, SEventConfig::MaxSlot() );
     431     int num_slice = igs_slice.size();

     444     for(int i=0 ; i < num_slice ; i++)
     445     {
     448         const sslice& sl = igs_slice[i] ;
     451 
     452         int rc = event->setGenstepUpload_NP(igs, &sl ) ; // upload gensteps, OR a slice of them if photon total would not fit VRAM
 
     466         double dt = rc == 0 && cx != nullptr ? cx->simulate_launch() : -1. ;  //SCSGOptiX protocol

     477         sev->gather();  // gather into *fold* just added to *topfold*

     482     }

     488     int concat_rc = sev->topfold->concat(out);    // concatenate result arrays from all the slices 

     499     int tot_ht = sev->getNumHit() ;  // NB from fold, so requires hits array gathering to be configured to get non-zero

     513     if(reset_) reset(eventID) ;


     543     return tot_dt ;
     544 }



     643 void QSim::reset(int eventID)
     644 {
     646     event->clear();
     647     sev->endOfEvent(eventID);
     650 }





