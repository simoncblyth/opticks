G4Opticks-need-to-move-JUNO-specific-outer-volume-kludge-into-JUNO-code-rather-than-here
==========================================================================================


The *outer_volume=true* causing assert problem for Hans.

::

     406 void G4Opticks::setGeometry(const GGeo* ggeo)
     407 {
     408     bool loaded = ggeo->isLoadedFromCache() ;
     409     unsigned num_sensor = ggeo->getNumSensorVolumes();
     410 
     411     m_sensorlib = new SensorLib();
     412     m_sensorlib->initSensorData(num_sensor);
     413 
     414 
     415     if( loaded == false )
     416     {
     417         bool outer_volume = true ;
     418         X4PhysicalVolume::GetSensorPlacements(ggeo, m_sensor_placements, outer_volume);
     419         assert( num_sensor == m_sensor_placements.size() ) ;
     420     }
     421 
     422     LOG(info)
     423         << " GGeo: "
     424         << ( loaded ? "LOADED FROM CACHE " : "LIVE TRANSLATED " )
     425         << " num_sensor " << num_sensor
     426         ;
     427 
     428     m_ggeo = ggeo ;
     429     m_blib = m_ggeo->getBndLib();
     430     m_hits_wrapper = new GPho(m_ggeo) ;   // geometry aware photon hits wrapper
     431 


::

    503 /**
    504 GNodeLib::getSensorPlacements
    505 ------------------------------
    506 
    507 TODO: eliminate the outer_volume kludge 
    508 
    509 When outer_volume = true the placements returned are not 
    510 those of the sensors themselves but rather those of the 
    511 outer volumes of the instances that contain the sensors.
    512 
    513 That is probably a kludge needed because it is the 
    514 CopyNo of the  outer volume that carries the sensorId
    515 for JUNO.  Need a way of getting that from the actual placed
    516 sensor volume in detector specific code, not here.
    517 
    518 **/
    519 
    520 void GNodeLib::getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const
    521 {
    522     unsigned numSensorVolumes = getNumSensorVolumes();
    523     for(unsigned i=0 ; i < numSensorVolumes ; i++)
    524     {
    525         unsigned sensorIdx = i ;
    526         const GVolume* sensor = getSensorVolume(sensorIdx) ;
    527         assert(sensor);
    528 
    529         void* origin = NULL ;
    530 
    531         if(outer_volume)
    532         {
    533             const GVolume* outer = sensor->getOuterVolume() ;
    534             assert(outer);
    535             origin = outer->getOriginNode() ;
    536         }
    537         else
    538         {
    539             origin = sensor->getOriginNode() ;
    540         }
    541 
    542         placements.push_back(origin);
    543     }
    544 }




Backtrace from Hans that was fixed by setting outer_volume=false in 

::

    thanks for the prompt reply.  here is the stack trace. I set
    outer_volume=false in GNodeLib.cc and it looks like the program is running fine
    (didn't have yet time to look at the plots yet and other statistics yet). 


    cheers Hans 
    __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:51
    51 ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) bt
        file=file@entry=0x7fffec95fb60 "/home/wenzel/gputest/opticks/ggeo/GNodeLib.cc", line=line@entry=525,
        function=function@entry=0x7fffec9604c0 <GNodeLib::getSensorPlacements(std::vector<void*, std::allocator<void*> >&, bool) const::__PRETTY_FUNCTION__> "void GNodeLib::getSensorPlacements(std::vector<void*>&, bool) const")
        at assert.c:92
    #3  0x00007fffe9cb3502 in __GI___assert_fail (assertion=0x7fffec95fe7f "outer", file=0x7fffec95fb60 "/home/wenzel/gputest/opticks/ggeo/GNodeLib.cc", line=525,
        function=0x7fffec9604c0 <GNodeLib::getSensorPlacements(std::vector<void*, std::allocator<void*> >&, bool) const::__PRETTY_FUNCTION__> "void GNodeLib::getSensorPlacements(std::vector<void*>&, bool) const") at assert.c:101
    #4  0x00007fffec932f6b in GNodeLib::getSensorPlacements (this=0x555556b11a60, placements=std::vector of length 0, capacity 0, outer_volume=true) at /home/wenzel/gputest/opticks/ggeo/GNodeLib.cc:525
    #5  0x00007fffec9293dd in GGeo::getSensorPlacements (this=0x555556b13db0, placements=std::vector of length 0, capacity 0, outer_volume=true) at /home/wenzel/gputest/opticks/ggeo/GGeo.cc:957
    #6  0x00007ffff65387e4 in X4PhysicalVolume::GetSensorPlacements (gg=0x555556b13db0, placements=std::vector of length 0, capacity 0, outer_volume=true) at /home/wenzel/gputest/opticks/extg4/X4PhysicalVolume.cc:1481
    #7  0x00007ffff7bc3251 in G4Opticks::setGeometry (this=0x5555569efd10, ggeo=0x555556b13db0) at /home/wenzel/gputest/opticks/g4ok/G4Opticks.cc:418
    #8  0x00007ffff7bc2fe4 in G4Opticks::setGeometry (this=0x5555569efd10, world=0x555555ce50c0) at /home/wenzel/gputest/opticks/g4ok/G4Opticks.cc:372
    #9  0x00007ffff7bc2d68 in G4Opticks::setGeometry (this=0x5555569efd10, world=0x555555ce50c0, standardize_geant4_materials=false) at /home/wenzel/gputest/opticks/g4ok/G4Opticks.cc:354
    #10 0x000055555557d5a4 in RunAction::BeginOfRunAction (this=0x555555c79ad0) at /home/wenzel/gputest/G4OpticksTest/src/RunAction.cc:44
    #11 0x00007ffff45c20b5 in G4RunManager::RunInitialization() () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4run.so
    #12 0x00007ffff45ba656 in G4RunManager::BeamOn(int, char const*, int) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4run.so
    #13 0x00007ffff45d96f3 in G4RunMessenger::SetNewValue(G4UIcommand*, G4String) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4run.so
    #14 0x00007fffee2913cc in G4UIcommand::DoIt(G4String) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4intercoms.so
    #15 0x00007fffee2b2d2c in G4UImanager::ApplyCommand(char const*) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4intercoms.so
    #16 0x00007fffee27cc87 in G4UIbatch::ExecCommand(G4String const&) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4intercoms.so
    #17 0x00007fffee27ecae in G4UIbatch::SessionStart() () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4intercoms.so
    #18 0x00007fffee2b422c in G4UImanager::ExecuteMacroFile(char const*) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4intercoms.so
    #19 0x00007fffee29e9db in G4UIcontrolMessenger::SetNewValue(G4UIcommand*, G4String) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4intercoms.so
    #20 0x00007fffee2913cc in G4UIcommand::DoIt(G4String) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4intercoms.so
    #21 0x00007fffee2b2d2c in G4UImanager::ApplyCommand(char const*) () from /home/wenzel/geant4.10.06.p02_clhep-install/lib/libG4intercoms.so
    #22 0x0000555555565e89 in main (argc=3, argv=0x7fffffffcdc8) at /home/wenzel/gputest/G4OpticksTest/G4OpticksTest.cc:81


