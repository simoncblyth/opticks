Geant4_output_stream_not_playing_nice_with_PLOG
==================================================

Seems that PLOG grabs the output stream and prevents
any output until the end of execution.  
Same thing with lldb and without.

But not all output (eg the small amounts from GDML or the banner works fine), 
think the problem is with big chunks of output from Geant4, such as
EM reporting.

Perhaps need to add some std::flush ? Or a delay after G4 initialization.::

   g4-cls G4ios
   g4-cls G4strstreambuf

Geant4 culprit for writing big chunks of text is G4EmModelManager::

   g4-cc "===== EM models for the G4Region "
   g4-cls G4EmModelManager


::

    130 /////////////////////////////////////////////
    131 inline G4int G4strstreambuf::ReceiveString ()
    132 /////////////////////////////////////////////
    133 {
    134   G4String stringToSend(buffer);
    135   G4int result= 0;
    136 
    137   if(this == &G4coutbuf && destination != 0)
    138   {
    139     result=  destination-> ReceiveG4cout_(stringToSend);
    140   }
    141   else if(this == &G4cerrbuf && destination != 0)
    142   {
    143     result=  destination-> ReceiveG4cerr_(stringToSend);
    144   }
    145   else if(this == &G4coutbuf && destination == 0)
    146   {
    147     std::cout << stringToSend << std::flush;
    148     result= 0;
    149   }
    150   else if(this == &G4cerrbuf && destination == 0)
    151   {
    152     std::cerr << stringToSend << std::flush;
    153     result= 0;
    154   }
    155 
    156   return result;
    157 }




::

    hIoni:   for  pi-    SubType= 2
          dE/dx and range tables from 100 eV  to 100 TeV in 84 bins
          Lambda tables from threshold to 100 TeV, 7 bins per decade, spline: 1
          finalRange(mm)= 0.1, dRoverRange= 0.2, integral: 1, fluct: 1, linLossLimit= 0.01
          ===== EM models for the G4Region  DefaultRegionForTheWorld ======
                ICRU73QO :  Emin=     2018-08-21 14:13:21.251 INFO  [578306] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2018-08-21 14:13:21.251 FATAL [578306] [CPrimarySource::GeneratePrimaryVertex@44] CPrimarySource::GeneratePrimaryVertex
    2018-08-21 14:13:21.252 INFO  [578306] [CSensitiveDetector::Initialize@56]  HCE 0x112a68b00 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    2018-08-21 14:13:21.252 INFO  [578306] [CSensitiveDetector::ProcessHits@47] .



    2018-08-21 14:13:22.032 INFO  [578306] [CG4::cleanup@362] opening geometry
    2018-08-21 14:13:22.032 INFO  [578306] [CG4::cleanup@364] opening geometry DONE 
       0 eV    Emax=  297.505 keV
              BetheBloch :  Emin=  297.505 keV   Emax=      100 TeV
    s from threshold to 100 TeV, 7 bins per decade, spline: 1
          finalRange(mm)= 0.1, dRoverRange= 0.2, integral: 1, fluct: 1, linLossLimit= 0.01
    ...

    hIoni:   for  pi-    SubType= 2
          dE/dx and range tables from 100 eV  to 100 TeV in 84 bins
          Lambda tables from threshold to 100 TeV, 7 bins per decade, spline: 1
          finalRange(mm)= 0.1, dRoverRange= 0.2, integral: 1, fluct: 1, linLossLimit= 0.01
          ===== EM models for the G4Region  DefaultRegionForTheWorld ======
                ICRU73QO :  Emin=     Process 12574 exited with status = 0 (0x00000000) 
    (lldb) 

