#include <cstdio>

#include "G4RunManager.hh"

#include "G4UImanager.hh"
#include "G4String.hh"
#include "G4UIExecutive.hh"

#include "PhysicsList.hh"
#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"
#include "Recorder.hh"

#include "NLog.hpp"
#include <boost/lexical_cast.hpp>

int parse(char* arg)
{
   int iarg = 0 ;
   try{ 
        iarg = boost::lexical_cast<int>(arg) ;
    }   
    catch (const boost::bad_lexical_cast& e ) { 
        LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
    }   
    catch( ... ){
        LOG(warning) << "Unknown exception caught!" ;
    }
    return iarg ;   
}

int main(int argc, char** argv)
{
    printf("%s\n", argv[0]);

    int nevt = argc > 1 ? parse(argv[argc-1]) : 0 ;
    if(!nevt) nevt = 1 ; 

    // G4 cant handle large numbers of primaries
    // so split the propagation into multiple "events" 
    unsigned int photons_per_event = 10000 ; 
    unsigned int nphotons = nevt*photons_per_event ; 

    LOG(info) << argv[0] 
              << " nevt " << nevt 
              << " nphotons " << nphotons
              ;

    G4RunManager* runManager = new G4RunManager;
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserInitialization(new DetectorConstruction());

    RecorderBase* recorder = new Recorder(nphotons,10, photons_per_event); 
    runManager->SetUserInitialization(new ActionInitialization(recorder));
    runManager->Initialize();
    runManager->BeamOn(nevt);

    recorder->save("/tmp/recorder.npy");

    delete runManager;
    return 0 ; 
}
