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

    const char* typ = "torch" ;
    //const char* tag = "-5" ;
    const char* tag = "-6" ;
    const char* det = "rainbow" ;

    unsigned int photons_per_event(0) ; 
    if(nevt==0) 
    {
        nevt = 1 ; 
        photons_per_event = 10 ;
    }
    else
    {
        photons_per_event = 10000 ; 
    } 

    unsigned int nphotons = nevt*photons_per_event ; 

    LOG(info) << argv[0] 
              << " nevt " << nevt 
              << " photons_per_event " << photons_per_event
              << " nphotons " << nphotons
              ;

    DetectorConstruction* dc = new DetectorConstruction() ; 
    RecorderBase* recorder = new Recorder(typ,tag,det,nphotons,10, photons_per_event); 

    if(strcmp(tag, "-5") == 0)  recorder->setIncidentSphereSPolarized(true) ;


    G4RunManager* runManager = new G4RunManager;
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserInitialization(dc);
    runManager->SetUserInitialization(new ActionInitialization(recorder));
    runManager->Initialize();

    recorder->setCenterExtent(dc->getCenterExtent());
    recorder->setBoundaryDomain(dc->getBoundaryDomain());
    

    runManager->BeamOn(nevt);

    recorder->save();

    delete runManager;
    return 0 ; 
}
