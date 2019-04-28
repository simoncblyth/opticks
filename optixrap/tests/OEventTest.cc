#include <string>
#include <sstream>

#include "NPY.hpp"

#include "OEvent.hh"
#include "OContext.hh"
#include "OConfig.hh"

#include "STimes.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksRun.hh"
#include "OpticksGen.hh"
#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"

#include "OPTICKS_LOG.hh"


/**
OEventTest
============

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    Opticks ok(argc, argv, "--machinery --compute");
    OpticksRun* run = ok.getRun(); 

    OpticksHub hub(&ok);


    NPY<float>* gs0 = hub.getInputGensteps(); 
    assert(gs0);

    unsigned version = OConfig::OptiXVersion()  ;
    LOG(info) << argv[0] << " OPTIX_VERSION " << version ; 
    //bool with_top = OConfig::DefaultWithTop() ;  // must set false with 3080, seemingly doesnt matter with 40000

    OContext* ctx = OContext::Create(&ok);
    int entry = ctx->addEntry("OEventTest.cu", "OEventTest", "exception");

    OEvent* oevt = new OEvent(&ok, ctx);   
 
    bool prelaunch = false ; 

    for(unsigned i=0 ; i < 10 ; i++)
    {
         NPY<float>* gs = gs0->clone();

         run->createEvent(i);
         OpticksEvent* evt = run->getEvent();

         assert(evt->isMachineryType() && "--machinery type is forced as this writes incomplete OpticksEvents which would otherwise cause test failures for event reading tests" );

         evt->setGenstepData(gs);

         evt->addBufferControl("photon", OpticksBufferControl::COMPUTE_MODE_ );
         evt->addBufferControl("record", OpticksBufferControl::COMPUTE_MODE_ );
         // defaults to INTEROP, need to set to COMPUTE to get OContext::download to not skip
        
          
         LOG(info) << "( upload " ;  
         oevt->upload();
         LOG(info) << ") upload " ;  

         if(!prelaunch)
         {
             LOG(info) << "( prelaunch " ;  
             ctx->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, evt->getPrelaunchTimes() );
             LOG(info) << ") prelaunch " ;  

             prelaunch = true ; 
         } 

         LOG(info) << "( launch " ;  
         ctx->launch( OContext::LAUNCH, entry,  evt->getNumPhotons(), 1, evt->getLaunchTimes());
         LOG(info) << ") launch " ;  

         LOG(info) << "( download " ;  
         oevt->download();
         LOG(info) << ") download " ;  

         evt->save();

         LOG(info) <<  evt->description() ;
    }

    delete ctx ; 

    return 0 ;     
}

/*
::

   ipython -i $(which tevt.py) -- --tag 5

*/


