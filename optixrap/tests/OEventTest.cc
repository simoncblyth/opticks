#include <string>
#include <sstream>

#include "NPY.hpp"

#include "OEvent.hh"
#include "OContext.hh"
#include "STimes.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksGen.hh"
#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"

#include "OKCORE_LOG.hh"
#include "OXRAP_LOG.hh"

#include "PLOG.hh"


/**
OEventTest
============

**/

int main(int argc, char** argv)
{
    PLOG_(argc, argv);    

    OKCORE_LOG__ ; 
    OXRAP_LOG__ ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);

    NPY<float>* gs0 = hub.getInputGensteps(); 
    assert(gs0);


    optix::Context context = optix::Context::create();
    OContext ctx(context, OContext::COMPUTE);
    int entry = ctx.addEntry("OEventTest.cu.ptx", "OEventTest", "exception");

    OEvent* oevt = new OEvent(&hub, &ctx);   
 
    bool prelaunch = false ; 

    for(unsigned i=0 ; i < 10 ; i++)
    {
         NPY<float>* gs = gs0->clone();

         hub.createEvent(i);

         OpticksEvent* evt = hub.getEvent();

         evt->setGenstepData(gs);


         NPYBase* ox = evt->getData("photon");
         OpticksBufferControl c_ox(ox->getBufferControlPtr());
         c_ox.add(OpticksBufferControl::COMPUTE_MODE_);    

         NPYBase* rx = evt->getData("record");
         OpticksBufferControl c_rx(rx->getBufferControlPtr());
         c_rx.add(OpticksBufferControl::COMPUTE_MODE_);    

         // without the buffer control the download is skipped, claiming interop 
         // TODO avoid this complication

         
         oevt->upload();

         if(!prelaunch)
         {
             ctx.launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, evt->getPrelaunchTimes() );
             prelaunch = true ; 
         } 

         ctx.launch( OContext::LAUNCH, entry,  evt->getNumPhotons(), 1, evt->getLaunchTimes());

         oevt->download();

         evt->save();

         //evt->check();

         LOG(info) <<  evt->description() ;
    }


    return 0 ;     
}

/*

::

   ipython -i $(which tevt.py) -- --tag 5


*/



