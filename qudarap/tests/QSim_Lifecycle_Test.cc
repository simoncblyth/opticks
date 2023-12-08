/**
QSim_Lifecycle_Test.cc
========================

**/

#include <sstream>

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"

#include "SEventConfig.hh"
#include "scuda.h"
#include "squad.h"
#include "ssys.h"
#include "spath.h"   
#include "sstate.h"

#include "SSim.hh"
#include "SBnd.h"
#include "SPrd.h"

#include "SEvt.hh"
#include "NP.hh"

#include "QSim.hh"


int main(int argc, char** argv)
{
    SSim* sim = SSim::Load(); 
    QSim::UploadComponents(sim);   // instanciates things like QBnd : NORMALLY FIRST GPU ACCESS 
    const SPrd* prd = sim->get_sprd() ; 

    LOG_IF(error, prd->rc != 0 )
        << " SPrd::rc NON-ZERO " << prd->rc 
        << " NOT ALL CONFIGURED BOUNDARIES ARE IN THE GEOMETRY "
        ;
    if(prd->rc != 0 ) return 1 ; 


    SEvt::Create(SEvt::EGPU) ; 

    return 0 ; 
}



