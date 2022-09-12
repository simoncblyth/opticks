

#ifdef DEBUG_CYLINDER

#include "SSys.hh"
#include "NP.hh"
#include "PLOG.hh"
#include "scuda.h"
#include "squad.h"

#include "CSGDebug_Cylinder.hh"

const plog::Severity CSGDebug_Cylinder::LEVEL = PLOG::EnvLevel("CSGDebug_Cylinder", "DEBUG") ; 

std::vector<CSGDebug_Cylinder> CSGDebug_Cylinder::record = {} ;     


void CSGDebug_Cylinder::Save(const char* dir)  // static
{
    unsigned num_record = record.size() ;  
    LOG(info) << " dir " << dir << " num_record " << num_record ; 

    if( num_record > 0)
    {
        NP::Write<float>(dir, NAME, (float*)record.data(),  num_record, NUM_QUAD, 4 );  
    }
    else
    {
        LOG(error) << "not writing as no records" ; 
    }
}


#endif

