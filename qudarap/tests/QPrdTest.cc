
// without these three include vector complains about pointer arithmetric on forward decl float3 quad2 
#include <vector_functions.h>
#include "scuda.h"
#include "squad.h"

#include "NP.hh"
#include "SOpticksResource.hh"
#include "SPath.hh"
#include "QBnd.hh"
#include "QPrd.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cfbase = SOpticksResource::CFBase("CFBASE") ; 
    LOG(info) << " cfbase " << cfbase ; 
    NP* bnd = NP::Load(cfbase, "CSGFoundry", "bnd.npy"); 

    QBnd qb(bnd); 
    QPrd qp ; 

    LOG(info) << qp.desc(); 

    unsigned num_photon = 8 ; 
    unsigned num_bounce = 6 ; 
    NP* prd = qp.duplicate_prd(num_photon, num_bounce); 
    prd->dump(); 

    const char* path = SPath::Resolve("$TMP/QPrdTest/prd.npy", FILEPATH ); 
    LOG(info) << " path " << path ; 
    prd->save(path); 

    return 0 ; 
}
