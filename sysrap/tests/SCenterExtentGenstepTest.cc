#include "OPTICKS_LOG.hh"

#include "scuda.h"
#include "squad.h"
#include "SCenterExtentGenstep.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    float4 ce ; 
    qvals(ce, "CE" , "1000,0,0,100" ); 
    LOG(info) << " CE " << ce ; 

    SCenterExtentGenstep* cegs = new SCenterExtentGenstep(&ce) ; 

    LOG(info) << " cegs " << cegs->desc();   

    cegs->dumpBoundingBox(); 

    cegs->save(); 

    return 0 ; 
}
