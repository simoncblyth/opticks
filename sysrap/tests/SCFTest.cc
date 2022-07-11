#include "OPTICKS_LOG.hh"
#include "SPath.hh"
#include "SSys.hh"
#include "SCF.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SCF* cf = SCF::Create(); 

    LOG(info) << " cf " << cf << " cf.desc " << ( cf ? cf->desc() : "-" ) ;  

    if(cf == nullptr) return 1 ; 

    std::vector<int>* instv = SSys::getenvintvec("INST") ; 
    if(instv == nullptr) return 1 ; 

    for(unsigned i=0 ; i < instv->size() ; i++)
    {
        int instIdx = (*instv)[i] ;  
        const qat4* q = cf->getInst( instIdx ); 
        if(q == nullptr) continue  ; 
        std::cout << std::setw(5) <<  instIdx << " " << q->desc() << " " << q->descId() << std::endl ; 
    }

    return 0 ; 
}
