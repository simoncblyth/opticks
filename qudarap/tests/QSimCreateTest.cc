#include "OPTICKS_LOG.hh"

//#include "scuda.h"
//#include "squad.h"

#include "SEvt.hh"
#include "QSim.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt evt ; 

    QSim* qs = QSim::Create() ; 
    std::cout << " qs.desc " << qs->desc() << std::endl ; 

    return 0 ; 
}
