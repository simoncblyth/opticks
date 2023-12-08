#include "OPTICKS_LOG.hh"

#include "SEvt.hh"
#include "QSim.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt::Create(SEvt::EGPU) ;

    QSim* qs = QSim::Create() ; 
    std::cout << " qs.desc " << qs->desc() << std::endl ; 
    std::cout << " qs.descFull " << qs->descFull() << std::endl ; 

    return 0 ; 
}
