#include "OPTICKS_LOG.hh"
#include "QCurandState.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    QCurandState* cs = QCurandState::Create() ; 
    LOG(info) << cs->desc() ;

    return 0 ; 
}
