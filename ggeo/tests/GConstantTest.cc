#include <cassert>
#include "GConstant.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << "GConstant::meter " << GConstant::meter ;

    assert(GConstant::meter == 1000.f ); 

    return 0 ;
}
