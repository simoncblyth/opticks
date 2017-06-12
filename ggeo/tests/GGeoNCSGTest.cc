#include "NCSG.hpp"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << NCSG::NTRAN ; 

    return 0 ; 
}
