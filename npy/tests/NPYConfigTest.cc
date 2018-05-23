#include "NPYConfig.hpp"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) 
         << "NPYConfig::OptionalExternals "
         << NPYConfig::OptionalExternals() 
         ; 

    return 0 ; 
}
