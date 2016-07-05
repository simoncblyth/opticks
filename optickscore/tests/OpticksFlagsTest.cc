#include "OpticksFlags.hh"
#include "Index.hpp"

#include "OKCORE_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG__ ;   


    OpticksFlags f ; 
    Index* i = f.getIndex();
    i->dump(argv[0]);
    return 0 ; 
}
