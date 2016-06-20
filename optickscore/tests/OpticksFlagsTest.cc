#include "OpticksFlags.hh"
#include "Index.hpp"

int main(int, char** argv)
{
    OpticksFlags f ; 
    Index* i = f.getIndex();
    i->dump(argv[0]);
    return 0 ; 
}
