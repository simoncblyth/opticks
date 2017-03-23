#include "NGenerator.hpp"

#include "PLOG.hh"
#include "NBox.hpp"
#include "NBBox.hpp"

#include "NPY_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    nbox box = make_nbox( 0,0,0, 10 );
    nbbox bb = box.bbox();

    NGenerator gen(bb);

    nvec3 xyz ; 
    for(int i=0 ; i < 100 ; i++, gen(xyz)) LOG(info) << std::setw(5) << i <<  " " << xyz.desc() ; 

    return 0 ; 
}
