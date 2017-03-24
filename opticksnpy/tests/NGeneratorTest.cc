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

    nvec3 p ;
    int n = 100 ;  

    while(n--)
    {
        gen(p);
        LOG(info) << std::setw(5) << n <<  " " << p.desc() ; 
    }

    return 0 ; 
}
