#include "NGrid3.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 


    int msk = (1 << 6) - 1 ;  // 0b111111 ;  

    NGrid3 grid(3);

    LOG(info) << grid.desc();
    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        if((loc & msk) == msk )
        {
             nivec3 ijk = grid.ijk(loc) ;   // z-order morton index -> ijk
             nvec3  xyz = grid.fpos(ijk);    // ijk -> fractional 
             nivec3 ijk2 = grid.ijk(xyz);   // fractional -> ijk 

             assert( ijk.x == ijk2.x );
             assert( ijk.y == ijk2.y );
             assert( ijk.z == ijk2.z );


             LOG(info) 
                   << " loc " << std::setw(5) << loc 
                   << " ijk " << ijk.desc()
                   << " fpos " << xyz.desc() 
                   ; 

        }

    }

    NGrid3::dump_levels();



    return 0 ; 
}
