#include "NGrid3.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


void test_basics()
{
    int msk = (1 << 6) - 1 ;  // 0b111111 ;  

    NGrid3 grid(3);

    LOG(info) << grid.desc();
    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        nivec3 ijk = grid.ijk(loc) ;   // z-order morton index -> ijk
        int loc2 = grid.loc(ijk);      // ijk -> morton
        assert( loc2 == loc);

        nvec3  xyz = grid.fpos(ijk);    // ijk -> fractional 
        int loc3 = grid.loc(xyz);       // fractional -> morton
        assert( loc3 == loc);

        nivec3 ijk2 = grid.ijk(xyz);   // fractional -> ijk 
        assert( ijk2 == ijk );

        if((loc & msk) == msk )
        {

             LOG(info) 
                   << " loc " << std::setw(5) << loc 
                   << " ijk " << ijk.desc()
                   << " fpos " << xyz.desc() 
                   ; 

        }

    }

}


void test_coarse_nominal()
{

    NGrid3 nominal(7);
    NGrid3 coarse(5);
    int elevation = nominal.level - coarse.level ;

    int n_loc = 0.45632*nominal.nloc ;    // some random loc

    nivec3 n_ijk = nominal.ijk(n_loc) ; 
    nvec3  n_xyz = nominal.fpos(n_ijk) ;


    int msk = (1 << (elevation*3)) - 1 ;  
    int n_loc_0 = n_loc & ~msk ;   // nominal loc, but with low bits scrubbed -> bottom left of tile 

    nivec3 n_ijk_0 = nominal.ijk(n_loc_0 ) ;  
    nvec3  n_xyz_0 = nominal.fpos(n_ijk_0 );
    
    // coarse level is parent or grandparent etc.. in tree
    // less nloc when go coarse : so must down shift 

    int c_loc = n_loc >> (3*elevation) ;
    int c_size = 1 << elevation ; 


    int c2n_loc = ( n_loc >> (3*elevation) ) << (3*elevation) ; // down and then back up
    assert(c2n_loc == n_loc_0);  // same as scrubbing the low bits


    nivec3 c_ijk = coarse.ijk(c_loc) ; 
    nvec3  c_xyz = coarse.fpos(c_ijk );  

    nivec3 c2n_ijk = make_nivec3( c_ijk.x*c_size , c_ijk.y*c_size, c_ijk.z*c_size );


    std::cout 
           << " n_loc   " << std::setw(6) << n_loc
           << " n_ijk   " << n_ijk.desc()
           << " n_xyz   " << n_xyz.desc()
           << " (nominal) " 
           << std::endl 
           ;

    std::cout 
           << " n_loc_0 " << std::setw(6) << n_loc_0
           << " n_ijk_0 " << n_ijk_0.desc()
           << " n_xyz_0 " << n_xyz_0.desc()
           << " (nominal) with high res bits scrubbed  " 
           << std::endl 
           ;

    std::cout 
           << " c_loc   " << std::setw(6) << c_loc 
           << " c_ijk   " << c_ijk.desc()
           << " c_xyz   " << c_xyz.desc()
           << " (coarse coordinates : a different ballpark   " 
           << std::endl 
           ;

    std::cout 
           << " c2n_ijk " << c2n_ijk.desc()
           << " (coarse coordinates scaled back to nominal)    " 
           << std::endl 
           ;
 

}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    test_basics();
    test_coarse_nominal();

    NMultiGrid3 mg ; 
    NGrid3* g5 = mg.grid[5] ; 
    NGrid3* g7 = mg.grid[7] ; 

    std::cout << " g5 " << g5->desc() << std::endl ; 
    std::cout << " g7 " << g7->desc() << std::endl ; 

    nvec3 fpos = make_nvec3(0.1f, 0.2f, 0.3f ); 
    mg.dump("NMultiGrid3 dump, eg pos", fpos);
    mg.dump("NMultiGrid3 dump");
   

    return 0 ; 
}
