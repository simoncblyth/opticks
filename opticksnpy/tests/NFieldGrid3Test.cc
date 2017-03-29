#include <bitset>
#include <boost/math/special_functions/sign.hpp>

#include "NField3.hpp"
#include "NGrid3.hpp"
#include "NFieldGrid3.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

#include "NBox.hpp"
#include "NBBox.hpp"



void test_fieldgrid(const NField3& field, const NGrid3& grid, NFieldGrid3* fg)
{

    int ncross = 0 ; 
    float sdf_prior = 0 ; 
    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        nvec3 fpos = grid.fpos(loc);
        float sdf = field(fpos);

        nivec3 ijk = grid.ijk(loc);
        float sdf2 = fg->value(ijk);
        assert( sdf == sdf2 );


        bool cross_isosurface = loc > 0 && boost::math::sign(sdf) != boost::math::sign(sdf_prior) ;
        if(cross_isosurface) 
        {
            ncross++ ; 
            int zc = field.zcorners( fpos, grid.elem ); 
            LOG(info)
                << " zcorners " << std::setw(3) << zc
                << " 0x" << std::setfill('0') << std::setw(2) << std::hex << zc
                << " 0b" << std::bitset<8>(zc) 
                 << std::dec 
                << " sdf " << sdf 
                << " sdf2 " << sdf2
                ; 

            ;
        }
        sdf_prior = sdf ;
    }
    LOG(info) << "(test_fieldgrid) ncross " << ncross ; 
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    nbox world = make_nbox(0,0,0,11) ; 
    nbbox wbb = world.bbox() ;

    nbox obj = make_nbox(0,0,0,7) ; 

    std::function<float(float,float,float)> fn = obj.sdf();

    NField3 field( &fn , wbb.min, wbb.max );
    LOG(info) << field.desc() ; 

    NGrid3 grid(3);
    LOG(info) << grid.desc() ; 

    NFieldGrid3 fg(&field, &grid);

    test_fieldgrid(field, grid, &fg);

    return 0 ; 
}
