#include <boost/math/special_functions/sign.hpp>

#include "NField3.hpp"
#include "NGrid3.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

#include "NBox.hpp"
#include "NBBox.hpp"

void test_cross_isosurface_0(const NField3& field, const NGrid3& grid)
{
    nivec3 ijk[2] = {{0,0,0},{0,0,0}} ; 
    nvec3  fpos[2];
    nvec3  fpos2[2];
    nvec3  pos[2];
    float sdf[2]; 

    int ncross = 0 ; 
    for(int loc=0 ; loc < grid.nloc - 1 ; loc++)
    {
        ijk[0] = loc > 0 ? ijk[1] : grid.ijk(loc) ;
        ijk[1] = grid.ijk(loc+1) ;
 
        fpos[0] = loc > 0 ? fpos[1] : grid.fpos(ijk[0]);
        fpos[1] = grid.fpos(ijk[1]);


        fpos2[0] = loc > 0 ? fpos2[1] : grid.fpos(loc);
        fpos2[1] = grid.fpos(loc+1);

        assert( fpos2[0] == fpos[0] );
        assert( fpos2[1] == fpos[1] );


        sdf[0] = loc > 0 ? sdf[1] : field(fpos[0]);
        sdf[1] = field(fpos[1]);
 
        pos[0] = loc > 0 ? pos[1] : field.pos(fpos[0]);
        pos[1] = field.pos(fpos[1]);

        bool cross_isosurface = boost::math::sign(sdf[0]) != boost::math::sign(sdf[1]) ;

        if(cross_isosurface)
        {
            ncross++ ; 
            std::cout << " loc " << std::setw(5) << loc 
                      << " ijk[0] " << ijk[0].desc() 
                      << " ijk[1] " << ijk[0].desc() 
                     // << " fpos[0] " << fpos[0].desc()
                     // << " fpos[1] " << fpos[1].desc()
                      << " pos[0] " << pos[0].desc()
                      << " pos[1] " << pos[1].desc()
                      << " sdf[0] " << sdf[0] 
                      << " sdf[1] " << sdf[1] 
                      << std::endl 
                      ; 
            } 
    }

    LOG(info) << "(test_cross_isosurface_0) ncross " << ncross ; 
}

void test_cross_isosurface_1(const NField3& field, const NGrid3& grid)
{
    int ncross = 0 ; 
    float sdf_prior = 0 ; 
    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        nvec3 fpos = grid.fpos(loc);
        float sdf = field(fpos);

        bool cross_isosurface = loc > 0 && boost::math::sign(sdf) != boost::math::sign(sdf_prior) ;
        if(cross_isosurface) ncross++ ; 

        sdf_prior = sdf ;
    }
    LOG(info) << "(test_cross_isosurface_1) ncross " << ncross ; 
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

    test_cross_isosurface_0(field, grid);
    test_cross_isosurface_1(field, grid);

    return 0 ; 
}
