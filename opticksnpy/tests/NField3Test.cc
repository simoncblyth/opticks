#include <bitset>
#include <boost/math/special_functions/sign.hpp>

#include "NField3.hpp"
#include "NGrid3.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

#include "NGLM.hpp"
#include "NBox.hpp"
#include "NBBox.hpp"


typedef NField<glm::vec3,glm::ivec3,3> F3 ; 
typedef NGrid<glm::vec3,glm::ivec3,3> G3 ; 

template struct NGrid<glm::vec3, glm::ivec3, 3> ; 


void test_cross_isosurface_0(const F3& field, const G3& grid)
{
    glm::ivec3 ijk[2] = {{0,0,0},{0,0,0}} ; 
    glm::vec3  fpos[2];
    glm::vec3  fpos2[2];
    glm::vec3  pos[2];
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
 
        pos[0] = loc > 0 ? pos[1] : field.position(fpos[0]);
        pos[1] = field.position(fpos[1]);

        bool cross_isosurface = boost::math::sign(sdf[0]) != boost::math::sign(sdf[1]) ;

        if(cross_isosurface)
        {
            ncross++ ; 
            std::cout << " loc " << std::setw(5) << loc 
                      << " ijk[0] " << glm::to_string(ijk[0]) 
                      << " ijk[1] " << glm::to_string(ijk[0]) 
                      << " pos[0] " << glm::to_string(pos[0])
                      << " pos[1] " << glm::to_string(pos[1])
                      << " sdf[0] " << sdf[0] 
                      << " sdf[1] " << sdf[1] 
                      << std::endl 
                      ; 
            } 
    }

    LOG(info) << "(test_cross_isosurface_0) ncross " << ncross ; 
}

void test_cross_isosurface_1(const F3& field, const G3& grid)
{
    int ncross = 0 ; 
    float sdf_prior = 0 ; 
    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        glm::vec3 fpos = grid.fpos(loc);
        float sdf = field(fpos);

        bool cross_isosurface = loc > 0 && boost::math::sign(sdf) != boost::math::sign(sdf_prior) ;
        if(cross_isosurface) ncross++ ; 

        sdf_prior = sdf ;
    }
    LOG(info) << "(test_cross_isosurface_1) ncross " << ncross ; 
}


void test_zcorners( const F3& field, const G3& grid)
{
    glm::vec3 fpos[3] = {{0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}, {1.f, 1.f, 1.f}} ;
    for(int i=0 ; i < 3 ; i++) field.zcorners( fpos[i] , grid.elem ); 
}

void test_cross_isosurface_2(const F3& field, const G3& grid)
{
    int ncross = 0 ; 
    float sdf_prior = 0 ; 
    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        glm::vec3 fpos = grid.fpos(loc);
        float sdf = field(fpos);

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
            ;
        }
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

    glm::vec3 wbb_min(wbb.min.x, wbb.min.y, wbb.min.z );
    glm::vec3 wbb_max(wbb.max.x, wbb.max.y, wbb.max.z );


    F3 field( &fn , wbb_min, wbb_max );
    LOG(info) << field.desc() ; 

    G3 grid(3);
    LOG(info) << grid.desc() ; 

    test_cross_isosurface_2(field, grid);

    return 0 ; 
}
