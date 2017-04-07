#include <bitset>
#include <boost/math/special_functions/sign.hpp>

#include "NGLMExt.hpp"

#include "NField3.hpp"
#include "NGrid3.hpp"
#include "NFieldGrid3.hpp"

#include "DualContouringSample/FGLite.h"


#include "NPY_LOG.hh"
#include "PLOG.hh"

#include "NBox.hpp"
#include "NBBox.hpp"


typedef NField<glm::vec3, glm::ivec3, 3> F3 ; 
typedef NGrid<glm::vec3, glm::ivec3, 3> G3 ; 
typedef NFieldGrid3<glm::vec3, glm::ivec3> FG3 ; 


void test_fieldgrid(const F3& field, const G3& grid, FG3* fg, FGLite* fgl)
{
    int ncross = 0 ; 
    float sdf_prior = 0 ; 

    bool debug = false ; 

    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        glm::vec3 fpos = grid.fpos(loc);
        float sdf = field(fpos);

        glm::ivec3 ijk = grid.ijk(loc);
        float sdf2 = fg->value(ijk);
        assert( sdf == sdf2 );

        glm::vec3 pfg = fg->position_f(ijk, debug);
        glm::vec3 pfgl = fgl->position_f( ijk );
        float sdf3 = fgl->value_f( ijk );

        assert( sdf == sdf3 ); 
        assert( pfg == pfgl );


        glm::vec3 ijk_f = ijk ; 
        ijk_f.x += 0.5f ; 
        ijk_f.y += 0.5f ; 
        ijk_f.z += 0.5f ; 

        glm::vec3 pfg_f = fg->position_f(ijk_f, debug);

        glm::vec3 pfgl_f = fgl->position_f( ijk_f );


        bool cross_isosurface = loc > 0 && boost::math::sign(sdf) != boost::math::sign(sdf_prior) ;
        if(cross_isosurface) 
        {
            ncross++ ; 
            int zc = field.zcorners( fpos, grid.elem ); 
            std::cout 
                << " zcorners " << std::setw(3) << zc
                << " 0x" << std::setfill('0') << std::setw(2) << std::hex << zc
                << " 0b" << std::bitset<8>(zc) 
                << std::dec 
                << std::setfill(' ')
                << " sdf " << std::setw(8) << sdf 
                << " sdf2 " << std::setw(8) << sdf2
                << " sdf3 " << std::setw(8) << sdf3
                << " ijk " << ijk 
                //<< " pfg " << pfg
                //<< " pfgl " << pfgl
                << " pfg_f " << pfg_f
                << " pfgl_f " << pfgl_f
                << std::endl 
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


    int level = 3 ; 


    nbox world = make_nbox(0,0,0,11) ; 
    nbbox wbb = world.bbox() ;

    nbox obj = make_nbox(0,0,0,7) ; 

    std::function<float(float,float,float)> fn = obj.sdf();


    glm::vec3 wbb_min(wbb.min.x, wbb.min.y, wbb.min.z);
    glm::vec3 wbb_max(wbb.max.x, wbb.max.y, wbb.max.z);


    std::cout << "wbb_min " << wbb_min << std::endl ; 
    std::cout << "wbb_max " << wbb_max << std::endl ; 



    F3 field( &fn , wbb_min, wbb_max );
    LOG(info) << field.desc() ; 

    G3 grid(level);
    LOG(info) << grid.desc() ; 

    FG3 fg(&field, &grid);



    int resolution = 1 << level ;

    FGLite fgl ; 
    fgl.func = &fn ; 

    fgl.resolution = resolution ; 
    fgl.offset = glm::ivec3(0,0,0) ;
    fgl.elem_offset = -0.5f ; // for centered  
    fgl.elem.x = wbb.side.x/resolution ;
    fgl.elem.y = wbb.side.y/resolution ;
    fgl.elem.z = wbb.side.z/resolution ;

    fgl.min = wbb_min ;  
    fgl.max = wbb_max ;  


    test_fieldgrid(field, grid, &fg, &fgl);



    return 0 ; 
}
