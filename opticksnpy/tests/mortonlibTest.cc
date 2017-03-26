#include "mortonlib/morton2d.h"
#include "mortonlib/morton3d.h"

#include "NPY_LOG.hh"
#include "NQuad.hpp"
#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "PLOG.hh"


void test_morton3d()
{
    uint32_t offsets[8][3] = {
        {0,0,0},
        {0,0,1},
        {0,1,0},
        {0,1,1},
        {1,0,0},
        {1,0,1},
        {1,1,0},
        {1,1,1}
    };

    for(int i=0 ; i < 8 ; i++)
    {
        nuvec3 p = make_nuvec3(offsets[i][0], offsets[i][1], offsets[i][2] );
        morton3 m = morton3d<uint64_t>(p.x, p.y, p.z); 
        morton3 m2 = morton3d<uint64_t>::morton3d_256(p.x, p.y, p.z); 
        LOG(info) << p.desc() << " " << m.key << " " << m2.key  ; 
    }

    for(int i=0 ; i < 8 ; i++)
    {
        ntvec3<uint64_t> p ;
        p.x = offsets[i][0] ;
        p.y = offsets[i][1] ;
        p.z = offsets[i][2] ;

        morton3 m = morton3d<uint64_t>(p.x, p.y, p.z); 
        morton3 m2 = morton3d<uint64_t>::morton3d_256(p.x, p.y, p.z); 
        LOG(info) << p.desc() << " " << m.key << " " << m2.key  ; 
    }

    for(unsigned loc=0 ; loc < 8 ; loc++)
    {
         morton3 m(loc);
         ntvec3<unsigned long long> p = {0,0,0} ;
         m.decode(p.x, p.y, p.z ); 
         LOG(info) << " loc " << loc << " --> " << p.desc()  ;
    }



    glm::tvec3<uint64_t> ijk ; 
    for(unsigned loc=0 ; loc < 8 ; loc++)
    {
         morton3 m(loc);
         m.decode(ijk.x, ijk.y, ijk.z ); 
         LOG(info) << " loc " << loc << " --> " << ijk.x << " " << ijk.y << " " << ijk.z   ;
    }





}

void test_morton2d()
{
    uint32_t offsets[4][2] = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    for(int i=0 ; i < 4 ; i++)
    {
        nuvec3 p = make_nuvec3(offsets[i][0], offsets[i][1], 0u );

        morton2 m = morton2d<uint64_t>(p.x, p.y ); 

        LOG(info) << p.desc() << " " << m.key ;
    }
}





int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    test_morton3d();
    test_morton2d();

    return 0 ; 
}
