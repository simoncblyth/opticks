//#include <sstream>
#include <cstring>

#include "NGLM.hpp"
#include "NBBox.hpp"


nbbox nbbox::transform( const glm::mat4& t )
{
    nbbox tbb ; 
    transform( tbb, *this, t );
    return tbb ; 
}

void nbbox::transform_brute(nbbox& tbb, const nbbox& bb, const glm::mat4& t )
{
    glm::vec4 min(bb.min.x, bb.min.y, bb.min.z , 1.f);
    glm::vec4 max(bb.max.x, bb.max.y, bb.max.z , 1.f);
    glm::vec4 dif(max - min);

    glm::vec4 t_min(FLT_MAX) ; 
    glm::vec4 t_max(-FLT_MAX) ; 

    for(int i=0 ; i < 8 ; i++) // over corners in Morton z-order
    {
        glm::vec4 corner(min.x + ( i & 1 ? dif.x : 0 ), min.y + ( i & 2 ? dif.y : 0 ), min.z + ( i & 4 ? dif.z : 0 ), 1.f ); 
        glm::vec4 t_corner = t * corner ; 
        gminf( t_min, t_min, t_corner );
        gmaxf( t_max, t_max, t_corner );
    }

    tbb.min = { t_min.x , t_min.y, t_min.z } ;
    tbb.max = { t_max.x , t_max.y, t_max.z } ;
    tbb.side = tbb.max - tbb.min ;
}


void nbbox::transform(nbbox& tbb, const nbbox& bb, const glm::mat4& t )
{
    // http://dev.theomader.com/transform-bounding-boxes/

    glm::vec4 xa = t[0] * bb.min.x ; 
    glm::vec4 xb = t[0] * bb.max.x ; 
    glm::vec4 xmi = gminf(xa, xb );      
    glm::vec4 xma = gmaxf(xa, xb );      

    glm::vec4 ya = t[1] * bb.min.y ; 
    glm::vec4 yb = t[1] * bb.max.y ; 
    glm::vec4 ymi = gminf(ya, yb );      
    glm::vec4 yma = gmaxf(ya, yb );      

    glm::vec4 za = t[2] * bb.min.z ; 
    glm::vec4 zb = t[2] * bb.max.z ; 
    glm::vec4 zmi = gminf(za, zb );      
    glm::vec4 zma = gmaxf(za, zb );      
     
    glm::vec4 t_min = xmi + ymi + zmi + t[3] ; 
    glm::vec4 t_max = xma + yma + zma + t[3] ; 

    tbb.min = { t_min.x , t_min.y, t_min.z } ;
    tbb.max = { t_max.x , t_max.y, t_max.z } ;
    tbb.side = tbb.max - tbb.min ;
}







const char* nbbox::desc() const
{
    char _desc[128];
    snprintf(_desc, 128, " mi %.32s mx %.32s ", min.desc(), max.desc() );
    return strdup(_desc);
}


void nbbox::dump(const char* msg)
{
    printf("%s\n", msg);
    min.dump("bb min");
    max.dump("bb max");
}

void nbbox::include(const nbbox& other)
{
    min = nminf( min, other.min );
    max = nmaxf( max, other.max );
}


