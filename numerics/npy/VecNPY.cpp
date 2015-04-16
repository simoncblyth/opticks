#include "VecNPY.hpp"

#include "NPY.hpp"
#include "float.h"
#include "GLMPrint.hpp"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>


VecNPY::VecNPY(NPY* npy, unsigned int j, unsigned int k) :
            m_bytes(npy->getBytes()),
            m_numbytes(npy->getNumBytes(0)),
            m_stride(npy->getNumBytes(1)),
            m_offset(npy->getByteIndex(0,j,k)),
            m_count(npy->getShape(0)),
            m_low(NULL),
            m_high(NULL),
            m_dimensions(NULL),
            m_center(NULL),
            m_model_to_world(),
            m_extent(0.f)
{
    findBounds();
}

glm::mat4& VecNPY::getModelToWorld()
{
    return m_model_to_world ; 
}
float* VecNPY::getModelToWorldPtr()
{
    return glm::value_ptr(m_model_to_world) ; 
}
float VecNPY::getExtent()
{
    return m_extent ; 
}


void VecNPY::dump(const char* msg)
{
    float xx[4] = { FLT_MAX, -FLT_MAX, 0.f, 0.f};
    float yy[4] = { FLT_MAX, -FLT_MAX, 0.f, 0.f};
    float zz[4] = { FLT_MAX, -FLT_MAX, 0.f, 0.f};

    const char* fmt = "VecNPY::dump %5s %6u/%6u :  %15f %15f %15f \n";

    for(unsigned int i=0 ; i < m_count ; ++i )
    {   
        char* ptr = (char*)m_bytes + m_offset + i*m_stride  ;   
        float* f = (float*)ptr ; 
        float x(*(f+0));
        float y(*(f+1));
        float z(*(f+2));

        if( x<xx[0] ) xx[0] = x ;  
        if( x>xx[1] ) xx[1] = x ;  

        if( y<yy[0] ) yy[0] = y ;  
        if( y>yy[1] ) yy[1] = y ;  

        if( z<zz[0] ) zz[0] = z ;  
        if( z>zz[1] ) zz[1] = z ;  

        if(i < 5 || i > m_count - 5) printf(fmt, "", i,m_count, x, y, z);
    }

    xx[2] = xx[1] - xx[0] ;
    yy[2] = yy[1] - yy[0] ;
    zz[2] = zz[1] - zz[0] ;

    xx[3] = (xx[1] + xx[0])/2.f ;
    yy[3] = (yy[1] + yy[0])/2.f ;
    zz[3] = (zz[1] + zz[0])/2.f ;

    printf(fmt, "min", 0,0,xx[0],yy[0],zz[0]);
    printf(fmt, "max", 0,0,xx[1],yy[1],zz[1]);
    printf(fmt, "dif", 0,0,xx[2],yy[2],zz[2]);
    printf(fmt, "cen", 0,0,xx[3],yy[3],zz[3]);
}


void VecNPY::findBounds()
{
    glm::vec3 lo( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for(unsigned int i=0 ; i < m_count ; ++i )
    {   
        char* ptr = (char*)m_bytes + m_offset + i*m_stride  ;   
        float* f = (float*)ptr ; 

        glm::vec3 v(*(f+0),*(f+1),*(f+2));

        lo.x = std::min( lo.x, v.x);
        lo.y = std::min( lo.y, v.y);
        lo.z = std::min( lo.z, v.z);

        hi.x = std::max( hi.x, v.x);
        hi.y = std::max( hi.y, v.y);
        hi.z = std::max( hi.z, v.z);
    }

    m_low = new glm::vec3(lo.x, lo.y, lo.z);
    m_high = new glm::vec3(hi.x, hi.y, hi.z);

    m_dimensions = new glm::vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z );
    m_center     = new glm::vec3((hi.x + lo.x)/2.0f, (hi.y + lo.y)/2.0f , (hi.z + lo.z)/2.0f );

    m_extent = 0.f ;
    m_extent = std::max( m_dimensions->x , m_extent );
    m_extent = std::max( m_dimensions->y , m_extent );
    m_extent = std::max( m_dimensions->z , m_extent );
    m_extent = m_extent / 2.0f ;    

    glm::vec3 s(m_extent);
    glm::vec3 t(*m_center);

    m_model_to_world = glm::scale(glm::translate(glm::mat4(1.0), t), s); 

    Summary("VecNPY::findBounds");
}


void VecNPY::Summary(const char* msg)
{
    printf("%s\n", msg);
    if(!m_low) return ;

    print(*m_low,  "m_low");
    print(*m_high, "m_high");
    print(*m_dimensions, "m_dimensions");
    print(*m_center,     "m_center");
    print(m_model_to_world, "m_model_to_world");
    print(glm::value_ptr(m_model_to_world), "glm::value_ptr(m_model_to_world)");
}




