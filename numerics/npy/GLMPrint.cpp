#include "GLMPrint.hpp"

#include "stdio.h"
#include "float.h"
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/quaternion.hpp>  
#include <glm/gtc/type_ptr.hpp>

void print(const glm::mat4& m, const char* msg)
{
    printf("%s mat4\n", msg);

    for (int j=0; j<4; j++)
    {
        for (int i=0; i<4; i++) printf("%10.3f ",m[i][j]);
        printf("\n");
    }
}

void minmax(glm::mat4& m, float& mn, float& mx) 
{
    float* f = glm::value_ptr(m);
    for(unsigned int i=0 ; i < 16 ; i++)
    {   
         float v = *(f+i);
         if(v>mx) mx = v ; 
         if(v<mn) mn = v ; 
         printf(" %2d %f \n", i, v );
    }   
}

float absmax(glm::mat4& m)
{
    float mn(FLT_MAX);
    float mx(-FLT_MIN);
    minmax(m, mn, mx);
    return std::max( fabs(mn), fabs(mx));
}


void print(const glm::quat& q, const char* msg)
{
    printf("%15s quat %15.6f : %15.6f %15.6f %15.6f  \n", msg, q.w, q.x, q.y, q.z);
}


void print(const glm::vec4& v, const char* msg)
{
    printf("%15s vec4  %10.3f %10.3f %10.3f %10.3f \n", msg, v.x, v.y, v.z, v.w);
}


void print(const glm::vec3& v, const char* msg)
{
    printf("%15s vec3  %10.3f %10.3f %10.3f  \n", msg, v.x, v.y, v.z );
}

void print(float* f, const char* msg)
{
    printf("%s", msg);
    for(unsigned int i=0 ; i < 16 ; i++)
    {   
        if(i % 4 == 0) printf("\n"); 
        printf(" %15.3f ", *(f+i)) ;
    }   
    printf("\n");
}



void print( const glm::vec4& a, const glm::vec4& b, const glm::vec4& c, const char* msg)
{
    printf("%15s vec4*3 "
      "%10.3f %10.3f %10.3f %10.3f    "
      "%10.3f %10.3f %10.3f %10.3f    "
      "%10.3f %10.3f %10.3f %10.3f    "
      " \n", 
       msg, 
       a.x, a.y, a.z, a.w,
       b.x, b.y, b.z, b.w,
       c.x, c.y, c.z, c.w
     );
}


void print( const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d, const char* msg)
{
    printf("%15s vec3*4 "
      "%10.3f %10.3f %10.3f     "
      "%10.3f %10.3f %10.3f     "
      "%10.3f %10.3f %10.3f     "
      "%10.3f %10.3f %10.3f     "
      " \n", 
       msg, 
       a.x, a.y, a.z,
       b.x, b.y, b.z,
       c.x, c.y, c.z,
       d.x, d.y, d.z
     );
}






