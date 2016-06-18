
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iomanip>

#include <boost/algorithm/string/join.hpp>

#ifdef _MSC_VER
// members needs to have dll-interface to be used by clients
#pragma warning( disable : 4251 )
#endif

#include "NGLM.hpp"
#include "GLMPrint.hpp"

void assert_same(const char* msg, const glm::vec4& a, const glm::vec4& b)
{
    for(unsigned int k=0 ; k < 4 ; k++)
    {
        if(a[k] == b[k]) continue ; 

        char rep[128];
        snprintf(rep, 128, "GLMPrint.assert_same %s  [%u] %.3f %.3f \n", msg, k, a[k], b[k] );
        //assert(0 && rep ); 

        printf("%s", rep );
    }

}


void fdump(float* f, unsigned int n, const char* msg)
{
    if(!f) return ;

    printf("%s\n", msg);
    for(unsigned int i=0 ; i < n ; i++)
    {   
        if(i%4 == 0) printf("\n");
        printf(" %10.4f ", f[i] );
    }   
    printf("\n");
}




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


void print(const glm::vec4& v0, const char* msg0, const glm::vec4& v1, const char* msg1)
{
    const char* fmt = 
"%15s vec4  %10.3f %10.3f %10.3f %10.3f   "
"%15s vec4  %10.3f %10.3f %10.3f %10.3f \n" ;
    printf(fmt, 
           msg0, v0.x, v0.y, v0.z, v0.w,
           msg1, v1.x, v1.y, v1.z, v1.w
         );
}


void print(const glm::vec4& v, const char* tmpl, unsigned int incl)
{
    char msg[128];
    snprintf(msg, 128, tmpl, incl);
    printf("%15s vec4  %10.3f %10.3f %10.3f %10.3f \n", msg, v.x, v.y, v.z, v.w);
}




void print_i(const glm::ivec4& v, const char* msg)
{
    printf("%15s ivec4  %7d %7d %7d %7d \n", msg, v.x, v.y, v.z, v.w);
}

void print_u(const glm::uvec4& v, const char* msg)
{
    printf("%15s uvec4  %7u %7u %7u %7u \n", msg, v.x, v.y, v.z, v.w);
}

void print(const glm::vec3& v, const char* msg)
{
    printf("%15s vec3  %10.3f %10.3f %10.3f  \n", msg, v.x, v.y, v.z );
}

void print(float* f, const char* msg, unsigned int n)
{
    printf("%s", msg);
    for(unsigned int i=0 ; i < n ; i++)
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






