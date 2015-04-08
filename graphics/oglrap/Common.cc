#include "Common.hh"
#include "stdio.h"

#include <glm/glm.hpp>
#include  <glm/gtc/matrix_transform.hpp>  
#include  <glm/gtc/quaternion.hpp>  


void print(const glm::mat4& m, const char* msg)
{
    printf("%s\n", msg);

    for (int j=0; j<4; j++)
    {
        for (int i=0; i<4; i++) printf("%10.3f ",m[i][j]);
        printf("\n");
    }
}



void print(const glm::quat& q, const char* msg)
{
    printf("%15s  quat{ %10.3f : %10.3f %10.3f %10.3f } \n", msg, q.w, q.x, q.y, q.z);
}


void print(const glm::vec4& v, const char* msg)
{
    printf("%15s   %10.3f %10.3f %10.3f %10.3f \n", msg, v.x, v.y, v.z, v.w);
}

void print( const glm::vec4& a, const glm::vec4& b, const glm::vec4& c, const char* msg)
{
    printf("%15s  "
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


