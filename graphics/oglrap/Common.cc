#include "Common.hh"
#include "stdio.h"

#include <glm/glm.hpp>
#include  <glm/gtc/matrix_transform.hpp>  


void print(const glm::mat4& m, const char* msg)
{
    printf("%s\n", msg);

    for (int j=0; j<4; j++)
    {
        for (int i=0; i<4; i++) printf("%10.3f ",m[i][j]);
        printf("\n");
    }
}


void print(const glm::vec4& v, const char* msg)
{
    printf("%15s   %10.3f %10.3f %10.3f %10.3f \n", msg, v.x, v.y, v.z, v.w);
}


