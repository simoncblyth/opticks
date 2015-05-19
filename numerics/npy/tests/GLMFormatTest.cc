
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>  

#include "GLMFormat.hpp"
#include "GLMPrint.hpp"

#include "stdio.h"
#include "assert.h"

int main()
{
    std::string sv = "1,2,3" ;
    glm::vec3 v = gvec3(sv);     
    print(v, "gvec3(sv)");

    std::string sq = "1,2,3,4" ;
    glm::quat q = gquat(sq);     
    print(q, "gquat(sq)");

    std::string sqq = gformat(q);
    printf("%s\n", sqq.c_str());

    glm::quat qq = gquat(sqq);
    print(qq, "qq:gquat(sqq)");

    assert( q == qq );


    return 0 ; 
}

