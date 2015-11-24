
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>  

#include "GLMFormat.hpp"
#include "GLMPrint.hpp"

#include "stdio.h"
#include "assert.h"

#include <vector>


void test_gmat4()
{
    std::string s = "0.500,-0.866,0.000,-86.603,0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000," ;
    glm::mat4 m = gmat4(s);
    print(m, "mat4");
}



void test_ivec4()
{
    std::vector<std::string> ss ; 

    //ss.push_back("");     // bad-lexical case TODO: trap this
    ss.push_back("1");    
    ss.push_back("1,2");    
    ss.push_back("1,2,3");    
    ss.push_back("1,2,3,4");    
    ss.push_back("1,2,3,4,5");    

    for(unsigned int i=0 ; i < ss.size() ; i++)
    {
         std::string s = ss[i];
         glm::ivec4 v = givec4(s);
         print(v, s.c_str());
    }
}


void test_misc()
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
}


int main()
{
    //test_ivec4();
    test_gmat4();
    return 0 ; 
}

