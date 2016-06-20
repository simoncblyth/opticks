#include "InterpolatedView.hh"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    View* a = new View ;   a->setEye(1,0,0) ;
    View* b = new View ;   b->setEye(-1,0,0) ;
  
    InterpolatedView* iv = new InterpolatedView ; 
    iv->addView(a);
    iv->addView(b);

    glm::mat4 m2w ; 
    print(m2w, "m2w");    

    unsigned int n = 10 ; 
    for(unsigned int i=0 ; i < n ; i++)
    {
        float f = float(i)/float(n) ;
        iv->setFraction(f);
        std::cout << "f " << f ; 
        glm::vec4 e = iv->getEye(m2w);
        print(e, "eye");    
    }


    return 0 ; 
}
