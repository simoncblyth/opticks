#include <iostream>
#include <GL/glew.h>

int main()
{

#ifdef GL_VERSION_1_1
    std::cout << "GL_VERSION_1_1" << std::endl ; 
#endif

#ifdef GL_VERSION_2_0
    std::cout << "GL_VERSION_2_0" << std::endl ; 
#endif

#ifdef GL_VERSION_3_0
    std::cout << "GL_VERSION_3_0" << std::endl ; 
#endif

#ifdef GL_VERSION_4_0
    std::cout << "GL_VERSION_4_0" << std::endl ; 
#endif

#ifdef GL_VERSION_4_5
    std::cout << "GL_VERSION_4_5" << std::endl ; 
#endif


    // all these are defined on mac, although only OpenGL upto 4.1(?) is supported

    return 0 ; 
}
