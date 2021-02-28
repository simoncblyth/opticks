// name=SIMGTest ; stb- ; gcc $name.cc -lstdc++ -std=c++11 -I$(stb-dir) -I. -o /tmp/$name && /tmp/$name

#include <iostream>
#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : nullptr ; 
    if(path == nullptr) 
    {
        std::cout << argv[0] << " : a single argument with the path to an image file is required " << std::endl ; 
        return 0 ; 
    }    

    SIMG img(path); 
    std::cout << img.desc() << std::endl ; 

    img.writePNG(); 
    img.writeJPG(100); 
    img.writeJPG(50); 
    img.writeJPG(10); 
    img.writeJPG(5); 
  
    return 0 ; 
}

