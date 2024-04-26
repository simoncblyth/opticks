/**
SIMGTest.cc : Loads PNG/JPG image from provided path and saves it in PNG and JPG with variety of quality settings 
=====================================================================================================================

The standardly build executable can be run with::

    SIMGTest /tmp/flower.jpg

For non-CMake build and run use::

    IMGPATH=/tmp/flower.jpg ~/o/sysrap/tests/SIMGTest.sh 

**/

#include <sstream>
#include <iostream>
#define SIMG_IMPLEMENTATION 1 
#include "SIMG.h"

#include "s_time.h"

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

    std::stringstream ss ;
    ss << "SIMGTest " << s_time::Format() ; 
    std::string s = ss.str(); 

    img.annotate(s.c_str()); 

    img.writePNG(); 
    img.writeJPG(100); 
    img.writeJPG(50); 
    img.writeJPG(10); 
    img.writeJPG(5); 
  
    return 0 ; 
}

