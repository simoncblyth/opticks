#include <cassert>

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.h"


int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : nullptr ; 

    if(path == nullptr) 
    {
        std::cerr << "UseSysRapSIMG : expecting path to jpg/png image \n" ; 
        return 1 ; 
    }


    SIMG img(path); 

    img.annotate("UseSysRapSIMG"); 

    const char* dstpath = "/tmp/UseSysRapSIMG.jpg" ; 
    int quality = 50 ; 

    std::cout << " write " << dstpath << "\n" ; 
    img.writeJPG(dstpath, quality); 

    return 0 ; 
}
