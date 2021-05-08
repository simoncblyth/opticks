#include <cassert>
#include "OPTICKS_LOG.hh"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"


int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : nullptr ; 

    OPTICKS_LOG(argc, argv); 

    if(path == nullptr) 
    {
        LOG(error) << " expecting path to jpg/png image " ; 
        return 1 ; 
    }


    assert( PLOG::instance ); 

    LOG(info) << " read " << path ; 

    SIMG img(path); 

    assert( img.ttf ); 

    img.annotate("UseSysRapSIMG"); 

    const char* dstpath = "/tmp/UseSysRapSIMG.jpg" ; 
    int quality = 50 ; 

    LOG(info) << " write " << dstpath ; 
    img.writeJPG(dstpath, quality); 

    return 0 ; 
}
