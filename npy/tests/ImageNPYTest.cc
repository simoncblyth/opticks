#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "SStr.hh"
#include "ImageNPY.hpp"

/**
ImageNPYTest
=============


::

    (base) epsilon:npy blyth$ ImageNPYTest $HOME/opticks_refs/Earth_Albedo_8192_4096.ppm
    2020-08-19 13:56:33.125 INFO  [9900421] [main@34] ImageNPY::LoadPPM from /Users/blyth/opticks_refs/Earth_Albedo_8192_4096.ppm
    2020-08-19 13:56:33.126 INFO  [9900421] [*ImageNPY::LoadPPM@38]  path /Users/blyth/opticks_refs/Earth_Albedo_8192_4096.ppm width 8192 height 4096 mode 6 bits 255
    2020-08-19 13:56:34.530 INFO  [9900421] [main@38]  array 4096,8192,3
    2020-08-19 13:56:34.531 INFO  [9900421] [main@41] saving array to /Users/blyth/opticks_refs/Earth_Albedo_8192_4096.npy
    (base) epsilon:npy blyth$ 
    (base) epsilon:npy blyth$ du -h /Users/blyth/opticks_refs/Earth_Albedo_8192_4096*
    4.2M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096.jpg
     96M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096.npy
     96M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096.ppm
    (base) epsilon:npy blyth$ 

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* path = argc > 1 ? argv[1] : "/tmp/SPPMTest.ppm" ; 
    const bool yflip = false ; 
    const unsigned ncomp = 3 ; 

    LOG(info) << "ImageNPY::LoadPPM from " << path ; 
    NPY<unsigned char>* a = ImageNPY::LoadPPM(path, yflip, ncomp ) ; 
    //a->dump();   // dumping of unsigned char array gives mess 

    LOG(info) << " array " << a->getShapeString() ; 

    const char* opath = SStr::ReplaceEnd(path, ".ppm", ".npy" ) ;
    LOG(info) << "saving array to " << opath ; 

    a->save(opath); 

    return 0 ; 
}

