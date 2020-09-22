#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "SStr.hh"
#include "ImageNPY.hpp"

/**
ImageNPYTest
=============

1. loads a ppm image from file "*.ppm" 
2. changes pixel colors to give: red border, green midline(equator) and blue quadlines.
3. saved the changed image to file  name "*_ImageNPYTest_annoted.ppm"

See ImageNPYTest.py for imshow plotting 

Example::

    epsilon:tests blyth$ ImageNPYTest $HOME/opticks_refs/Earth_Albedo_8192_4096.ppm
    2020-09-22 12:12:26.961 INFO  [27462190] [main@95]  load ipath /Users/blyth/opticks_refs/Earth_Albedo_8192_4096.ppm
    2020-09-22 12:12:26.961 INFO  [27462190] [*test_LoadPPM@51]  path /Users/blyth/opticks_refs/Earth_Albedo_8192_4096.ppm yflip 0 ncomp 3 config add_border,add_midline,add_quadline
    2020-09-22 12:12:28.989 INFO  [27462190] [*test_LoadPPM@62]  array 4096,8192,3
    2020-09-22 12:12:28.989 INFO  [27462190] [*test_LoadPPM@65] saving array to /Users/blyth/opticks_refs/Earth_Albedo_8192_4096.npy
    2020-09-22 12:12:29.125 INFO  [27462190] [main@100]  save to opath /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_ImageNPYTest_annotated.ppm

    epsilon:tests blyth$ du -h /Users/blyth/opticks_refs/Earth_Albedo_8192_4096*
    4.2M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096.jpg
     96M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096.npy
     96M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096.ppm
     96M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096_ImageNPYTest_annotated.ppm


Use env/bin/img.py to creating a scaled down PPM from a JPG 
-------------------------------------------------------------

::

    ~/env/bin/img.py ~/opticks_refs/Earth_Albedo_8192_4096.jpg --saveppm --scale 8 

    epsilon:opticks blyth$ ImageNPYTest $HOME/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm
    2020-09-22 12:22:00.905 INFO  [27597017] [main@93]  load ipath /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm
    2020-09-22 12:22:00.907 INFO  [27597017] [*test_LoadPPM@59]  path /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm yflip 0 ncomp 3 config add_border,add_midline,add_quadline
    2020-09-22 12:22:00.953 INFO  [27597017] [*test_LoadPPM@70]  array 512,1024,3
    2020-09-22 12:22:00.953 INFO  [27597017] [*test_LoadPPM@73] saving array to /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.npy
    2020-09-22 12:22:00.956 INFO  [27597017] [main@97]  save to opath /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8_ImageNPYTest_annotated.ppm

    epsilon:opticks blyth$ du -hs $HOME/opticks_refs/Earth_Albedo_8192_4096_scaled_8*
    2.1M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.npy
    1.5M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm
    1.5M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8_ImageNPYTest_annotated.ppm

**/


NPY<unsigned char>*  test_LoadPPM(const char* path, const bool yflip)
{
    const unsigned ncomp = 3 ; 
    const char* config = "add_border,add_midline,add_quadline" ;  

    LOG(info) 
         << " path " << path 
         << " yflip " << yflip
         << " ncomp " << ncomp 
         << " config " << config
         ; 

    bool layer_dimension = false ; 
    NPY<unsigned char>* a = ImageNPY::LoadPPM(path, yflip, ncomp, config, layer_dimension ) ; 
    //a->dump();   // dumping of unsigned char array gives mess 

    LOG(info) << " array " << a->getShapeString() ; 

    const char* opath = SStr::ReplaceEnd(path, ".ppm", ".npy" ) ;
    LOG(info) << "saving array to " << opath ; 

    a->save(opath); 

    return a ; 
}

void test_SavePPM(const char* path, NPY<unsigned char>* a, bool yflip)
{
    ImageNPY::SavePPM(path, a, yflip); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* ipath = argc > 1 ? argv[1] : "/tmp/SPPMTest.ppm" ; 
    const char* opath = argc > 2 ? argv[2] : SStr::ReplaceEnd(ipath, ".ppm", "_ImageNPYTest_annotated.ppm" ) ; 

    LOG(info) << " load ipath " << ipath ; 
    bool yflip0 = false ; 
    NPY<unsigned char>* a = test_LoadPPM(ipath, yflip0); 

    LOG(info) << " save to opath " << opath ; 
    bool yflip1 = false ;  
    test_SavePPM(opath, a, yflip1); 

    return 0 ; 
}

