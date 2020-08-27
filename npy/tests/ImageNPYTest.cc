#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "SStr.hh"
#include "ImageNPY.hpp"

/**
ImageNPYTest
=============

See ImageNPYTest.py for imshow plotting 


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


    ~/env/bin/img.py ~/opticks_refs/Earth_Albedo_8192_4096.jpg --saveppm --scale 8 


    (base) epsilon:npy blyth$ ImageNPYTest $HOME/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm
    2020-08-21 19:43:31.507 INFO  [11622457] [*test_LoadPPM@33]  path /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm yflip 0 ncomp 3 config add_border,add_midline,add_quadline
    2020-08-21 19:43:31.509 INFO  [11622457] [*ImageNPY::LoadPPM@52]  path /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm width 1024 height 512 mode 6 bits 255
    2020-08-21 19:43:31.543 INFO  [11622457] [*test_LoadPPM@43]  array 512,1024,3
    2020-08-21 19:43:31.543 INFO  [11622457] [*test_LoadPPM@46] saving array to /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.npy
    2020-08-21 19:43:31.548 INFO  [11622457] [ImageNPY::SavePPMImp@102]  path /tmp/SPPMTest2.ppm width 1024 height 512 ncomp 3 yflip 0
    2020-08-21 19:43:31.548 INFO  [11622457] [ImageNPY::SavePPMImp@111]  write to /tmp/SPPMTest2.ppm
    (base) epsilon:npy blyth$ du -hs $HOME/opticks_refs/Earth_Albedo_8192_4096_scaled_8*
    1.5M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.npy
    1.5M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm
    (base) epsilon:npy blyth$ 

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


NPY<unsigned char>*  test_LoadPPMConcat(const char* path, const bool yflip, const unsigned num_concat)
{
    const unsigned ncomp = 3 ; 
    const char* config = "add_border,add_midline,add_quadline" ;  

    std::vector<std::string> paths ; 
    for(int i=0 ; i < num_concat ; i++) paths.push_back(path); 

    LOG(info) 
         << " num_concat " << num_concat
         << " path " << path 
         << " yflip " << yflip
         << " ncomp " << ncomp 
         << " config " << config
         ; 

    NPY<unsigned char>* a = ImageNPY::LoadPPMConcat(paths, yflip, ncomp, config) ; 

    LOG(info) << " array " << a->getShapeString() ; 

    const char* opath = SStr::ReplaceEnd(path, ".ppm", "_concat.npy" ) ;
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
    const char* path  = argc > 1 ? argv[1] : "/tmp/SPPMTest.ppm" ; 
    const char* path2 = argc > 2 ? argv[2] : "/tmp/SPPMTest2.ppm" ; 

    bool yflip0 = false ; 
    NPY<unsigned char>* a = test_LoadPPM(path, yflip0); 

    //unsigned num_concat = 3 ; 
    //test_LoadPPMConcat(path, yflip0, num_concat );  

    bool yflip1 = false ;  
    test_SavePPM(path2, a, yflip1); 


    return 0 ; 
}

