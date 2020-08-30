#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "SStr.hh"
#include "ImageNPY.hpp"

/**
ImageNPYConcatTest
======================

See ImageNPYConcatTest.py for imshow plotting 

**/


NPY<unsigned char>*  test_LoadPPMConcat(const char* path, const bool yflip, const unsigned num_concat)
{
    const unsigned ncomp = 3 ; 
    const char* config0 = "add_border" ;  
    const char* config1 = "add_midline" ; 
    const char* config2 = "add_quadline" ;  

    std::vector<std::string> paths ; 
    std::vector<std::string> configs ; 
    for(unsigned i=0 ; i < num_concat ; i++) paths.push_back(path); 
    for(unsigned i=0 ; i < num_concat ; i++) configs.push_back( i % 3 == 0 ? config0 : ( i % 3 == 1 ? config1  : config2 ) ); 

    LOG(info) 
         << " num_concat " << num_concat
         << " path " << path 
         << " yflip " << yflip
         << " ncomp " << ncomp 
         << " config0 " << config0
         << " config1 " << config1
         ; 

    NPY<unsigned char>* a = ImageNPY::LoadPPMConcat(paths, configs, yflip, ncomp) ; 

    LOG(info) << " array " << a->getShapeString() ; 

    const char* opath = SStr::ReplaceEnd(path, ".ppm", "_concat.npy" ) ;
    LOG(info) << "saving array to " << opath ; 

    a->save(opath); 

    return a ; 
}


void test_SavePPMConcat( const NPY<unsigned char>* imgs, const char* basepath, bool yflip)
{
    ImageNPY::SavePPMConcat(imgs, basepath, yflip);  
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    const char* path  = argc > 1 ? argv[1] : "/tmp/SPPMTest_MakeTestImage.ppm" ; 

    bool yflip0 = false ; 
    unsigned num_concat = 3 ; 
    NPY<unsigned char>* imgs = test_LoadPPMConcat(path, yflip0, num_concat );  
    LOG(info) << " imgs " << imgs->getShapeString(); 

    test_SavePPMConcat( imgs, path, yflip0 ); 

    return 0 ; 
}

