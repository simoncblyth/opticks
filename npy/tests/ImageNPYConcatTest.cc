#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "SStr.hh"
#include "SPath.hh"
#include "ImageNPY.hpp"

/**
ImageNPYConcatTest
======================

See ImageNPYConcatTest.py for imshow plotting 

**/

/**
test_LoadPPMConcat
-------------------

Mockup of the concat of three different image paths into a single array,
but repeatedly uses a single path for testing convenience. 
To visually distinguish the 3 "layers" a different config 
is applied to each. 

The ImageNPY::LoadPPMConcat is done twice with old_concat true and false
and the resulting arrays are compared element by element using NPY<unsigned char>::compare.

**/

NPY<unsigned char>*  test_LoadPPMConcat(const char* path, const bool yflip, const unsigned num_concat)
{
    LOG(info) << "[" ; 
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

    bool old_concat ;
    old_concat = true ;   
    NPY<unsigned char>* a = ImageNPY::LoadPPMConcat(paths, configs, yflip, ncomp, old_concat) ;

    old_concat = false ;   
    NPY<unsigned char>* b = ImageNPY::LoadPPMConcat(paths, configs, yflip, ncomp, old_concat) ; 

    LOG(info) << " array " << a->getShapeString() ; 
    LOG(info) << " array " << b->getShapeString() ; 

    bool dump = false ;   // dumping causes funny char problems with opticks-tl
    unsigned char epsilon = 0 ; 
    unsigned diffs = NPY<unsigned char>::compare(a,b,epsilon, dump);  
    LOG(info) << " diffs " ;  

    const char* a_path = SStr::ReplaceEnd(path, ".ppm", "_old_concat.npy" ) ;
    LOG(info) << "saving array to " << a_path ; 
    a->save(a_path); 

    const char* b_path = SStr::ReplaceEnd(path, ".ppm", "_new_concat.npy" ) ;
    LOG(info) << "saving array to " << b_path ; 
    b->save(b_path); 

    assert( diffs == 0 ); 

    LOG(info) << "]" ; 
    return a ; 
}


void test_SavePPMConcat( const NPY<unsigned char>* imgs, const char* basepath, bool yflip)
{
    ImageNPY::SavePPMConcat(imgs, basepath, yflip);  
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    const char* path  = argc > 1 ? argv[1] : SPath::Resolve("$TMP/SPPMTest_MakeTestImage.ppm") ; 

    bool yflip0 = false ; 
    unsigned num_concat = 3 ; 
    NPY<unsigned char>* imgs = test_LoadPPMConcat(path, yflip0, num_concat );  
    LOG(info) << " imgs " << imgs->getShapeString(); 

    test_SavePPMConcat( imgs, path, yflip0 ); 

    return 0 ; 
}

