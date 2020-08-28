#include "SStr.hh"
#include "SPPM.hh"
#include "OPTICKS_LOG.hh"


void test_MakeWriteRead()
{
    const char* path = "/tmp/SPPMTest.ppm" ;

    //const char* config = "checkerboard" ; 
    //const char* config = "horizontal_gradient" ; 
    //const char* config = "vertical_gradient" ; 
    const char* config = "vertical_gradient,add_border,add_midline,add_quadline" ; 

    const int width = 1024 ; 
    const int height = 512 ; 
    const int ncomp = 3 ;    
    const bool yflip = true ; 
    const int size = height*width*ncomp ; 

    LOG(info) 
         << " path " << path 
         << " width " << width
         << " height " << height
         << " size " << size
         << " yflip " << yflip
         << " config " << config
         ;    

    unsigned char* imgdata = SPPM::MakeTestImage(width, height, ncomp, yflip, config);

    SPPM::write(path, imgdata, width, height, ncomp, yflip );

    SPPM::dumpHeader(path); 

    std::vector<unsigned char> img ; 
    unsigned width2(0); 
    unsigned height2(0); 

    int rc = SPPM::read( path, img, width2, height2, ncomp, yflip ); 

    assert( rc == 0 ); 
    assert( width2 == width ); 
    assert( height2 == height ); 


    unsigned char* imgdata2 = img.data(); 
    unsigned mismatch = SPPM::ImageCompare( height, width, ncomp, imgdata, imgdata2 ); 
    LOG(info) << " mismatch " << mismatch ; 

    assert( mismatch == 0 ); 
}




int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv);

    test_MakeWriteRead();      

    return 0 ; 
}
