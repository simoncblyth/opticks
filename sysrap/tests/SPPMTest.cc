#include "SPPM.hh"
#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv);
    LOG(info) ;  

    const char* path = "/tmp/SPPMTest.ppm" ;

    int height = 128 ; 
    int width = 256 ; 
    bool yflip = true ; 
    int ncomp = 3 ;    

    int size = width*height*ncomp ; 
    unsigned char* imgdata = new unsigned char[size] ;  

    for(int i=0 ; i < height ; i++){
    for(int j=0 ; j < width  ; j++){

        unsigned idx = i*width + j ;
        imgdata[idx*ncomp+0] = i ; 
        imgdata[idx*ncomp+1] = j ; 
        imgdata[idx*ncomp+2] = 0xff ; 
    }
    }

    LOG(info) 
         << " path " << path 
         << " width " << width
         << " height " << height
         << " yflip " << yflip
         ;    

    SPPM::write(path, imgdata, width, height, ncomp, yflip );

    SPPM::dumpHeader(path); 


    std::vector<unsigned char> img ; 
    unsigned width2(0); 
    unsigned height2(0); 

    
    const unsigned ncomp2 = 3 ; 
    const bool yflip2 = true ; 

    int rc = SPPM::read( path, img, width2, height2, ncomp2, yflip2 ); 

    unsigned size2 = width2*height2*ncomp2 ; 

    assert( rc == 0 ); 
    assert( width2 == width ); 
    assert( height2 == height ); 
    assert( ncomp2 == ncomp ); 
    assert( size2 == size ); 

    unsigned char* imgdata2 = img.data(); 

    unsigned count(0); 

    for(int h=0 ; h < height ; h++){
    for(int w=0 ; w < width  ; w++){

        unsigned idx = h*width + w ; 
        assert( idx*ncomp == count ); 
        count += ncomp ;  

        unsigned r = idx*ncomp+0 ; 
        unsigned g = idx*ncomp+1 ; 
        unsigned b = idx*ncomp+2 ; 

        bool match = 
               imgdata[r] == imgdata2[r] && 
               imgdata[g] == imgdata2[g] && 
               imgdata[b] == imgdata2[b] ; 

        if(!match) 
            std::cout 
                << " h " << std::setw(3) << h 
                << " w " << std::setw(3) << w 
                << " idx " << idx
                << " imgdata[rgb] "
                << "(" 
                << std::setw(3) << unsigned(imgdata[r]) 
                << " "
                << std::setw(3) << unsigned(imgdata[g]) 
                << " "
                << std::setw(3) << unsigned(imgdata[b]) 
                << ")"
                << " imgdata2[rgb] "
                << "(" 
                << std::setw(3) << unsigned(imgdata2[r]) 
                << " "
                << std::setw(3) << unsigned(imgdata2[g]) 
                << " "
                << std::setw(3) << unsigned(imgdata2[b]) 
                << ")"
                << std::endl
                ;

        //assert( match );  
    } 
    }


    return 0 ; 
}
