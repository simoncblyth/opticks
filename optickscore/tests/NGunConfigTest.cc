// op --ngunconfig

#include "Opticks.hh"
#include "NGunConfig.hpp"

#include "NPY.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "NGunConfigTest.log" );

    NGunConfig* gc = new NGunConfig ; 
    gc->parse();

    std::string cachedir = ok.getObjectPath("CGDMLDetector", 0);

    NPY<float>* transforms = NPY<float>::load(cachedir.c_str(), "gtransforms.npy");
    assert(transforms);

    unsigned int frameIndex = gc->getFrame() ;

    if(frameIndex < transforms->getShape(0))
    {
        glm::mat4 frame = transforms->getMat4( frameIndex ); 
        gc->setFrameTransform(frame) ; 
    }
    else
    {
        std::cout << "frameIndex not found " << frameIndex << std::endl ; 
    }

    gc->Summary();


    return 0 ; 
}
