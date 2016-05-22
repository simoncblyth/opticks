// op --ngunconfig

#include "NGunConfig.hpp"

#include "NPY.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    NGunConfig* gc = new NGunConfig ; 
    gc->parse();

    // hmm GDML needs to write things into geocache too..
    NPY<float>* transforms = NPY<float>::load("/tmp/gdml.npy");

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
