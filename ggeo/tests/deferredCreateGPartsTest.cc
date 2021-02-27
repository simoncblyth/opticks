#include "OPTICKS_LOG.hh"
#include "GLMFormat.hpp"
#include "NPY.hpp"
#include "nmat4triple.hpp"
#include "SPack.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    const char* path = "$TMP/GGeo__deferredCreateGParts/mm0/mismatch_placements.npy" ; 
    NPY<float>* a = NPY<float>::load(path);  
    if(!a)
    {
        LOG(fatal) << "failed to load " << path ; 
        return 0 ; 
    }
    LOG(info) << a->getShapeString() ; 

    float* values = a->getValues() ;
    for(unsigned i=0 ; i < a->getNumItems() ; i++)
    {
        glm::mat4 placement = glm::make_mat4( values + i*16 ) ;

        unsigned ptIdx = SPack::float_as_unsigned(placement[0][3]); 
        unsigned lvIdx = SPack::float_as_unsigned(placement[1][3]); 
        unsigned ndIdx = SPack::float_as_unsigned(placement[2][3]); 
        unsigned num_mismatch = SPack::float_as_unsigned(placement[3][3]); 
        std::cout 
            << " ptIdx " << std::setw(4) << ptIdx 
            << " lvIdx " << std::setw(5) << lvIdx 
            << " ndIdx " << std::setw(8) << ndIdx 
            << " num_mismatch " << std::setw(7) << num_mismatch
            << std::endl  
            ;

        placement[0][3] = 0.f ;   // scrub identity info 
        placement[1][3] = 0.f ; 
        placement[2][3] = 0.f ; 
        placement[3][3] = 1.f ; 
        std::cout << gpresent__("tr", placement ) << std::endl ;          

        // nmat4triple::make_transformed
        nmat4triple perturb(placement);
        if(perturb.match == false)
        {   
            LOG(error) << "perturb.match false : precision issue in inverse ? " ; 
        }   



        
    }
    return 0 ; 
}

// TEST=deferredCreateGPartsTest ; om-; om-t 

