#include <iostream>
#include <iomanip>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SProc.hh"

#include "NPY.hpp"


unsigned mock_num_photons( unsigned gs )
{
    unsigned np = 0 ; 
    switch( gs % 10 )
    {   
       case 0: np = 3 ; break ;
       case 1: np = 2 ; break ;
       case 2: np = 1 ; break ;
       case 3: np = 2 ; break ;
       case 4: np = 3 ; break ;
       case 5: np = 1 ; break ;
       case 6: np = 5 ; break ;
       case 7: np = 4 ; break ;
       case 8: np = 8 ; break ;
       case 9: np = 6 ; break ;
   }   
   return np ;   
}


struct test_dynamic_random
{
    NPY<float>* photons ;

    test_dynamic_random()
        :
        photons(NPY<float>::make(0,4,4))
    {
    }

    void run()
    {  
        unsigned photon_offset = 0 ;  
        for(unsigned gs=0 ; gs < 10 ; gs++)
        {
            unsigned num_photons = mock_num_photons(gs); 
            unsigned ni = photons->expand(num_photons);
            LOG(info) << " expand " << num_photons << " ni " << ni ;  

            // increase array size to accomodate the photons of this genstep
            // expanding will often reallocate and thus invalidate any pointers

            for( unsigned p=0 ; p < num_photons ; p++)
            {
                unsigned target_record_id = photon_offset + p ;  
                writePhoton_(target_record_id); 
            }  
            photon_offset += num_photons ; 
        }

    }

    void writePhoton_(unsigned target_record_id)
    {
        LOG(info) << " target_record_id " << target_record_id ; 
        float f = float(target_record_id); 

        glm::vec4 post(f, f, f, f); 
        glm::vec4 dirw(f, f, f, f); 
        glm::vec4 polw(f, f, f, f); 

        photons->setQuad(target_record_id, 0, 0, post.x , post.y , post.z , post.w  );
        photons->setQuad(target_record_id, 1, 0, dirw.x , dirw.y , dirw.z , dirw.w  );
        photons->setQuad(target_record_id, 2, 0, polw.x , polw.y , polw.z , polw.w  );

        photons->setUInt(target_record_id, 3, 0, 0, target_record_id );
        photons->setUInt(target_record_id, 3, 0, 1, 0u );
        photons->setUInt(target_record_id, 3, 0, 2, 0u );
        photons->setUInt(target_record_id, 3, 0, 3, 0u );
    }

};


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_dynamic_random tdr ; 
    tdr.run(); 

    tdr.photons->dump(); 

    return 0 ; 
}

// om-;TEST=NPY8DynamicRandomTest om-t

