#pragma once

#include "assert.h"


union uif_t {
    unsigned int u;
    int i ;
    float f;
} ;



class NPY ;

// resist temptation to use inheritance here, 
// it causes much grief for little benefit 
// instead using "friend class" status to 
// give G4StepNPY access to innards of NPY
//
 
class G4StepNPY {
   public:  
       G4StepNPY(NPY* npy);

   public:  
       void dump(const char* msg);
 
   private:
        NPY* m_npy ; 
 
};


G4StepNPY::G4StepNPY(NPY* npy) :
     m_npy(npy)
{
}

//
// hmm CerenkovStep and ScintillationStep have same shapes but different meanings see
//     /usr/local/env/chroma_env/src/chroma/chroma/cuda/cerenkov.h
//     /usr/local/env/chroma_env/src/chroma/chroma/cuda/scintillation.h
//
//  but whats needed for visualization should be in the same locations ?
//

void G4StepNPY::dump(const char* msg)
{
    if(!m_npy) return ;

    printf("%s\n", msg);

    unsigned int ni = m_npy->m_len0 ;
    unsigned int nj = m_npy->m_len1 ;
    unsigned int nk = m_npy->m_len2 ;
    std::vector<float>& data = m_npy->m_data ; 

    printf(" ni %u nj %u nk %u nj*nk %u \n", ni, nj, nk, nj*nk ); 

    uif_t uif ; 

    unsigned int check = 0 ;
    for(unsigned int i=0 ; i<ni ; i++ ){
    for(unsigned int j=0 ; j<nj ; j++ )
    {
       bool out = i == 0 || i == ni-1 ; 
       if(out) printf(" (%5u,%5u) ", i,j );
       for(unsigned int k=0 ; k<nk ; k++ )
       {
           unsigned int index = i*nj*nk + j*nk + k ;
           if(out)
           {
               uif.f = data[index] ;
               if( j == 0 || (j == 3 && k == 0)) printf(" %15d ",   uif.i );
               else         printf(" %15.3f ", uif.f );
           }
           assert(index == check);
           check += 1 ; 
       }
       if(out)
       {
           if( j == 0 ) printf(" sid/parentId/materialIndex/numPhotons ");
           if( j == 1 ) printf(" position/time ");
           if( j == 2 ) printf(" deltaPosition/stepLength ");
           if( j == 3 ) printf(" code ");

           printf("\n");
       }
    }
    }
}


