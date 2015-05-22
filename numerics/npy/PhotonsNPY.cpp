#include "PhotonsNPY.hpp"
#include "uif.h"
#include "NPY.hpp"
#include <set>
#include <map>

void PhotonsNPY::classify()
{
    if(!m_npy) return ;
    std::set<int> uniq = m_npy->uniquei(3,0) ;
    for(std::set<int>::iterator it=uniq.begin() ; it != uniq.end() ; it++)
    {
        int val = *it ; 
        printf("%d \n", val );
    }  

    std::map<int,int> uniqn = m_npy->count_uniquei(3,0) ;
    for(std::map<int,int>::iterator it=uniqn.begin() ; it != uniqn.end() ; it++)
    {
        printf("%d : %d \n", it->first, it->second );
    }


}


void PhotonsNPY::dump(const char* msg)
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
       if(out) printf(" (%7u,%1u) ", i,j );
       for(unsigned int k=0 ; k<nk ; k++ )
       {
           unsigned int index = i*nj*nk + j*nk + k ;
           if(out)
           {
               uif.f = data[index] ;
               if( j == 3 && k == 0 ) printf(" %15d ",   uif.i );
               else                   printf(" %15.3f ", uif.f );
           }
           assert(index == check);
           check += 1 ; 
       }
       if(out)
       {
           if( j == 0 ) printf(" position/time ");
           if( j == 1 ) printf(" direction/wavelength ");
           if( j == 2 ) printf(" polarization/weight ");
           if( j == 3 ) printf(" boundary/cos_theta/distance_to_boundary/- ");

           printf("\n");
       }
    }
    }
}


