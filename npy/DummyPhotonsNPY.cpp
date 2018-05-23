#include "NPY.hpp"
#include "DummyPhotonsNPY.hpp"

DummyPhotonsNPY::DummyPhotonsNPY(unsigned num_photons, unsigned hitmask)
   :
   m_data(NPY<float>::make(num_photons, 4, 4)),
   m_hitmask(hitmask)
{
    m_data->zero();   
    makeStriped();
}

void DummyPhotonsNPY::makeStriped()
{
    unsigned numHit(0);
    unsigned numPhoton = m_data->getNumItems();
    for(unsigned i=0 ; i < numPhoton ; i++)
    {   
         nvec4 q0 = make_nvec4(i,i,i,i) ;
         nvec4 q1 = make_nvec4(1000+i,1000+i,1000+i,1000+i) ;
         nvec4 q2 = make_nvec4(2000+i,2000+i,2000+i,2000+i) ;

         unsigned uhit = i % 10 == 0 ? m_hitmask  : 0  ;   // one in 10 are mock "hits"  
         if(uhit & m_hitmask ) numHit += 1 ; 

         nuvec4 u3 = make_nuvec4(3000+i,3000+i,3000+i,uhit) ;

         m_data->setQuad( q0, i, 0 );
         m_data->setQuad( q1, i, 1 );
         m_data->setQuad( q2, i, 2 );
         m_data->setQuadU( u3, i, 3 );  // flags at the end
    }   
    m_data->setNumHit(numHit);
}

NPY<float>* DummyPhotonsNPY::getNPY()
{
    return m_data ; 
}

NPY<float>* DummyPhotonsNPY::make(unsigned num_photons, unsigned hitmask)
{
    DummyPhotonsNPY* dp = new DummyPhotonsNPY(num_photons, hitmask);
    return dp->getNPY();
}

