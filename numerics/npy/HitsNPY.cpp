

#include "NSensor.hpp"
#include "NSensorList.hpp"
#include "NPY.hpp"

#include "HitsNPY.hpp"

#include "PLOG.hh"


HitsNPY::HitsNPY(NPY<float>* photons, NSensorList* sensorlist) 
       :  
       m_photons(photons),
       m_sensorlist(sensorlist)
{
}


void HitsNPY::debugdump(const char* msg)
{
    LOG(info) << msg ; 

    unsigned int ni = m_photons->m_ni ;
    unsigned int nj = m_photons->m_nj ;
    unsigned int nk = m_photons->m_nk ;
    assert( nj == 4 && nk == 4 && ni > 0 );

    enum { X, Y, Z, W } ;

    typedef std::map<unsigned int,unsigned int> UU ; 

    UU cuu = m_photons->count_unique_u(3,Y) ;

    for(UU::const_iterator it=cuu.begin() ; it != cuu.end() ; it++)
    {
        unsigned int sensorIndex = it->first ; 

        NSensor* sensor = sensorIndex > 0 ? m_sensorlist->getSensor(sensorIndex-1) : NULL ;

        printf(" %4u : %4u : %s  \n", it->first, it->second, sensor ? sensor->description().c_str() : "-"  );

    }

}

