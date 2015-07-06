#include "PhotonsNPY.hpp"
#include "uif.h"
#include "NPY.hpp"
#include "RecordsNPY.hpp"

#include <map>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <glm/glm.hpp>
#include "limits.h"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"


#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



void PhotonsNPY::setRecs(RecordsNPY* recs)
{
    m_recs = recs ; 
    m_maxrec = recs->getMaxRec();
}


void PhotonsNPY::dump(unsigned int photon_id)
{
    dumpPhotonRecord(photon_id);
}


void PhotonsNPY::dumpPhotons(const char* msg, unsigned int ndump)
{
    if(!m_photons) return ;
    printf("%s\n", msg);

    unsigned int ni = m_photons->m_len0 ;
    unsigned int nj = m_photons->m_len1 ;
    unsigned int nk = m_photons->m_len2 ;
    assert( nj == 4 && nk == 4 );

    for(unsigned int i=0 ; i<ni ; i++ )
    {
        bool out = i < ndump || i > ni-ndump ; 
        if(out) dumpPhotonRecord(i);
    }
}


void PhotonsNPY::dumpPhotonRecord(unsigned int photon_id, const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int r=0 ; r<m_maxrec ; r++)
    {
        unsigned int record_id = photon_id*m_maxrec + r ;
        m_recs->dumpRecord(record_id);
    }  
    dumpPhoton(photon_id);
    printf("\n");
}


void PhotonsNPY::dumpPhoton(unsigned int i, const char* msg)
{
    unsigned int history = m_photons->getUInt(i, 3, 3);
    std::string phistory = m_types->getHistoryString( history );

    glm::vec4 post = m_photons->getQuad(i,0);
    glm::vec4 dirw = m_photons->getQuad(i,1);
    glm::vec4 polw = m_photons->getQuad(i,2);

    std::string seqmat = m_recs->getSequenceString(i, Types::MATERIAL) ;
    std::string seqhis = m_recs->getSequenceString(i, Types::HISTORY) ;

    std::string dseqmat = m_types->decodeSequenceString(seqmat, Types::MATERIAL);
    std::string dseqhis = m_types->decodeSequenceString(seqhis, Types::HISTORY);


    printf("%s %8u %s %s %25s %25s %s \n", 
                msg,
                i, 
                gpresent(post,2,11).c_str(),
                gpresent(polw,2,7).c_str(),
                seqmat.c_str(),
                seqhis.c_str(),
                phistory.c_str());

    printf("%s\n", dseqmat.c_str());
    printf("%s\n", dseqhis.c_str());
}


void PhotonsNPY::debugdump(const char* msg)
{
    if(!m_photons) return ;
    printf("%s\n", msg);

    unsigned int ni = m_photons->m_len0 ;
    unsigned int nj = m_photons->m_len1 ;
    unsigned int nk = m_photons->m_len2 ;

    assert( nj == 4 && nk == 4 );

    std::vector<float>& data = m_photons->m_data ; 

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
           assert(index == m_photons->getValueIndex(i,j,k));

           if(out)
           {
               uif.f = data[index] ;
               if(      j == 3 && k == 0 ) printf(" %15d ",   uif.i );
               else if( j == 3 && k == 3 ) printf(" %15d ",   uif.u );
               else                        printf(" %15.3f ", uif.f );
           }
           assert(index == check);
           check += 1 ; 
       }
       if(out)
       {
           if( j == 0 ) printf(" position/time ");
           if( j == 1 ) printf(" direction/wavelength ");
           if( j == 2 ) printf(" polarization/weight ");
           if( j == 3 ) printf(" boundary/cos_theta/distance_to_boundary/flags ");

           printf("\n");
       }
    }
    }
}




//glm::ivec4 PhotonsNPY::getFlags()
//{
//    return m_types->getFlags();
//}




