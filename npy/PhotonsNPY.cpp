/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <climits>


#include "NGLM.hpp"
#include "uif.h"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"

#include "NPY.hpp"
#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"

#include "Types.hpp"
#include "Typ.hpp"

#include "PLOG.hh"


PhotonsNPY::PhotonsNPY(NPY<float>* photons) 
       :  
       m_photons(photons),
       m_flat(false),
       m_recs(NULL),
       m_types(NULL),
       m_typ(NULL),
       m_maxrec(0)
{
}

void PhotonsNPY::setTypes(Types* types)
{  
    m_types = types ; 
}
void PhotonsNPY::setTyp(Typ* typ)
{  
    m_typ = typ ; 
}

NPY<float>* PhotonsNPY::getPhotons()
{
    return m_photons ; 
}
RecordsNPY* PhotonsNPY::getRecs()
{
    return m_recs ; 
}
Types* PhotonsNPY::getTypes()
{
    return m_types ; 
}



void PhotonsNPY::setRecs(RecordsNPY* recs)
{
    m_recs = recs ; 
    m_maxrec = recs->getMaxRec();
    m_flat = recs->isFlat();
}

void PhotonsNPY::dump(unsigned int photon_id, const char* msg)
{
    dumpPhotonRecord(photon_id, msg);
}

void PhotonsNPY::dumpPhotons(const char* msg, unsigned int ndump)
{
    if(!m_photons) return ;
    printf("%s\n", msg);

    unsigned int ni = m_photons->m_ni ;
    unsigned int nj = m_photons->m_nj ;
    unsigned int nk = m_photons->m_nk ;
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
         unsigned int i = m_flat ? record_id : photon_id ;
         unsigned int j = m_flat ? 0         : r ;
         m_recs->dumpRecord(i, j, "PhotonsNPY::dumpPhotonRecord (i,j)");
    }


    dumpPhoton(photon_id);
    printf("\n");

    glm::vec4 ce = m_recs->getCenterExtent(photon_id);
    print(ce, "ce" );
    glm::vec4 ldd = m_recs->getLengthDistanceDuration(photon_id);
    print(ldd, "ldd" );
}


NPY<float>* PhotonsNPY::make_pathinfo()
{
    unsigned int num_photons = m_photons->m_ni ;
    NPY<float>* pathinfo = NPY<float>::make(num_photons,6,4) ;
    pathinfo->zero();
    for(unsigned int i=0 ; i < num_photons ; i++)
    {
        unsigned int photon_id = i ;
        glm::vec4 ce = m_recs->getCenterExtent(photon_id);
        //print(ce, "ce" );
        glm::vec4 ldd = m_recs->getLengthDistanceDuration(photon_id);
        //print(ldd, "ldd" );

        glm::vec4 post = m_photons->getQuad_(i,0);
        glm::vec4 dirw = m_photons->getQuad_(i,1);
        glm::vec4 polw = m_photons->getQuad_(i,2);
        glm::vec4 flag = m_photons->getQuad_(i,3);  // int as floats ?

        pathinfo->setQuad( post, i, 0);
        pathinfo->setQuad( dirw, i, 1);
        pathinfo->setQuad( polw, i, 2);
        pathinfo->setQuad( flag, i, 3);

        pathinfo->setQuad( ce,   i, 4 );
        pathinfo->setQuad( ldd,  i, 5 );
    }  
    return pathinfo ; 
}


void PhotonsNPY::dumpPhoton(unsigned int i, const char* msg)
{
    unsigned int history = m_photons->getUInt(i, 3, 3);
    std::string phistory = m_types->getHistoryString( history );

    glm::vec4 post = m_photons->getQuad_(i,0);
    //glm::vec4 dirw = m_photons->getQuad_(i,1);
    glm::vec4 polw = m_photons->getQuad_(i,2);

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

    unsigned int ni = m_photons->m_ni ;
    unsigned int nj = m_photons->m_nj ;
    unsigned int nk = m_photons->m_nk ;

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






