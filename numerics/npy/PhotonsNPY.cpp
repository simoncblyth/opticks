#include "PhotonsNPY.hpp"
#include "uif.h"
#include "NPY.hpp"
#include "RecordsNPY.hpp"
#include "Index.hpp"

#include <set>
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



bool second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b)
{
    return a.second > b.second ;
}

bool su_second_value_order(const std::pair<std::string,unsigned int>&a, const std::pair<std::string,unsigned int>&b)
{
    return a.second > b.second ;
}


void PhotonsNPY::setRecs(RecordsNPY* recs)
{
    m_recs = recs ; 
    m_maxrec = recs->getMaxRec();
}



void PhotonsNPY::classify(bool sign)
{
    m_boundaries.clear();
    m_boundaries = findBoundaries(sign);
    delete m_boundaries_selection ; 
    m_boundaries_selection = m_types->initBooleanSelection(m_boundaries.size());
    //dumpBoundaries("PhotonsNPY::classify");
}
glm::ivec4 PhotonsNPY::getSelection()
{
    // ivec4 containing 1st four boundary codes provided by the selection

    int v[4] ;
    unsigned int count(0) ; 
    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
        if(m_boundaries_selection[i])
        {
            std::pair<int, std::string> p = m_boundaries[i];
            if(count < 4)
            {
                v[count] = p.first ; 
                count++ ; 
            }
            else
            {
                 break ;
            }
        }
    }  
    glm::ivec4 iv(-INT_MAX,-INT_MAX,-INT_MAX,-INT_MAX);   // zero tends to be meaningful, so bad default for "unset"
    if(count > 0) iv.x = v[0] ;
    if(count > 1) iv.y = v[1] ;
    if(count > 2) iv.z = v[2] ;
    if(count > 3) iv.w = v[3] ;
    return iv ;     
}



void PhotonsNPY::dumpBoundaries(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
         std::pair<int, std::string> p = m_boundaries[i];
         printf(" %2d : %s \n", p.first, p.second.c_str() );
    }
}


std::vector<std::pair<int, std::string> > PhotonsNPY::findBoundaries(bool sign)
{
    assert(m_photons);

    std::vector<std::pair<int, std::string> > boundaries ;  

    printf("PhotonsNPY::findBoundaries \n");


    std::map<int,int> uniqn = sign ? m_photons->count_uniquei(3,0,2,0) : m_photons->count_uniquei(3,0) ;

    // To allow sorting by count
    //      map<boundary_code, count> --> vector <pair<boundary_code,count>>

    std::vector<std::pair<int,int> > pairs ; 
    for(std::map<int,int>::iterator it=uniqn.begin() ; it != uniqn.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), second_value_order );


    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        std::pair<int,int> p = pairs[i]; 
        int code = p.first ;
        std::string name ;
        if(m_names.count(abs(code)) > 0) name = m_names[abs(code)] ; 

        char line[128] ;
        snprintf(line, 128, " %3d : %7d %s ", p.first, p.second, name.c_str() );
        boundaries.push_back( std::pair<int, std::string>( code, line ));
    }   

    return boundaries ;
}



unsigned char msb_( unsigned short x )
{
    return ( x & 0xFF00 ) >> 8 ;
}

unsigned char lsb_( unsigned short x )
{
    return ( x & 0xFF)  ;
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

    std::string dseqmat = m_recs->decodeSequenceString(seqmat, Types::MATERIAL);
    std::string dseqhis = m_recs->decodeSequenceString(seqhis, Types::HISTORY);


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


void PhotonsNPY::dump(const char* msg)
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




glm::ivec4 PhotonsNPY::getFlags()
{
    return m_types->getFlags();
}


void PhotonsNPY::examinePhotonHistories()
{
    // find counts of all histories 
    typedef std::map<unsigned int,unsigned int>  MUU ; 

    MUU uu = m_photons->count_unique_u(3,3) ; 

    dumpMaskCounts("PhotonsNPY::examinePhotonHistories : ", Types::HISTORY, uu, 1);
}

void PhotonsNPY::prepSequenceIndex()
{
    unsigned int ni = m_photons->m_len0 ;

    typedef std::vector<unsigned int> VU ;
    VU mismatch ;

    typedef std::map<unsigned int, unsigned int> MUU ;
    MUU uuh ;  
    MUU uum ;  

    typedef std::map<std::string, unsigned int>  MSU ;
    MSU sum ;  
    MSU suh ;  

    typedef std::map<std::string, std::vector<unsigned int> >  MSV ;
    MSV svh ;  
    MSV svm ;  


    for(unsigned int i=0 ; i < ni ; i++) // over all photons
    { 
         unsigned int photon_id = i ; 

         // from the (upto m_maxrec) records for each photon 
         // fabricate history and material masks : by or-ing together the bits
         unsigned int history(0) ;
         unsigned int bounce(0) ;
         unsigned int material(0) ;
         m_recs->constructFromRecord(photon_id, bounce, history, material); 

         // compare with the photon mask, formed GPU side 
         // should match perfectly so long as bounce_max < maxrec 
         // (that truncates big-bouncers in the same way on GPU/CPU)
         unsigned int phistory = m_photons->getUInt(photon_id, 3, 3);
         if(history != phistory) mismatch.push_back(photon_id);
         assert(history == phistory);

         // map counting different history/material masks 
         uuh[history] += 1 ; 
         uum[material] += 1 ; 

         // construct sequences of materials or history flags for each step of the photon
         std::string seqmat = m_recs->getSequenceString(photon_id, Types::MATERIAL);
         std::string seqhis = m_recs->getSequenceString(photon_id, Types::HISTORY);

         // map counting difference history/material sequences
         suh[seqhis] += 1; 
         sum[seqmat] += 1 ; 

         // collect vectors of photon_id for each distinct sequence
         svh[seqhis].push_back(photon_id); 
         svm[seqmat].push_back(photon_id); 
    }
    assert( mismatch.size() == 0);

    printf("PhotonsNPY::consistencyCheck photons %u mismatch %lu \n", ni, mismatch.size());
    dumpMaskCounts("PhotonsNPY::consistencyCheck histories", Types::HISTORY, uuh, 1 );
    dumpMaskCounts("PhotonsNPY::consistencyCheck materials", Types::MATERIAL, uum, 1000 );
    dumpSequenceCounts("PhotonsNPY::consistencyCheck seqhis", Types::HISTORY, suh , svh, 1000);
    dumpSequenceCounts("PhotonsNPY::consistencyCheck seqmat", Types::MATERIAL, sum , svm, 1000);


    Index* idxh = makeSequenceCountsIndex( Types::HISTORYSEQ,  suh , svh, 1000 );
    idxh->dump();
    fillSequenceIndex( e_seqhis , idxh, svh );

    Index* idxm = makeSequenceCountsIndex( Types::MATERIALSEQ,  sum , svm, 1000 );
    idxm->dump();
    fillSequenceIndex( e_seqmat, idxm, svm );

    m_seqhis = idxh ; 
    m_seqmat = idxm ; 
}


void PhotonsNPY::fillSequenceIndex(
       unsigned int k,
       Index* idx, 
       std::map<std::string, std::vector<unsigned int> >&  sv 
)
{
    assert( k < 4 );
    unsigned int ni = m_photons->m_len0 ;
    NPY<unsigned char>* seqidx = getSeqIdx(); // creates if not exists

    unsigned int nseq(0) ; 

    for(unsigned int iseq=0 ; iseq < idx->getNumItems() ; iseq++)
    {
        unsigned int pseq = iseq + 1 ; // 1-based local seq index
        std::string seq = idx->getNameLocal(pseq); 
        typedef std::vector<unsigned int> VU ; 
        VU& pids = sv[seq];

        if(pseq >= 255)
        {
            LOG(warning) << "PhotonsNPY::fillSequenceIndex TOO MANY SEQUENCES : TRUNCATING " ; 
            break ; 
        }

        for(VU::iterator it=pids.begin() ; it != pids.end() ; it++)
        {
            unsigned int photon_id = *it ;  
            seqidx->setValue(photon_id, 0, k, pseq );
            nseq++;
        }
    }

    std::cout << "PhotonsNPY::fillSequenceIndex " 
              << std::setw(3) << k
              << std::setw(15) << idx->getItemType()
              << " sequenced/total " 
              << std::setw(7) << nseq 
              << "/"
              << std::setw(7) << ni
              << std::endl ; 

}





void PhotonsNPY::dumpMaskCounts(const char* msg, Types::Item_t etype, 
        std::map<unsigned int, unsigned int>& uu, 
        unsigned int cutoff)
{
    typedef std::map<unsigned int, unsigned int> MUU ;
    typedef std::pair<unsigned int, unsigned int> PUU ;

    std::vector<PUU> pairs ; 
    for(MUU::iterator it=uu.begin() ; it != uu.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), second_value_order );

    std::cout << msg << std::endl ; 

    unsigned int total(0);

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        PUU p = pairs[i];
        total += p.second ;  

        if(p.second > cutoff) 
            std::cout 
               << std::setw(5) << i 
               << " : "
               << std::setw(10) << std::hex << p.first
               << " : " 
               << std::setw(10) << std::dec << p.second
               << " : "
               << m_types->getMaskString(p.first, etype) 
               << std::endl ; 
    }

    std::cout 
              << " total " << total 
              << " cutoff " << cutoff 
              << std::endl ; 
}



Index* PhotonsNPY::makeSequenceCountsIndex(
       Types::Item_t etype, 
       std::map<std::string, unsigned int>& su,
       std::map<std::string, std::vector<unsigned int> >&  sv,
       unsigned int cutoff
       )
{
    Index* idx = new Index(m_types->getItemName(etype));

    typedef std::map<std::string, std::vector<unsigned int> >  MSV ;
    typedef std::map<std::string, unsigned int> MSU ;
    typedef std::pair<std::string, unsigned int> PSU ;

    // order by counts of that sequence
    std::vector<PSU> pairs ; 
    for(MSU::iterator it=su.begin() ; it != su.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), su_second_value_order );


    // populate idx with the sequences having greater than cutoff ocurrences
    unsigned int total(0);
    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        PSU p = pairs[i];
        total += p.second ;  
        assert( sv[p.first].size() == p.second );
        if(p.second > cutoff)
            idx->add( p.first.c_str(), i );
    }

    std::cout 
              << "PhotonsNPY::makeSequenceCountsIndex" 
              << " total " << total 
              << " cutoff " << cutoff 
              << std::endl ; 


    for(unsigned int i=0 ; i < idx->getNumItems() ; i++)
    {
         std::cout << std::setw(3) << i + 1 
                   << std::setw(20) << idx->getNameLocal(i+1)
                   << std::endl ; 
    }  


    return idx ; 
}



void PhotonsNPY::dumpSequenceCounts(const char* msg, Types::Item_t etype, 
       std::map<std::string, unsigned int>& su,
       std::map<std::string, std::vector<unsigned int> >& sv,
       unsigned int cutoff
    )
{
    typedef std::map<std::string, unsigned int> MSU ;
    typedef std::pair<std::string, unsigned int> PSU ;

    std::vector<PSU> pairs ; 
    for(MSU::iterator it=su.begin() ; it != su.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), su_second_value_order );

    std::cout << msg << std::endl ; 

    unsigned int total(0);

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        PSU p = pairs[i];
        total += p.second ;  

        assert( sv[p.first].size() == p.second );

        if(p.second > cutoff)
            std::cout 
               << std::setw(5) << i          
               << " : "
               << std::setw(30) << p.first
               << " : " 
               << std::setw(10) << std::dec << p.second
               << std::setw(10) << std::dec << sv[p.first].size()
               << " : "
               << m_recs->decodeSequenceString(p.first, etype) 
               << std::endl ; 
    }

    std::cout 
              << " total " << total 
              << " cutoff " << cutoff 
              << std::endl ; 

}




