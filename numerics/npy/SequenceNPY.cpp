#include "SequenceNPY.hpp"
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


void SequenceNPY::setRecs(RecordsNPY* recs)
{
    m_recs = recs ; 
    m_maxrec = recs->getMaxRec();
}

void SequenceNPY::dumpUniqueHistories()
{
    // find counts of all histories 
    typedef std::map<unsigned int,unsigned int>  MUU ; 
    MUU uu = m_photons->count_unique_u(3,3) ; 
    dumpMaskCounts("SequenceNPY::dumpUniqueHistories : ", Types::HISTORY, uu, 1);
}

/*
   This is slow... better to do it GPU side with thrust ...

   * could collect per photon flag sequences into a big integer
     optix types  

     * 128 bit ints are not yet supported by thrust/CUDA 
     * CUDA long long is 64 bit (8 bytes)
     * squeezing to 4 bits per entry (1-15 with 0 for overflow "other")
       even a 64 bit could usefully fit 16 steps

     * the big int would play role of the sequence string in the below

     * see thrustexamples-




*/



void SequenceNPY::indexSequences()
{
    LOG(info)<<"SequenceNPY::indexSequences START ... this takes a while " ; 

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

         // map counting difference history/material sequences : noddy approach to sparse histogramming 
         suh[seqhis] += 1; 
         sum[seqmat] += 1 ; 

         // collect vectors of photon_id for each distinct sequence
         svh[seqhis].push_back(photon_id); 
         svm[seqmat].push_back(photon_id); 

         //
         // no need to collect lists of photon_id in thrust approach 
         // as the sequence bigint is created in place within a per-photon history_buffer 
         //
         // but will need to pass over all those doing a lookup from the sorted 
         // frequency histogram to the "bin" index
         // 
         // actually maybe not...  
         //
         //   * frequency histogram is a small structure with 2/3 equal length lists
         //     (could be fixed length, with an "Other")
         //
         //     * keys   : sequence big ints 
         //     * counts : how many occurences of that sequence
         //     * values : indices in asc/desc count order
         //
         // can do this lookup as needed by furnishing the 
         // frequency histogram as uniform buffer or texture to the shader ?
         // ... hmm not so easy as its a sparse histogram, more of a map  
         //
         // maybe have to resort to plain CUDA (or OptiX launch without rtTrace) 
         // to do the lookup ? 
         //
    }
    assert( mismatch.size() == 0);

    printf("SequenceNPY::indexSequences photons %u mismatch %lu \n", ni, mismatch.size());
    dumpMaskCounts("SequenceNPY::indexSequences histories", Types::HISTORY, uuh, 1 );
    dumpMaskCounts("SequenceNPY::indexSequences materials", Types::MATERIAL, uum, 1000 );
    dumpSequenceCounts("SequenceNPY::indexSequences seqhis", Types::HISTORY, suh , svh, 1000);
    dumpSequenceCounts("SequenceNPY::indexSequences seqmat", Types::MATERIAL, sum , svm, 1000);


    Index* idxh = makeSequenceCountsIndex( Types::HISTORYSEQ,  suh , svh, 1000 );
    idxh->dump("SequenceNPY::indexSequences");
    fillSequenceIndex( e_seqhis , idxh, svh );

    Index* idxm = makeSequenceCountsIndex( Types::MATERIALSEQ,  sum , svm, 1000 );
    idxm->dump("SequenceNPY::indexSequences");
    fillSequenceIndex( e_seqmat, idxm, svm );

    m_seqhis = idxh ; 
    m_seqmat = idxm ; 

    LOG(info)<<"SequenceNPY::indexSequences DONE " ; 
}



NPY<unsigned char>* SequenceNPY::getSeqIdx()
{
    if(!m_seqidx)
    { 
        unsigned int nr = m_recs->getRecords()->getShape(0) ;
        m_seqidx = NPY<unsigned char>::make_vec4(nr,1,0) ;
    }
    return m_seqidx ; 
}


void SequenceNPY::fillSequenceIndex(
       unsigned int k,
       Index* idx, 
       std::map<std::string, std::vector<unsigned int> >&  sv 
)
{
    assert( k < 4 );

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
            LOG(warning) << "SequenceNPY::fillSequenceIndex TOO MANY SEQUENCES : TRUNCATING " ; 
            break ; 
        }

        for(VU::iterator it=pids.begin() ; it != pids.end() ; it++)
        {
            unsigned int photon_id = *it ;  
            // duplicates seq indices for all records of the photon
            for(unsigned int r=0 ; r < m_maxrec ; r++)
            {
                unsigned int record_id = photon_id*m_maxrec + r  ;  
                seqidx->setValue(record_id, 0, k, pseq );
                nseq++;
            }
        }
    }

    std::cout << "SequenceNPY::fillSequenceIndex " 
              << std::setw(3) << k
              << std::setw(15) << idx->getItemType()
              << " sequenced " 
              << std::setw(7) << nseq 
              << std::endl ; 

}



bool SequenceNPY::second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b)
{
    return a.second > b.second ;
}

void SequenceNPY::dumpMaskCounts(const char* msg, Types::Item_t etype, 
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


bool SequenceNPY::su_second_value_order(const std::pair<std::string,unsigned int>&a, const std::pair<std::string,unsigned int>&b)
{
    return a.second > b.second ;
}


Index* SequenceNPY::makeSequenceCountsIndex(
       Types::Item_t etype, 
       std::map<std::string, unsigned int>& su,
       std::map<std::string, std::vector<unsigned int> >&  sv,
       unsigned int cutoff
       )
{
    const char* itemname = m_types->getItemName(etype);
    Index* idx = new Index(itemname);

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
              << "SequenceNPY::makeSequenceCountsIndex" 
              << " total " << total 
              << " cutoff " << cutoff 
              << " itemname " << itemname
              << std::endl ; 


    for(unsigned int i=0 ; i < idx->getNumItems() ; i++)
    {
         std::string label = idx->getNameLocal(i+1) ;
         std::string dlabel = m_recs->decodeSequenceString(label, etype);

         std::cout << std::setw(3) << i + 1 
                   << std::setw(35) << label
                   << " : "  
                   << dlabel
                   << std::endl ; 
    }  


    return idx ; 
}



void SequenceNPY::dumpSequenceCounts(const char* msg, Types::Item_t etype, 
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




