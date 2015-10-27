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
#include "stringutil.hpp"


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


void SequenceNPY::countMaterials()
{
    typedef std::vector<unsigned int> VU ;
    typedef std::map<unsigned int, unsigned int> MUU ; 
    typedef std::pair<unsigned int, unsigned int> PUU ;

    unsigned int ni = m_photons->m_len0 ;
    VU materials ; 

    // collect m1, m2 material codes from all records of all photons
    for(unsigned int id=0 ; id < ni ; id++) m_recs->appendMaterials(materials, id);

    // count occurence of each material code
    MUU mmat ; 
    for(VU::iterator it=materials.begin() ; it != materials.end() ; it++) mmat[*it] += 1 ;  

     // arrange into pairs for sorting 
    std::vector<PUU> matocc ; 
    for(MUU::iterator it=mmat.begin() ; it != mmat.end() ; it++) matocc.push_back(*it);
    std::sort(matocc.begin(), matocc.end(), second_value_order );

    // check that set sees the same count of uniques
    std::set<unsigned int> smat(materials.begin(), materials.end()) ; 
    assert(smat.size() == mmat.size() );

    LOG(info) << "SequenceNPY::countMaterials " 
              << " m1/m2 codes in all records " << materials.size() 
              << " unique material codes " << smat.size() ; 


    unsigned int idx(0);
    for(std::vector<PUU>::iterator it=matocc.begin() ; it != matocc.end() ; it++)
    {
        unsigned int mat = it->first ; 
        unsigned int occ = it->second ; 
        assert(mmat[mat] == occ);
        std::string matn = m_types->getMaterialString(1 << (mat - 1));
        std::cout 
                  << std::setw(5)  << idx 
                  << std::setw(5)  << mat 
                  << std::setw(10) << occ 
                  << std::setw(20) << matn << "."
                  << std::endl ; 

        idx++ ; 
    }
}


void SequenceNPY::indexSequences(unsigned int maxidx)
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
    }
    assert( mismatch.size() == 0);
    LOG(info) << "SequenceNPY::indexSequences photon loop DONE photons " <<  ni ;


    // order by counts of that sequence

    m_history_counts.addMap(suh) ;
    m_history_counts.sort(false) ;
    m_history_counts.dump("m_history_counts", maxidx) ;

    m_seqhis = makeSequenceCountsIndex( Types::HISTORYSEQ,  m_history_counts.counts(), maxidx );
    m_seqhis_npy = makeSequenceCountsArray(Types::HISTORYSEQ,  m_history_counts.counts()  );


    m_material_counts.addMap(sum) ;
    m_material_counts.sort(false) ;
    m_material_counts.dump("m_material_counts", maxidx) ;

    m_seqmat = makeSequenceCountsIndex( Types::MATERIALSEQ,  m_material_counts.counts(), maxidx );
    m_seqmat->dump("SequenceNPY::indexSequences (seqmat)");


    assert(m_seqidx);
    fillSequenceIndex( e_seqhis, m_seqhis, svh );
    fillSequenceIndex( e_seqmat, m_seqmat, svm );


    LOG(info)<<"SequenceNPY::indexSequences DONE " ; 
}



void SequenceNPY::fillSequenceIndex(
       unsigned int k,
       Index* idx, 
       std::map<std::string, std::vector<unsigned int> >&  sv 
)
{
    // uses the vectors of photon_id to fill in the sequence 
    // index repeating the index for maxrec items 

    assert( k < 4 );

    NPY<unsigned char>* seqidx = getSeqIdx(); 
    assert(seqidx && "must setSeqIdx before can populate the index");

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



NPY<unsigned long long>* SequenceNPY::makeSequenceCountsArray( 
       Types::Item_t etype, 
       std::vector< std::pair<std::string, unsigned int> >& vp
     ) 
{
    typedef std::pair<std::string, unsigned int> PSU ;
    unsigned int ni = vp.size() ;
    unsigned int nj = 1 ; 
    unsigned int nk = 2 ; 

    std::vector<unsigned long long> values ; 
    for(unsigned int i=0 ; i < ni ; i++)
    {
        PSU p = vp[i];
        unsigned long long xseq = m_types->convertSequenceString( p.first, etype, false ) ;
        values.push_back(xseq);
        values.push_back(p.second);
    } 
    NPY<unsigned long long>* npy = NPY<unsigned long long>::make(ni, nj, nk);
    npy->setData( values.data() );
    return npy ;  
}




Index* SequenceNPY::makeSequenceCountsIndex(
       Types::Item_t etype, 
       std::vector< std::pair<std::string, unsigned int> >& vp,
       unsigned long maxidx, 
       bool hex
       )
{
    const char* itemname = m_types->getItemName(etype);
    std::string idxname = hex ? std::string("Hex") + itemname : itemname ;  

    LOG(info) << "SequenceNPY::makeSequenceCountsIndex " 
              << " itemname " << itemname 
              << " idxname " << idxname 
              << " maxidx " << maxidx 
              ;
    Index* idx = new Index(idxname.c_str());


    // populate idx with the sequences having greater than cutoff ocurrences
    unsigned int total(0);
    typedef std::pair<std::string, unsigned int> PSU ;

    // index truncation is a feature, not a limitation
    for(unsigned int i=0 ; i < std::min(maxidx, vp.size()) ; i++)
    {
        PSU p = vp[i];
        total += p.second ;  

        std::string xkey = p.first ; 
        if(hex)
        {
            // use sequence hex string as key to enable comparison with ThrustHistogram saves
            unsigned long long xseq = m_types->convertSequenceString( p.first, etype, false ) ;
            xkey = as_hex(xseq);
        }

        idx->add( xkey.c_str(), i, false ); // dont sort names while adding
    }

    idx->sortNames();


    for(unsigned int i=0 ; i < idx->getNumItems() ; i++)
    {
         std::string label = idx->getNameLocal(i+1) ;
         std::string dlabel = m_types->decodeSequenceString(label, etype, hex);
         unsigned long long xseq = m_types->convertSequenceString(label, etype, hex);

         std::cout << std::setw(3) << std::dec << i + 1 
                   << std::setw(18) << std::hex << xseq 
                   << std::setw(35) << label
                   << " : "  
                   << dlabel
                   << std::endl ; 
    }  

    std::cout 
              << "SequenceNPY::makeSequenceCountsIndex  DONE " 
              << " total " << total 
              << " maxidx " << maxidx 
              << " itemname " << itemname
              << std::endl ; 


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
               << m_types->decodeSequenceString(p.first, etype) 
               << std::endl ; 
    }

    std::cout 
              << " total " << total 
              << " cutoff " << cutoff 
              << std::endl ; 

}


