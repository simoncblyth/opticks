#include <iostream>
#include <iomanip>

#include "NGLM.hpp"
#include "NPY.hpp"
#include "SeqNPY.hpp"

#include "PLOG.hh"

const unsigned SeqNPY::N = 16 ; 

SeqNPY::SeqNPY(NPY<unsigned long long>* sequence) 
    :
    m_sequence(sequence),
    m_counts(new int[N])
{
    init();
}

void SeqNPY::init()
{
    for(unsigned i=0 ; i < N ; i++) m_counts[i] = 0 ; 
    countPhotons();
}


// hmm : would be better to use the GPU derived
//       indices rather than gpong back to the raw sequence
//       BUT: this is handy as a check anyhow

void SeqNPY::countPhotons()
{
    unsigned int ni = m_sequence->m_ni ;
    unsigned int nj = m_sequence->m_nj ;
    unsigned int nk = m_sequence->m_nk ;
    assert(nj == 1 && nk == 2);

    unsigned long long* values = m_sequence->getValues();
    if(!values) 
    {
        LOG(warning) << "SeqNPY::countPhotons requires the sequence values to be copied back to host" ; 
        return ; 
    }

    for(unsigned i=0 ; i<ni ; i++ )
    {
         unsigned long long val = values[i*2+0] ; // 0:seqhis, 1:seqmat
         //if(i < 100) 
         //    std::cout << std::setw(16) << std::hex << val << std::dec << std::endl ; 

         int first = int(val & 0xF) ; 
         m_counts[first] += 1 ;   
    }
}

void SeqNPY::dump(const char* msg)
{
    assert(m_sequence);
    unsigned int ni = m_sequence->m_ni ;
    unsigned int nj = m_sequence->m_nj ;
    unsigned int nk = m_sequence->m_nk ;

    LOG(info) << msg
              << " shape " << m_sequence->getShapeString()
              << " ni " << ni 
              << " nj " << nj
              << " nk " << nk
              ;

    for(unsigned i=0 ; i < N ; i++)
        std::cout 
              << std::setw(10) << i 
              << std::setw(10) << std::hex << i <<  std::dec
              << std::setw(10) << getCount(i) 
              << std::endl ; 
}
 
int SeqNPY::getCount(unsigned code)
{
    return code < N ? m_counts[code] : -1 ;  
}

std::vector<int> SeqNPY::getCounts()
{
    return std::vector<int>(m_counts, m_counts + N );
}


