#include <sstream>

#include "NPY.hpp"
#include "NNode.hpp"
#include "NCSG.hpp"

#include "OpticksFlags.hh"
#include "OpticksEvent.hh"
#include "OpticksEventAna.hh"
#include "PLOG.hh"

OpticksEventAna::OpticksEventAna( OpticksEvent* evt, NCSG* csg )
    :
    m_evt(evt),
    m_pho(evt->getPhotonData()),
    m_seq(evt->getSequenceData()),
    m_pho_num(m_pho->getShape(0)),
    m_seq_num(m_seq->getShape(0)),
    m_csg(csg),
    m_root(csg->getRoot()),
    m_sdf(m_root->sdf()),
    m_epsilon(0.1f)
{
    init();
}

void OpticksEventAna::init()
{
    assert(m_pho_num == m_seq_num);
    countExcursions();
}


void OpticksEventAna::dump(const char* msg)
{
    LOG(info) << msg << " " << desc() ; 

    m_pho->dump();
    m_seq->dump();

    dumpExcursions();

}

std::string OpticksEventAna::desc()
{
    std::stringstream ss ; 
    ss  << "OpticksEventAna"
        << " pho " << m_pho->getShapeString()
        << " seq " << m_seq->getShapeString()
        ;
    return ss.str();
}



void OpticksEventAna::countExcursions()
{
    unsigned long long seqhis_ ; 

    for(unsigned i=0 ; i < m_pho_num ; i++)
    {
        glm::vec4 post = m_pho->getQuad(i,0,0);
        glm::vec4 pos(post);
        pos.w = 1.0f ; 

        //glm::vec4 lpos = igtr * pos ; 
        glm::vec4 lpos = pos ; 
        float sd = m_sdf(lpos.x, lpos.y, lpos.z);
        float asd = std::abs(sd) ;

        seqhis_ = m_seq->getValue(i,0,0);
        //seqmat_ = seq->getValue(i,0,1);
        
        m_tot[seqhis_]++;
        if(asd > m_epsilon) m_exc[seqhis_]++ ;  
   }
}

void OpticksEventAna::dumpExcursions()
{
    LOG(info) << "OpticksEventAna::dumpExcursions"
              << " seqhis ending AB or truncated seqhis : exc expected "
              ; 

    for(MQC::const_iterator it=m_tot.begin() ; it != m_tot.end() ; it++)
    {
        unsigned long long _seqhis = it->first ; 
        unsigned tot_ = it->second ; 
        unsigned exc_ = m_exc[_seqhis];
        float frac = exc_/tot_ ; 

        std::cout 
             << " seqhis " << std::setw(16) << std::hex << _seqhis << std::dec
             << " " << std::setw(64) << OpticksFlags::FlagSequence( _seqhis, true )
             << " tot " << std::setw(6) << tot_ 
             << " exc " << std::setw(6) << exc_
             << " exc/tot " << std::setw(6) << frac
             << std::endl ;  
    }
}







