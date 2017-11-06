#include <sstream>

#include "NPY.hpp"
#include "GLMFormat.hpp"
#include "NNode.hpp"
#include "RecordsNPY.hpp"
#include "NCSGList.hpp"
#include "NCSG.hpp"

#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"
#include "OpticksEventStat.hh"
#include "OpticksEventAna.hh"
#include "PLOG.hh"

OpticksEventAna::OpticksEventAna( Opticks* ok, OpticksEvent* evt, NCSGList* csglist )
    :
    m_ok(ok),
    m_epsilon(0.1f),
    m_dbgseqhis(m_ok->getDbgSeqhis()),
    m_dbgseqmat(m_ok->getDbgSeqmat()),

    m_evt(evt),
    m_csglist(csglist),
    m_tree_num(csglist->getNumTrees()),

    m_sdflist(new SDF[m_tree_num]), 

    m_stat(new OpticksEventStat(evt, m_tree_num)),   

    m_records(m_evt->getRecordsNPY()),   // setupRecordsNPY done by OpticksEventStat
    m_pho(evt->getPhotonData()),
    m_seq(evt->getSequenceData()),
    m_pho_num(m_pho->getShape(0)),
    m_seq_num(m_seq->getShape(0))
{
    init();
}

void OpticksEventAna::init()
{
    assert(m_pho_num == m_seq_num);

    for(unsigned i=0 ; i < m_tree_num ; i++)
    {
        NCSG* csg = m_csglist->getTree(i);
        nnode* root = csg->getRoot();
        m_sdflist[i] = root->sdf() ; 
    }
    countExcursions();
}


void OpticksEventAna::dump(const char* msg)
{
    LOG(info) << msg << " " << desc() ; 

    //m_pho->dump();
    //m_seq->dump();

    m_stat->dump("per-seqhis per-tree counts on NCSG tree surface"  );

    dumpStepByStepCSGExcursions();

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
    LOG(info) << "OpticksEventAna::countExcursions"
              << " pho_num " << m_pho_num
              << " epsilon " << m_epsilon 
              << " dbgseqhis " << std::hex << m_dbgseqhis << std::dec 
              << " dbgseqhis " << OpticksFlags::FlagSequence( m_dbgseqhis, true )
              ;
              
    // as photon positions are final ones, this
    // aint so useful for intersect checking : as the photons
    // mostly end up being absorbed on container walls 

    bool dump = false ; 
 
    for(unsigned tree = 0 ; tree < m_tree_num ; tree++ )
    {
        unsigned count = 0 ; 
        for(unsigned i=0 ; i < m_pho_num ; i++)
        { 
            //unsigned long long seqhis_ = m_seq->getValue(i,0,0);
            //unsigned long long seqmat_ = m_seq->getValue(i,0,1);

            unsigned long long seqhis_ = m_evt->getSeqHis(i);


            glm::vec4 post = m_pho->getQuad(i,0,0);
            glm::vec4 pos(post);
            pos.w = 1.0f ; 

            //glm::vec4 lpos = igtr * pos ; 
            glm::vec4 lpos = pos ; 

            float sd = m_sdflist[tree](lpos.x, lpos.y, lpos.z);
            float asd = std::abs(sd) ;
            bool surf = asd < m_epsilon ;


            if(seqhis_ == m_dbgseqhis )
            {

                if( count < 100 && dump )
                {
                    std::cout 
                              << " sd " << std::setw(10) << sd 
                              << " " << ( surf ? "YES" : "NO " )
                              << " " << gpresent("post", post)  
                              ;
                }
                count++ ; 
            }

            //if(surf) m_surf[tree][seqhis_]++ ;  
            if(surf) 
                 m_stat->increment(tree, seqhis_) ;  
       }

        LOG(info) << "OpticksEventAna::countExcursions"
                  << " pho_num " << m_pho_num
                  << " dbgseqhis " << std::hex << m_dbgseqhis << std::dec 
                  << " dbgseqhis " << OpticksFlags::FlagSequence( m_dbgseqhis, true )
                  << " count " << count 
                  ;
     
   }

}





void OpticksEventAna::dumpStepByStepCSGExcursions()
{
    unsigned count = 0 ;
    unsigned dumpmax = 10 ;
    for(unsigned i=0 ; i < m_pho_num ; i++)
    { 
         unsigned photon_id=i ; 
         unsigned long long seqhis_ = m_seq->getValue(photon_id,0,0);
       
         if(seqhis_ == m_dbgseqhis)
         {
             count++ ; 
             if(count < dumpmax)
             dumpStepByStepCSGExcursions(photon_id) ;
         }
    }

    LOG(info) << "OpticksEventAna::dumpStepByStepCSGExcursions"
              << " m_dbgseqhis " 
              << std::setw(16) << std::hex << m_dbgseqhis << std::dec
              << " count " << count 
              << " dumpmax " << dumpmax 
              ;
}

void OpticksEventAna::dumpStepByStepCSGExcursions(unsigned photon_id )
{ 
    unsigned long long seqhis_ = m_seq->getValue(photon_id,0,0);
    //unsigned long long seqmat_ = m_seq->getValue(i,0,1);

    std::vector<NRec> recs ; 
    glm::vec4 ldd = m_records->getLengthDistanceDurationRecs(recs, photon_id ); 

    LOG(info) << "OpticksEventAna::dumpStepByStepCSGExcursions"
              << " photon_id " << photon_id 
              << " num_recs " << recs.size()
              << " num_tree " << m_tree_num
              << " seqhis " << std::setw(16) << std::hex << seqhis_ << std::dec
              << " " << std::setw(64) << OpticksFlags::FlagSequence( seqhis_, true )
              ;
    
    for(unsigned p=0 ; p < recs.size() ; p++)
    {         
        const glm::vec4& post = recs[p].post  ;
        std::cout << gpresent_( "post", post ) ; 

        glm::vec4 lpos(post) ;
        lpos.w = 1.f ;  
        if(post.w > 0.f ) 
        {
            for(unsigned tree=0 ; tree < m_tree_num ; tree++)
            {       
                float sd = m_sdflist[tree](lpos.x, lpos.y, lpos.z);
                std::cout << " " << std::setw(10) << sd ;
            }
        }
        std::cout << std::endl ; 
    }
}





