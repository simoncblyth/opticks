#include <sstream>

#include "NPY.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "RecordsNPY.hpp"
#include "NCSGList.hpp"
#include "NCSGIntersect.hpp"
#include "NCSG.hpp"
#include "NGeoTestConfig.hpp"

#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"
#include "OpticksEventAna.hh"
#include "PLOG.hh"

OpticksEventAna::OpticksEventAna( Opticks* ok, OpticksEvent* evt, NCSGList* csglist )
    :
    m_ok(ok),
    m_epsilon(0.1f),
    m_dbgseqhis(m_ok->getDbgSeqhis()),
    m_dbgseqmat(m_ok->getDbgSeqmat()),

    m_seqmap_his(0ull),
    m_seqmap_val(0ull),
    m_seqmap_has(false),
    m_seqhis_select(false),  // seqmap trumps dbgseqhis

    m_evt(evt),
    m_evtgtc(evt->getTestConfig()),  // NGeoTestConfig reconstructed from evt metadata

    m_csglist(csglist),
    m_tree_num(csglist->getNumTrees()),
    m_csgi(new NCSGIntersect[m_tree_num]),

    m_records(m_evt->getRecordsNPY()),  
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

    if(m_evtgtc) initOverride(m_evtgtc);
    initSeqMap(); 
    
    for(unsigned i=0 ; i < m_tree_num ; i++)
    {
        NCSG* csg = m_csglist->getTree(i);
        m_csgi[i].init(csg) ;
    }

    countPointExcursions();
    checkPointExcursions();
}


void OpticksEventAna::initOverride(NGeoTestConfig* gtc)
{
    const char* autoseqmap = gtc->getAutoSeqMap(); 
    LOG(info) << " autoseqmap " << autoseqmap ; 
    m_ok->setSeqMapString(autoseqmap);
}

void OpticksEventAna::initSeqMap()
{
    m_seqmap_his = 0ull ; 
    m_seqmap_val = 0ull ;
    m_seqmap_has = m_ok->getSeqMap(m_seqmap_his, m_seqmap_val) ;
    m_seqhis_select = m_seqmap_has ? m_seqmap_his : m_dbgseqhis ;   // seqmap trumps dbgseqhis
}



void OpticksEventAna::countPointExcursions()
{
    for(unsigned i=0 ; i < m_pho_num ; i++)
    { 
        unsigned photon_id=i ; 
        unsigned long long seqhis_ = m_seq->getValue(photon_id,0,0);   
        if(seqhis_ != m_seqhis_select) continue ; 

        std::vector<NRec> recs ; 
        glm::vec4 ldd = m_records->getLengthDistanceDurationRecs(recs, photon_id ); 

        for(unsigned p=0 ; p < recs.size() ; p++)
        {         
            const glm::vec4& post = recs[p].post  ;
            if(post.w == 0.f ) continue ; 
            for(unsigned tree=0 ; tree < m_tree_num ; tree++) m_csgi[tree].add(p, post);
        }
    }
}




void OpticksEventAna::checkPointExcursions()
{
    if(!m_seqmap_has) return ; 

     const std::string& seqmap = m_ok->getSeqMapString();

     LOG(info) 
           << " seqmap " << seqmap
           ;

     LOG(info)
          << " seqmap_his " << std::setw(16) << std::hex << m_seqmap_his << std::dec 
          << " seqmap_val " << std::setw(16) << std::hex << m_seqmap_val << std::dec 
          ;



    unsigned num_excursions = 0 ; 


    glm::vec4 xdist0(-0.1);  // kludge hard duplication of emitconfig.deltashift 0.1mm with emit=-1 inwards photons
    glm::vec4 xdist1(0.);    

    for(unsigned p=0 ; p < 16 ; p++)
    {
        const char* abbrev = OpticksFlags::PointAbbrev(m_seqmap_his, p );
        unsigned val1 = OpticksFlags::PointVal1(m_seqmap_val, p );

        if(val1 == 0 ) continue ; 

        unsigned tree = val1 - 1 ;
        const NCSGIntersect& csgi = m_csgi[tree] ; 
        unsigned count = csgi._count[p] ;

        const glm::vec4& dist = csgi._dist[p] ; 
        const glm::vec4& xdist = p == 0 ? xdist0 : xdist1 ;

        float df = nglmext::compDiff(dist, xdist ); 
        // max absolute difference of min/max/avg values 

        bool excursion = df > m_epsilon ;
        if(excursion) num_excursions++ ; 

 
        std::cout 
             << " p " << std::setw(2) << p  
             << " abbrev " << std::setw(2) << abbrev 
             << " val1 " << std::setw(2) << val1 
             << " tree " << tree
             << " count " << count
             << " dist " << gpresent( dist )
             << " xdist " << gpresent( xdist )
             << " df " << std::setw(10) << std::fixed << df 
             << " " << ( excursion ? "EXCURSION" : "expected" )
             << std::endl ; 
             ;

        assert( count > 0 ); 

    }   

    bool has_excursions = num_excursions > 0 ; 
    LOG( has_excursions ? fatal : info ) << " num_excursions " << num_excursions ; 
    //assert( !has_excursions );
    if(has_excursions) 
    {
        m_ok->setRC(202, "OpticksEventAna::checkPointExcursions found some" ) ;
    }


}


/*

Emitconfig here using a delta shift of 0.1, hmm does that config  travel with the evt ?
Kludge it with xdist(0.2)

2017-11-22 15:07:32.385 INFO  [6385639] [OpticksEventAna::checkPointExcursions@91]  dbgseqhismap TO:0,SR:1,SA:0
2017-11-22 15:07:32.385 INFO  [6385639] [OpticksEventAna::checkPointExcursions@95]  seqmap_his              8ad seqmap_val              121
 p  0 abbrev TO val1  1 tree 0 count 19700 csgi.desc_dist [p: 0](  19700)(     -0.100    -0.100    -0.100       0.000) mm
 p  1 abbrev SR val1  2 tree 1 count 19700 csgi.desc_dist [p: 1](  19700)(     -0.022     0.023     0.004       0.000) mm
 p  2 abbrev SA val1  1 tree 0 count 19700 csgi.desc_dist [p: 2](  19700)(     -0.008    -0.008    -0.008       0.000) mm


2017-11-22 15:07:33.038 INFO  [6385639] [OpticksEventAna::checkPointExcursions@95]  seqmap_his              8ad seqmap_val              121
 p  0 abbrev TO val1  1 tree 0 count 19699 csgi.desc_dist [p: 0](  19699)(     -0.100    -0.100    -0.100       0.000) mm
 p  1 abbrev SR val1  2 tree 1 count 19699 csgi.desc_dist [p: 1](  19699)(   -198.534     0.023   -56.019       0.000) mm
 p  2 abbrev SA val1  1 tree 0 count 19699 csgi.desc_dist [p: 2](  19699)(     -0.008    -0.008    -0.008       0.000) mm



*/





void OpticksEventAna::dumpPointExcursions(const char* msg)
{
    LOG(info) << msg 
              << " seqhis_select "
              << " " << std::hex << m_seqhis_select << std::dec 
              << " " << OpticksFlags::FlagSequence( m_seqhis_select, true )
              ;

    std::cout << "min/max/avg signed-distance(mm) and time(ns) of each photon step point from each NCSG tree" << std::endl ; 
 
    for(unsigned p=0 ; p < 16 ; p++)
    {
        if(m_csgi[0]._count[p] == 0) continue ;  
        for(unsigned t=0 ; t < m_tree_num ; t++) 
        {
            std::cout << m_csgi[t].desc_dist(p) ; 
        }
        std::cout << std::endl ; 
    }   


    /* 
    for(unsigned p=0 ; p < 16 ; p++)
    {
        if(m_csgi[0]._count[p] == 0) continue ;  
        for(unsigned t=0 ; t < m_tree_num ; t++) 
        {
            std::cout << m_csgi[t].desc_time(p) ; 
        }
        std::cout << std::endl ; 
    } 
    */  

}



void OpticksEventAna::dump(const char* msg)
{
    LOG(info) << msg << " " << desc() ; 
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



