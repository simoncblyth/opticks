#include  "GTreeCheck.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GMatrix.hh"
#include "Counts.hpp"

#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



void GTreeCheck::init()
{
    m_root = m_ggeo->getSolid(0);
    m_digest_count = new Counts<unsigned int>("progenyDigest");
}

void GTreeCheck::traverse()
{
    traverse(m_root, 0);

    m_digest_count->sort(false);
    //m_digest_count->dump();

    // minrep 120 removes repeats from headonPMT, calibration sources and RPC leaving just PMTs 
    findRepeatCandidates(120); 
    dumpRepeatCandidates();
}


void GTreeCheck::dumpRepeatCandidates()
{
    LOG(info) << "GTreeCheck::dumpRepeatCandidates " ;
    for(unsigned int i=0 ; i < m_repeat_candidates.size() ; i++) dumpRepeatCandidate(i) ;
}

void GTreeCheck::dumpRepeatCandidate(unsigned int index)
{
    std::string pdig = m_repeat_candidates[index];
    unsigned int ndig = m_digest_count->getCount(pdig.c_str());

    GNode* first = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    std::vector<GNode*> placements = m_root->findAllProgenyDigest(pdig);
    std::cout  
                  << " pdig "  << std::setw(32) << pdig  
                  << " ndig "  << std::setw(6) << std::dec << ndig
                  << " nprog " <<  std::setw(6) << std::dec << first->getProgenyCount() 
                  << " placements " << std::setw(6) << placements.size()
                  << " n "          <<  first->getName() 
                  << std::endl 
                  ;  

    assert(placements.size() == ndig );

    for(unsigned int i=0 ; i < placements.size() ; i++)
    {
        GNode* place = placements[i] ;
        GMatrix<float>* t = place->getTransform();
        std::cout 
               << " t " << t->brief() 
               << std::endl 
               ;  
    }
}

void GTreeCheck::findRepeatCandidates(unsigned int minrep)
{
    for(unsigned int i=0 ; i < m_digest_count->size() ; i++)
    {
        std::pair<std::string,unsigned int>&  kv = m_digest_count->get(i) ;

        std::string& pdig = kv.first ; 
        unsigned int ndig = kv.second ;                 // number of occurences of the progeny digest 
        GNode* node = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest

        if( ndig > minrep && node->getProgenyCount() > 0 )
        {
            LOG(info) 
                  << "GTreeCheck::findRepeatCandidates "
                  << " pdig "  << std::setw(32) << pdig  
                  << " ndig "  << std::setw(6) << ndig
                  << " nprog " <<  std::setw(6) << node->getProgenyCount() 
                  << " n "     <<  node->getName() 
                  ;  

            m_repeat_candidates.push_back(pdig);
        }   
    }

    // erase repeats that are enclosed within other repeats 
    // ie that have an ancestor which is also a repeat candidate

    m_repeat_candidates.erase(
         std::remove_if(m_repeat_candidates.begin(), m_repeat_candidates.end(), *this ),
         m_repeat_candidates.end()
    ); 
    

}

bool GTreeCheck::operator()(const std::string& dig)  
{
    bool cr = isContainedRepeat(dig, 3);
 
    if(cr) LOG(info) 
                  << "GTreeCheck::operator() "
                  << " pdig "  << std::setw(32) << dig  
                  << " disallowd as isContainedRepeat "
                  ;

    return cr ;  
} 

bool GTreeCheck::isContainedRepeat( const std::string& pdig, unsigned int levels ) const 
{
    // for the first node that matches the *pdig* progeny digest
    // look back *levels* ancestors to see if any of the immediate ancestors 
    // are also repeat candidates, if they are then this is a contained repeat
    // and is thus disallowed in favor of the ancestor that contains it 

    GNode* node = m_root->findProgenyDigest(pdig) ; 
    std::vector<GNode*>& ancestors = node->getAncestors();
    unsigned int asize = ancestors.size(); 

    for(unsigned int i=0 ; i < std::min(levels, asize) ; i++)
    {
        GNode* a = ancestors[asize - 1 - i] ;
        std::string& adig = a->getProgenyDigest();
        if(std::find(m_repeat_candidates.begin(), m_repeat_candidates.end(), adig ) != m_repeat_candidates.end())
        { 
            return true ;
        }
    }
    return false ; 
} 


void GTreeCheck::traverse( GNode* node, unsigned int depth)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    //bool selected = solid->isSelected();

    GMatrixF* gtransform = solid->getTransform();
    GMatrixF* ltransform = solid->getLevelTransform();
    GMatrixF* ctransform = solid->calculateTransform();

    //gtransform->Summary("GTreeCheck::traverse gtransform");
    //ctransform->Summary("GTreeCheck::traverse ctransform");

    float delta = gtransform->largestDiff(*ctransform);

    std::string& pdig = node->getProgenyDigest();
    unsigned int nprogeny = node->getProgenyCount() ;
        
    m_digest_count->add(pdig.c_str());

    if(nprogeny > 0 ) 
         LOG(info) 
              << "GTreeCheck::traverse " 
              << " count "     << std::setw(6) << m_count
              << " #progeny "  << std::setw(6) << nprogeny 
              << " pdig "      << std::setw(32) << pdig 
              << " delta*1e6 " << std::setprecision(6) << std::fixed << delta*1e6 
              << " name " << node->getName() 
              ;

    assert(delta < 1e-6) ;

    m_count++ ; 

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1 );
}




