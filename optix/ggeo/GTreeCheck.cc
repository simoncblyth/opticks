#include  "GTreeCheck.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GMatrix.hh"

#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GTreeCheck::init()
{
}

void GTreeCheck::traverse()
{
    GSolid* root = m_ggeo->getSolid(0);
    traverse(root, 0);
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

    LOG(info) << "GTreeCheck::traverse " 
              << " count " << std::setw(6) << m_count  
              << " delta*1e6 " << std::setprecision(6) << std::fixed << delta*1e6 
              << " name " << node->getName() 
              ;

    assert(delta < 1e-6) ;

    m_count++ ; 

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1 );
}




