#include  "GTraverse.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GBoundary.hh"
#include "Counts.hpp"


void GTraverse::init()
{
    m_materials_count = new Counts<unsigned int>("materials");
}

void GTraverse::traverse()
{
    GSolid* root = m_ggeo->getSolid(0);
    traverse(root, 0);

    m_materials_count->sort(false);
    m_materials_count->dump();
}

void GTraverse::traverse( GNode* node, unsigned int depth)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    bool selected = solid->isSelected();
    if(selected)
    {
        GBoundary* boundary = solid->getBoundary();
        GPropertyMap<float>* imat = boundary->getInnerMaterial() ;
        GPropertyMap<float>* omat = boundary->getOuterMaterial() ;

        std::string im = imat->getShortName();
        std::string om = omat->getShortName();

        m_materials_count->add(im.c_str());
        m_materials_count->add(om.c_str());
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1 );
}




