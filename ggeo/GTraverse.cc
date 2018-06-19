#include <cstddef>

#include "Counts.hpp"

#include "GGeo.hh"
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GVolume.hh"

#include "GTraverse.hh"

GTraverse::GTraverse(GGeo* ggeo) 
    :
    m_ggeo(ggeo),
    m_blib(NULL),
    m_mlib(NULL),
    m_materials_count(NULL)
{
    init();
}

void GTraverse::init()
{
    m_blib = m_ggeo->getBndLib();
    m_mlib = m_ggeo->getMaterialLib();
    m_materials_count = new Counts<unsigned int>("materials");
}

void GTraverse::traverse()
{
    GVolume* root = m_ggeo->getVolume(0);
    traverse(root, 0);

    m_materials_count->sort(false);
    m_materials_count->dump();
}

void GTraverse::traverse( GNode* node, unsigned int depth)
{
    GVolume* volume = dynamic_cast<GVolume*>(node) ;

    bool selected = volume->isSelected();
    if(selected)
    {
        unsigned boundary = volume->getBoundary();
        guint4 bnd = m_blib->getBnd(boundary);
        const char* im = m_mlib->getName(bnd.x);
        const char* om = m_mlib->getName(bnd.y);
        m_materials_count->add(im);
        m_materials_count->add(om);
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1 );
}



