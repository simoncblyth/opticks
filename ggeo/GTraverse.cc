/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
    const GVolume* root = m_ggeo->getVolume(0);
    traverse(root, 0);

    m_materials_count->sort(false);
    m_materials_count->dump();
}

void GTraverse::traverse( const GNode* node, unsigned int depth)
{
    const GVolume* volume = dynamic_cast<const GVolume*>(node) ;

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



