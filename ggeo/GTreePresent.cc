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

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iterator>

// brap-
#include "BFile.hh"

#include "GTreePresent.hh"
#include "GMesh.hh"
#include "GVolume.hh"

#include "PLOG.hh"


// elimnate state that is only used once, instead pass in as arguments when needed


const char* GTreePresent::NONAME = "noname" ; 

GTreePresent::GTreePresent(unsigned int depth_max, unsigned int sibling_max) 
       :
       m_depth_max(depth_max),
       m_sibling_max(sibling_max)
{
}


void GTreePresent::traverse(GNode* top)
{
    if(!top)
    {
        LOG(error) << "GTreePresent::traverse top NULL " ;
        return ; 
    }
    traverse(top, 0, 0, 0, false);
}

void GTreePresent::traverse( GNode* node, unsigned int depth, unsigned int numSibling, unsigned int siblingIndex, bool elide )
{

    std::string indent(depth, ' ');
    unsigned int numChildren = node->getNumChildren() ;
    int nodeIndex   = node->getIndex() ; 
    unsigned int ridx = node->getRepeatIndex();   
    const char* name = node->getName() ; 

    if(!name) name = NONAME ; 

    //GVolume* solid = dynamic_cast<GVolume*>(node) ;
    //bool selected = solid->isSelected();

    std::stringstream ss ; 
    ss 
       << "  " << std::setw(7) << nodeIndex 
       << " [" << std::setw(3) << depth 
       << ":"  << std::setw(4) << siblingIndex 
       << "/"  << std::setw(4) << numSibling
       << "] " << std::setw(4) << numChildren   
       << " (" << std::setw(2) << ridx 
       << ") " << indent 
       << "  " << name
       << "  " << node->getMesh()->getName()
       << "  " << ( elide ? "..." : " " ) 
       ;

    m_flat.push_back(ss.str()); 
    if(elide) return ; 

    unsigned int hmax = m_sibling_max/2 ;

    if(depth < m_depth_max)
    {
       if( numChildren < 2*hmax )
       { 
           for(unsigned int i = 0; i < numChildren ; i++) 
               traverse(node->getChild(i), depth + 1, numChildren, i, false );
       }
       else
       {
           for(unsigned int i = 0; i < hmax ; i++) 
               traverse(node->getChild(i), depth + 1, numChildren, i, false);

           traverse(node->getChild(hmax), depth + 1, numChildren, hmax, true );

           for(unsigned int i = numChildren - hmax ; i < numChildren ; i++) 
               traverse(node->getChild(i), depth + 1, numChildren, i, false );
       }
    }
}



void GTreePresent::dump(const char* msg)
{
    LOG(info) << msg ; 
    std::copy(m_flat.begin(), m_flat.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
}

void GTreePresent::write(const char* dir, const char* reldir)
{
    BFile::CreateDir(dir, reldir);

    std::string txtpath = BFile::FormPath(dir, reldir, "GTreePresent.txt");
    const char* path = txtpath.c_str();
    LOG(debug) << "GTreePresent::write " << path ;  
    { 
        std::ofstream fp(path, std::ios::out );
        std::copy(m_flat.begin(), m_flat.end(), std::ostream_iterator<std::string>(fp, "\n"));
    }
    LOG(debug) << "GTreePresent::write " << path << "DONE"  ;  

}



