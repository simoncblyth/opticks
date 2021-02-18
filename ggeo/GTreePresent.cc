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

void GTreePresent::traverse(const GNode* top)
{
    if(!top)
    {
        LOG(error) << "GTreePresent::traverse top NULL " ;
        return ; 
    }
    traverse(top, 0, 0, 0, false);
}

/**
GTreePresent::traverse
------------------------

::

       nodeIndex 
        |   depth 
        |    |   siblingIndex
        |    |    |   numSibling 
        |    |    |    |    numChildren
        |    |    |    |     |  ridx 
        |    |    |    |     |   | 
        |    |    |    |     |   | 
        |    |    |    |     |   | 
        |    |    |    |     |   |
        0 [  0:   0/   0]    2 ( 0)   pWorld  sWorld   
        1 [  1:   0/   2]    1 ( 0)    pTopRock  sTopRock   
        2 [  2:   0/   1]    2 ( 0)     pExpHall  sExpHall   
        3 [  3:   0/   2]    3 ( 0)      lUpperChimney_phys  Upper_Chimney   
        4 [  4:   0/   3]    0 ( 0)       pUpperChimneyLS  Upper_LS_tube   
        5 [  4:   1/   3]    0 ( 0)       pUpperChimneySteel  Upper_Steel_tube   
        6 [  4:   2/   3]    0 ( 0)       pUpperChimneyTyvek  Upper_Tyvek_tube   
        7 [  3:   1/   2]   63 ( 0)      pTopTracker  sAirTT   
        8 [  4:   0/  63]    2 ( 0)       pWall_000_  sWall   
        9 [  5:   0/   2]    4 ( 0)        pPlane_0_ff_  sPlane   
       10 [  6:   0/   4]    1 ( 9)         pPanel_0_f_  sPanel   
       11 [  7:   0/   1]   64 ( 9)          pPanelTape  sPanelTape   




       nodeIndex 
        |   depth 
        |    |   siblingIndex
        |    |    |       numSibling 
        |    |    |        |    numChildren
        |    |    |        |     |  ridx 
        |    |    |        |     |   | 
        |    |    |        |     |   | 
        |    |    |        |     |   | 
        |    |    |        |     |   |
    65715 [  8:  63/      64]    1  ( 9)           pCoating_63_  sBar
    65716 [  9:   0/       1]    0  ( 9)            pBar  sBar
    65717 [  1:   1/       2]    1  ( 0)    pBtmRock  sBottomRock
    65718 [  2:   0/       1]    1  ( 0)     pPoolLining  sPoolLining
    65719 [  3:   0/       1] 2401  ( 0)      pOuterWaterPool  sOuterWaterPool
    65720 [  4:   0/    2401]    1  ( 0)       pCentralDetector  sReflectorInCD
    65721 [  5:   0/       1] 45686 ( 0)        pInnerWater  sInnerWater
    65722 [  6:   0/   45686]    1  ( 0)         pAcylic  sAcrylic
    65723 [  7:   0/       1]   54  ( 0)          pTarget  sTarget
    65724 [  8:   0/      54]    0  ( 0)           lSJCLSanchor_phys  solidSJCLSanchor
    65725 [  8:   1/      54]    0  ( 0)           lSJCLSanchor_phys  solidSJCLSanchor
    65726 [  8:   2/      54]    0  ( 0)           lSJFixture_phys  solidSJFixture
    65727 [  8:   3/      54]    0  ( 0)           lSJFixture_phys  solidSJFixture
    65728 [  8:   4/      54]    0  ( 0)           lSJFixture_phys  solidSJFixture
    65729 [  8:   5/      54]    0  ( 0)           lSJFixture_phys  solidSJFixture
    ...
    65776 [  8:  52/      54]    0 ( 0)           lXJfixture_phys  solidXJfixture
    65777 [  8:  53/      54]    0 ( 0)           lXJfixture_phys  solidXJfixture
    65778 [  6:   1/   45686]    0 ( 8)         lSteel_phys  sStrut
    65779 [  6:   2/   45686]    0 ( 8)         lSteel_phys  sStrut
    65780 [  6:   3/   45686]    0 ( 8)         lSteel_phys  sStrut
    65781 [  6:   4/   45686]    0 ( 8)         lSteel_phys  sStrut
    65782 [  6:   5/   45686]    0 ( 8)         lSteel_phys  sStrut
    65783 [  6:   6/   45686]    0 ( 8)         lSteel_phys  sStrut
    65784 [  6:   7/   45686]    0 ( 8)         lSteel_phys  sStrut
    65785 [  6:   8/   45686]    0 ( 8)         lSteel_phys  sStrut

    * note that searching for depth markers eg "6:" is handy to navigate the tree




**/

void GTreePresent::traverse(const GNode* node, unsigned int depth, unsigned int numSibling, unsigned int siblingIndex, bool elide )
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



