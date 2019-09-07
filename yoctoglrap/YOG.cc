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

#include <cassert>
#include <sstream>
#include <iomanip>

#include "PLOG.hh"
#include "BStr.hh"
#include "NGLM.hpp"
#include "YOG.hh"

namespace YOG {

std::string Mt::desc() const 
{
    std::stringstream ss ; 
    ss
        << "Mt "
        << std::setw(30) << name
        ; 
    return ss.str();
}

std::string Mh::desc() const 
{
    std::stringstream ss ; 
    ss
        << "Mh "
        << std::setw(4) << lvIdx 
        << " : "
        << std::setw(30) << soName
        << " "
        << lvName 
        ; 
    return ss.str();
}

std::string Pr::desc() const 
{
    std::stringstream ss ; 
    ss
        << "Pr "
        << std::setw(4) << lvIdx 
        << " : "
        << std::setw(4) << mtIdx 
        ; 
    return ss.str();
}

std::string Nd::desc() const 
{
    std::stringstream ss ; 
    ss
        << "Nd "
        << " ndIdx:" << std::setw(4) << ndIdx 
        << " prIdx:" << std::setw(4) << prIdx 
        << " nch:"   << std::setw(4) << children.size()
        << " par:"   << std::setw(4) << parent  
        ; 
    return ss.str();
}

Sc::Sc(int root_)
   :
   root(root_)
{
}

std::string Sc::desc() const 
{
    std::stringstream ss ; 
    ss
        << "Sc "
        << " nodes:" << std::setw(4) << nodes.size()
        << " meshes:" << std::setw(4) << meshes.size()
        << " prims:" << std::setw(4) << prims.size()
        ; 
    return ss.str();
}

int Sc::lv2so(int lvIdx) const   // find local mesh index from the external lvIdx 
{
    int index(-1); 
    unsigned count(0); 
    for(int i=0 ; i < int(meshes.size()) ; i++)
    {
       const Mh* mh = meshes[i] ; 
       if(mh->lvIdx == lvIdx ) 
       {
           index = i ; 
           count++ ; 
       } 
    }
    assert( count == 1);
    return index ;  
}

bool Sc::has_mesh(int lvIdx) const 
{
    unsigned count(0); 
    for(int i=0 ; i < int(meshes.size()) ; i++)
    {
       const Mh* mh = meshes[i] ; 
       if(mh->lvIdx == lvIdx ) count++ ; 
    }
    assert( count == 0 || count == 1); 
    return count == 1 ; 
}


int Sc::find_prim( int lvIdx, int mtIdx ) const  // return -1 if not present, or the index if found
{
    unsigned count(0); 
    int prIdx = -1 ; 
    for(int i=0 ; i < int(prims.size()) ; i++)
    {
       const Pr& pr = prims[i] ; 
       if(pr.lvIdx == lvIdx && pr.mtIdx == mtIdx ) 
       {
           prIdx = i ; 
           count++ ; 
       }
    }
    assert( count == 0 || count == 1); 
    return prIdx ; 
}

int Sc::add_prim( int lvIdx, int mtIdx )  // only adds if not already present, returns prIdx  
{
    int prIdx = find_prim(lvIdx, mtIdx); 
    if( prIdx == -1) 
    {
        prIdx = prims.size();  
        prims.push_back( Pr(lvIdx, mtIdx) ) ; 
    }
    return prIdx ; 
}




/**
add_mesh
---------

* only adds if no mesh with lvIdx is present already 


* mtIdx is there because glTF mesh primitives carry a material index, 
  was initially surprised by this : but when realise that 
  the glTF mesh is very lightweight because it refers to accessors 
  for the data it makes more sense

  * but nevertheless this is a bit of a pain for X4PhysicalVolume 
    and trying to do a convertSolids before the convertStructure ...
    leaving it -1 : to be filled at point of use at node level 

**/

int Sc::add_mesh( 
                 int lvIdx, 
                 const std::string& lvName, 
                 const std::string& soName)
{
    int soIdx = -1 ; 
    if(!has_mesh(lvIdx))
    {
        soIdx = meshes.size(); 

        nnode* csgnode = NULL ; 
        const NCSG*  csg  = NULL ; 
        const GMesh* mesh  = NULL ; 
        const NPY<float>* vtx  = NULL ; 
        const NPY<unsigned>* idx  = NULL ; 

        meshes.push_back(new Mh { lvIdx, lvName, soName, soIdx, csgnode, csg, mesh, vtx, idx }) ;
    }
    int soIdx2 = lv2so(lvIdx);
    if(soIdx > -1 ) assert( soIdx2 == soIdx ) ; // when a new mesh is added, can check local indices match
    return soIdx2 ; 
}





int Sc::get_material_idx(const std::string& matName) const 
{
    int idx = -1 ; 
    unsigned count(0); 
    for(int i=0 ; i < int(materials.size()) ; i++)
    {
        const Mt* mt = materials[i];
        if(strcmp(mt->name.c_str(), matName.c_str()) == 0) 
        {
            idx = i ; 
            count++ ; 
        }
    }
    assert( count < 2 ); 
    return idx ; 
}


int Sc::add_material(const std::string& matName) // only adds if no material with that name is already present
{
    int idx = get_material_idx(matName) ;
    if( idx == -1 )
    {
        idx = materials.size();
        materials.push_back(new Mt { matName }); 
    }     
    return idx ; 
}



int Sc::get_num_meshes() const 
{
    return meshes.size();
}
int Sc::get_num_nodes() const 
{
    return nodes.size();
}
int Sc::get_num_prims() const 
{
    return prims.size();
}


int Sc::add_node(int lvIdx, 
                 int mtIdx, 
                 const std::string& pvName, 
                 const nmat4triple* transform, 
                 const std::string& boundary,
                 int depth, 
                 bool selected,
                 Nd* parent 
                )
{
     int prIdx = add_prim( lvIdx, mtIdx ) ;  // only adds when not already present

     int ndIdx = nodes.size() ;

     Nd* nd = new Nd {ndIdx, prIdx, transform, boundary, pvName, depth, selected, parent }  ;

     nodes.push_back(nd) ;

     if(parent) parent->children.push_back(ndIdx);

     return ndIdx ; 
}


Nd* Sc::get_node(int nodeIdx) const 
{
    assert( nodeIdx < int(nodes.size()) );
    return nodes[nodeIdx] ;  
}

Mh* Sc::get_mesh_for_node(int nodeIdx) const  // node->mesh association via nd->soIdx
{
    Nd* nd = get_node(nodeIdx) ;
    int prIdx = nd->prIdx ; 
    int lvIdx = prims[prIdx].lvIdx ;   
    Mh* mh = get_mesh(lvIdx); 
    return mh ;  
}

Mh* Sc::get_mesh(int lvIdx) const 
{
    assert( lvIdx < int(meshes.size()) ) ; 
    Mh* mh = meshes[lvIdx];  
    assert( mh );
    return mh ;  
}






int Sc::add_test_node(int lvIdx)
{
    int mtIdx = lvIdx ; 
    std::string pvName = BStr::concat<int>("pv", lvIdx, NULL) ;   
    const nmat4triple* transform = NULL ; 
    std::string boundary = BStr::concat<int>("bd", lvIdx, NULL) ;   
    int depth = 0 ; 
    bool selected = true ;  
    Nd* parent = NULL ;  

    int ndIdx = add_node(lvIdx, 
                         mtIdx,
                         pvName, 
                         transform, 
                         boundary,
                         depth, 
                         selected,
                         parent
                         );  

    return ndIdx ; 
}



} // namespace
