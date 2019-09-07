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

#include <cmath>
#include <stack>
#include "NQuad.hpp"
#include "NOctNode.hpp"

#include "PLOG.hh"


const nivec3 NOctNode::OFFSETS[] = 
{
   nivec3(0,0,0),
   nivec3(0,0,1),
   nivec3(0,1,0),
   nivec3(0,1,1),
   nivec3(1,0,0),
   nivec3(1,0,1),
   nivec3(1,1,0),
   nivec3(1,1,1)
};



NOctNode::NOctNode( NOctNode_t type ) : 
    type(type), 
    min(0,0,0), 
    size(0), 
    scale(1.f),
    data(NULL)
{
    for(int i=0 ; i < 8 ; i++) child[i] = NULL ; 
}  


std::string NOctNode::desc()
{
    std::stringstream ss ; 
    nivec3 max = min + size*OFFSETS[7] ; 
    ss 
        << NOctNodeEnum::NOCTName(type)
        <<  " min " << min.desc() 
        <<  " max " << max.desc() 
        << " size " << size
        << " scale " << scale
        ;
    return ss.str(); 
}

void NOctNode::Traverse(NOctNode* node, int depth)
{
    std::cout << "T " << std::setw(2) << depth << " " << node->desc() << std::endl ;  
    for(int i=0 ; i < 8 ; i++) if(node->child[i] != NULL) Traverse(node->child[i], depth+1 );
}

int NOctNode::TraverseIt(NOctNode* root)
{
    std::stack<NOctNode*> s ;
    s.push(root);

    int count = 0 ; 
    while(!s.empty())
    {
        NOctNode* node = s.top();
        s.pop();

        count += 1 ; 
        std::cout << "T " << std::setw(2) << count << " " << node->desc() << std::endl ;  

        for(int i=0 ; i < 8 ; i++)
        {
            NOctNode* child = node->child[i] ;
            if(child != NULL) s.push(child) ;
        }
    }
    return count ; 
}




NOctNode* NOctNode::Construct(const nivec3& min, const int size, const float scale)
{
    NOctNode* root = new NOctNode(NOCT_INTERNAL) ;  
    root->size = size ; 
    root->min = min ;  
    root->scale = scale ; 

    LOG(info) << "NOctNode::Construct" 
              << " root " << root->desc()
              ; 

    Construct_r( root );

    return root ; 
}


float NOctNode::sdf( const nivec3& min, const float scale)
{
    nvec3 smin = make_nvec3(min.x*scale, min.y*scale, min.z*scale);

    nvec3 center = {0.f,0.f,0.f} ;
    float radius = 20.f ;

    nvec3 dist = smin - center ; 
    return sqrt( dist*dist ) - radius ;      
}

int NOctNode::sdf_corners( const nivec3& min, const int size, const float scale)
{
    int corners = 0 ;
    for(int i=0 ; i < 8 ; i++)
    {
        nivec3 pos = min + OFFSETS[i]*size ; 
        float d = sdf( pos, scale) ;
        int inside(d < 0.f) ; 
        corners |= ( inside << i ) ; 
    }
    return corners ; 
}



NOctNode* NOctNode::ConstructLeaf( NOctNode* node )
{
    int corners = sdf_corners( node->min, node->size, node->scale ); 
    if(corners == 0 || corners == 255) 
    {
        delete node ;  
        return NULL ; 
    }



    NLeafData* data = new NLeafData ; 
    data->corners = corners ;

    node->data = data ;
    node->type = NOCT_LEAF ; 

    LOG(info) << "NOctNode::ConstructLeaf"
              << " node " << node->desc()
              << " corners " << std::hex << corners << std::dec 
              ;

    return node ; 
}


NOctNode* NOctNode::Construct_r( NOctNode* parent)
{
    if(parent->size == 1) return ConstructLeaf(parent);

    int child_size = parent->size / 2 ; 
    int nchild(0);  

    for(int i=0 ; i < 8 ; i++)
    {
         NOctNode* candidate = new NOctNode(NOCT_INTERNAL) ;
         candidate->size = child_size ;
         candidate->min  = parent->min + OFFSETS[i]*child_size ;  
         candidate->scale = parent->scale ; 

         NOctNode* child = Construct_r( candidate );
    
         if(child != NULL) nchild++ ; 

         parent->child[i] = child ; 
    } 

    if( nchild == 0)
    {
         delete parent ; 
         parent = NULL ; 
    }

    return parent ;  
}

 
    

