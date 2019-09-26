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


#include <sstream>
#include <algorithm>
#include "NTreeAnalyse.hpp"
#include "NNodeCollector.hpp"
#include "NGrid.hpp"


template <typename T>
std::string  NTreeAnalyse<T>::Desc(const T* root)  // static
{
    NTreeAnalyse<T> ana(root); 
    return ana.desc(); 
}

template <typename T>
std::string  NTreeAnalyse<T>::Brief(const T* root)  // static
{
    NTreeAnalyse<T> ana(root); 
    return ana.brief(); 
}




template <typename T>
NTreeAnalyse<T>::NTreeAnalyse(const T* root_)
    :
    root(root_),
    height(depth_(true)),
    nodes(new NNodeCollector<T>(root)),
    count(nodes->inorder.size()),
    grid(new NGrid<T>(height+1, count))
{
    init(); 
}


template <typename T>
NTreeAnalyse<T>::~NTreeAnalyse()
{
    delete nodes ; 
    delete grid ; 
}


template <typename T>
void NTreeAnalyse<T>::init()
{
    initGrid();
}

template <typename T>
void NTreeAnalyse<T>::initGrid()
{
    for(unsigned i=0 ; i < count ; i++)
    {
        const T* node = nodes->inorder[i] ;  
        grid->set(node->depth, i, node) ; 
    }
}


template <typename T>
unsigned NTreeAnalyse<T>::depth_(bool label)
{
    return depth_r(root, 0, label);
}

template <typename T>
unsigned NTreeAnalyse<T>::depth_r(const T* node, unsigned depth, bool label)
{
     if(node == NULL) return depth ; 
     if(label) const_cast<T*>(node)->depth = depth ; 
     if(node->left == NULL && node->right == NULL) return depth ; 

     unsigned ldepth = depth_r(node->left,  depth+1, label ); 
     unsigned rdepth = depth_r(node->right, depth+1, label ); 
     return std::max(ldepth, rdepth) ;
}


template <typename T>
std::string NTreeAnalyse<T>::desc() const 
{
    std::stringstream ss ; 
    ss 
       << "NTreeAnalyse"
       << " height " << height 
       << " count " << count 
       << std::endl 
       << grid->desc()
       << std::endl
       << "inorder (left-to-right) " 
       << std::endl
       << nodes->desc_inorder() 
       ;

    return ss.str(); 
}


template <typename T>
std::string NTreeAnalyse<T>::brief() const 
{
    std::stringstream ss ; 
    ss 
       << "NTreeAnalyse"
       << " height " << height 
       << " count " << count 
       ;

    return ss.str(); 
}














#include "No.hpp"
#include "NNode.hpp"

template struct NTreeAnalyse<no> ; 
template struct NTreeAnalyse<nnode> ; 



