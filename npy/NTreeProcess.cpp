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
#include <vector>

#include "SSys.hh"
#include "PLOG.hh"


#include "NPY.hpp"

#include "NTreeBalance.hpp"
#include "NTreePositive.hpp"
#include "NTreeAnalyse.hpp"
#include "NTreeProcess.hpp"
#ifdef WITH_CHOPPER
#include "NTreeChopper.hpp"
#endif

#include "PLOG.hh"

template <typename T>
const plog::Severity NTreeProcess<T>::LEVEL = PLOG::EnvLevel("NTreeProcess", "DEBUG") ; 

template <typename T>
unsigned NTreeProcess<T>::MaxHeight0 = 3 ;   // was discrepantly 4 previously   

template <typename T>
std::vector<int>* NTreeProcess<T>::LVList = SSys::getenvintvec("NTREEPROCESS_LVLIST") ;  

template <typename T>
NPY<unsigned>* NTreeProcess<T>::ProcBuffer = NULL ;  

template <typename T>
void NTreeProcess<T>::SaveBuffer(const char* path)
{
    ProcBuffer->save(path); 
}
template <typename T>
void NTreeProcess<T>::SaveBuffer(const char* dir, const char* name)
{
    ProcBuffer->save(dir, name); 
}



/**
NTreeProcess<T>::Process
--------------------------



**/


template <typename T>
T* NTreeProcess<T>::Process( T* root_ , int soIdx, int lvIdx )  // static
{
 
    if( ProcBuffer == NULL ) ProcBuffer = NPY<unsigned>::make(0,4) ; 

    bool listed = LVList != NULL && std::find(LVList->begin(), LVList->end(), lvIdx ) != LVList->end() ; 

    if(listed) LOG(info) << "before\n" << NTreeAnalyse<T>::Desc(root_) ; 
     // dump it here, prior to the inplace positivization 
 
    unsigned height0 = root_->maxdepth(); 

    NTreeProcess<T> proc(root_ ); 

    assert( height0 == proc.balancer->height0 ); 

    T* result = proc.result ; 

    unsigned height1 = result->maxdepth();   

    if(listed) 
    {
         LOG(info) << "after\n" << NTreeAnalyse<T>::Desc(result) ; 
         LOG(info) 
         << " soIdx " << soIdx
         << " lvIdx " << lvIdx
         << " height0 " << height0
         << " height1 " << height1
         << " " << ( listed ? "### LISTED" : "" ) 
         ;
    }

    if(ProcBuffer) ProcBuffer->add(soIdx, lvIdx, height0, height1);   

    return result ; 
} 

/**
NTreeProcess::NTreeProcess
---------------------------- 

**/

template <typename T>
NTreeProcess<T>::NTreeProcess( T* root_ ) 
    :
    root(root_),
    balanced(NULL),
    result(NULL),
#ifdef WITH_CHOPPER
    chopper(new NTreeChopper<T>(root_,1e-6)),
#endif
    balancer(new NTreeBalance<T>(root_)),    // writes depth, subdepth to all nodes
    positiver(NULL)
{
    init();
}

/**
NTreeProcess::init
---------------------

After a CSG tree has been positivized it is straightforward to 
decide whether to include a primitive node bbox into the tree bbox 
as can simply check if a primitive is complemented and when it is 
exclude its bbox. This makes the sensible assumption that 
effective node subtractions can never increase the tree bbox. 

Prior to Oct 25, 2021 tree positivization was only applied to 
trees of a height greater than MaxHeight0(=3). 
From that date (Opticks::GEOCACHE_CODE_VERSION = 15) 
positivization is applied to all trees irrespective of 
their heights. Note that this only has an effect on trees 
with subtraction operators. 

**/

template <typename T>
void NTreeProcess<T>::init()
{
    positiver = new NTreePositive<T>(root) ;  // inplace changes operator types and sets complements on primitives
    if(balancer->height0 > MaxHeight0 )
    {
        //formerly the positiver ctor was done here, restricting its goodness to trees that need to be balanced
        balanced = balancer->create_balanced() ;  
        result = balanced ; 

        if(balancer->is_unable_to_balance())
        {
            LOG(error) 
                << " unable_to_balance treeidx " << root->get_treeidx() 
                << " NTreeAnalyse::Desc " << std::endl 
                << NTreeAnalyse<T>::Desc(root) 
                ; 
        }
    }
    else
    {
        result = root ; 
    }
}

#include "No.hpp"
#include "NNode.hpp"

#ifdef WITH_CHOPPER
#else
template struct NTreeProcess<no> ; 
#endif


template struct NTreeProcess<nnode> ; 


