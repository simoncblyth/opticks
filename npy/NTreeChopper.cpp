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
#include <csignal>
#include <map>

#include "SSys.hh"
#include "PLOG.hh"

#include "OpticksCSG.h"

#include "NPY.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NTreeChopper.hpp"


template <typename T>
const plog::Severity NTreeChopper<T>::LEVEL = PLOG::EnvLevel("NTreeChopper", "DEBUG"); 

template <typename T>
NTreeChopper<T>::NTreeChopper(T* root_, float epsilon_) 
     :
     root(root_),
     epsilon(epsilon_), 
     verbosity(SSys::getenvint("VERBOSITY",1)),
     enabled(true)
{
    root->check_tree( FEATURE_GTRANSFORMS | FEATURE_PARENT_LINKS );
    init();
}

template <typename T>
void NTreeChopper<T>::init()
{
    LOG(LEVEL) << "[" ; 

    root->collect_prim_for_edit(prim);  // recursive collector 
    update_prim_bb();                   // find z-order of prim using bb.min.z

    dump("NTreeChopper<nnode>::init");  


    LOG(LEVEL) << "] " << brief() ; 
}

template <typename T>
void NTreeChopper<T>::update_prim_bb()
{
    zorder.clear();
    bb.clear(); 
    for(unsigned i=0 ; i < prim.size() ; i++)
    {
        const T* p = prim[i] ; 

        nbbox pbb = p->bbox(); 
        bb.push_back(pbb);
        zorder.push_back(i);
    }
    std::sort(zorder.begin(), zorder.end(), *this );   // np.argsort style : sort the indices
} 

template <typename T>
bool NTreeChopper<T>::operator()( int i, int j)  
{
    return bb[i].min.z < bb[j].min.z ;    // ascending bb.min.z
}  

template <typename T>
unsigned NTreeChopper<T>::get_num_prim() const 
{
    return prim.size() ;
}



template <typename T>
std::string NTreeChopper<T>::brief() const 
{
    std::stringstream ss ; 
    ss
        << "NTreeChopper::brief"
        << " root.treeidx " << std::setw(3) << root->treeidx 
        << " num_prim " << std::setw(2) << prim.size() 
        ;
    return ss.str();
}


template <typename T>
void NTreeChopper<T>::dump(const char* msg)
{
      LOG(info) 
          << msg 
          << " treedir " << ( root->treedir ? root->treedir : "-" )
          << " typmsk " << root->get_type_mask_string() 
          << " nprim " << prim.size()
          << " verbosity " << verbosity
           ; 

      dump_qty('R');
      dump_qty('Z');
      dump_qty('B');

      dump_joins(); 

}



template <typename T>
void NTreeChopper<T>::dump_qty(char qty, int wid)
{
     switch(qty)
     {
        case 'B': std::cout << "dump_qty : bbox (globally transformed) " << std::endl ; break ; 
        case 'Z': std::cout << "dump_qty : bbox.min/max.z (globally transformed) " << std::endl ; break ; 
        case 'R': std::cout << "dump_qty : model frame r1/r2 (local) " << std::endl ; break ; 
     }

     for(unsigned i=0 ; i < prim.size() ; i++)
     {
          unsigned j = zorder[i] ; 
          std::cout << std::setw(15) << prim[j]->tag() ;

          if(qty == 'Z' ) 
          {
              for(unsigned indent=0 ; indent < i ; indent++ ) std::cout << std::setw(wid*2) << " " ;  
              std::cout 
                    << std::setw(wid) << " bb.min.z " 
                    << std::setw(wid) << std::fixed << std::setprecision(3) << bb[j].min.z 
                    << std::setw(wid) << " bb.max.z " 
                    << std::setw(wid) << std::fixed << std::setprecision(3) << bb[j].max.z
                    << std::endl ; 
          } 
          else if( qty == 'R' )
          {
              for(unsigned indent=0 ; indent < i ; indent++ ) std::cout << std::setw(wid*2) << " " ;  
              std::cout 
                    << std::setw(wid) << " r1 " 
                    << std::setw(wid) << std::fixed << std::setprecision(3) << prim[j]->r1() 
                    << std::setw(wid) << " r2 " 
                    << std::setw(wid) << std::fixed << std::setprecision(3) << prim[j]->r2()
                    << std::endl ; 
          }
          else if( qty == 'B' )
          {
               std::cout << bb[j].desc() << std::endl ; 
          }
     }
}

template <typename T>
void NTreeChopper<T>::dump_joins()
{
     int wid = 10 ;
     std::cout << "dump_joins" << std::endl ; 

     for(unsigned i=1 ; i < prim.size() ; i++)
     {
         unsigned ja = zorder[i-1] ; 
         unsigned jb = zorder[i] ; 

         const T* a = prim[ja] ;
         const T* b = prim[jb] ;

         float za = bb[ja].max.z ; 
         float ra = a->r2() ; 

         float zb = bb[jb].min.z ; 
         float rb = b->r1() ; 

         NNodeJoinType join = NNodeEnum::JoinClassify( za, zb, epsilon );
         std::cout 
                 << " ja: " << std::setw(15) << prim[ja]->tag()
                 << " jb: " << std::setw(15) << prim[jb]->tag()
                 << " za: " << std::setw(wid) << std::fixed << std::setprecision(3) << za 
                 << " zb: " << std::setw(wid) << std::fixed << std::setprecision(3) << zb 
                 << " join " << std::setw(2*wid) << NNodeEnum::JoinType(join)
                 << " ra: " << std::setw(wid) << std::fixed << std::setprecision(3) << ra 
                 << " rb: " << std::setw(wid) << std::fixed << std::setprecision(3) << rb 
                 << std::endl ; 
    }
}



//#include "No.hpp"
#include "NNode.hpp"

//template struct NTreeChopper<no> ; 
template struct NTreeChopper<nnode> ; 


