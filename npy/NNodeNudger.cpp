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
#include "OpticksCSGMask.h"

#include "NPY.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NNode.hpp"
#include "NNodeNudger.hpp"
#include "NNodeCoincidence.hpp"


const plog::Severity NNodeNudger::LEVEL = PLOG::EnvLevel("NNodeNudger", "DEBUG"); 


std::vector<unsigned>* NNodeNudger::TreeList = NULL ;  


NPY<unsigned>* NNodeNudger::NudgeBuffer = NULL ;  

void NNodeNudger::SaveBuffer(const char* path)  // static
{
    NudgeBuffer->save(path); 
}
void NNodeNudger::SaveBuffer(const char* dir, const char* name)  // static
{
    NudgeBuffer->save(dir, name); 
}



NNodeNudger::NNodeNudger(nnode* root_, float epsilon_, unsigned /*verbosity*/) 
     :
     root(root_),
     epsilon(epsilon_), 
     verbosity(SSys::getenvint("VERBOSITY",1)),
     listed(false),
     enabled(true)
{
    root->check_tree( FEATURE_GTRANSFORMS | FEATURE_PARENT_LINKS );
    init();
}

void NNodeNudger::init()
{
    LOG(LEVEL) << " init " ; 

    //if( TreeList == NULL )
    //     TreeList = new std::vector<unsigned> {24, 42, 37, 22};   // iav, oav, lso, gds

    listed = TreeList != NULL && std::find(TreeList->begin(), TreeList->end(), root->treeidx ) != TreeList->end() ; 
 
    if( NudgeBuffer == NULL ) NudgeBuffer = NPY<unsigned>::make(0,4) ; 

    root->collect_prim_for_edit(prim);  // recursive collector 
    update_prim_bb();                   // find z-order of prim using bb.min.z
    collect_coincidence();

    if(enabled)
       uncoincide();

    if( listed || nudges.size() > 0 ) LOG(error) << brief() ;  

    if(NudgeBuffer) NudgeBuffer->add(root->treeidx, prim.size(), coincidence.size(), nudges.size() );   

    LOG(LEVEL) << brief() ; 

}

void NNodeNudger::update_prim_bb()
{
    //LOG(info) << "NNodeNudger::update_prim_bb nprim " << prim.size() ;
    zorder.clear();
    bb.clear(); 
    for(unsigned i=0 ; i < prim.size() ; i++)
    {
        const nnode* p = prim[i] ; 

        nbbox pbb = p->bbox(); 
        bb.push_back(pbb);
        zorder.push_back(i);

        /*
        const nmat4triple* gtransform = p->gtransform ; 
        assert( gtransform ) ; 
        const glm::mat4& t  = gtransform->t ;                
        std::cout << " prim " << i << std::endl 
                  << gpresent("t", t )  
                  << " pbb " << pbb.desc()
                  << std::endl ; 
        */ 

    }
    std::sort(zorder.begin(), zorder.end(), *this );   // np.argsort style : sort the indices
} 

bool NNodeNudger::operator()( int i, int j)  
{
    return bb[i].min.z < bb[j].min.z ;    // ascending bb.min.z
}  

unsigned NNodeNudger::get_num_prim() const 
{
    return prim.size() ;
}


/**
NNodeNudger::collect_coincidence
----------------------------------

1. collects pairs of primitive indices with coincident 
   bbox min.z or max.z recording the type of 
   pairing PAIR_MINMIN PAIR_MINMAX etc..

General collection of prim-prim coincidence is not useful
for issue detection, because there are so many such coincidences
that cause no problem.

Nevertheless it may prove useful for classification of
issues and automated decision making regards fixes. i.e.
deciding which primitive to nudge and in which direction, 
so as to avoid the issue and not change geometry.


**/



void NNodeNudger::collect_coincidence()
{
    LOG(LEVEL) << " collect_concidence " << root->treeidx ;  

    if(root->treeidx < 0) return ; 

    coincidence.clear();

    unsigned num_prim = get_num_prim() ;

    for(unsigned i=0 ; i < num_prim ; i++){
    for(unsigned j=0 ; j < num_prim ; j++)
    {
        if( i < j ) collect_coincidence(i, j);  // all pairs once
    }
    }

    LOG(debug) << "NNodeNudger::collect_coincidence" 
              << " root.treeidx " << root->treeidx 
              << " num_prim " << num_prim 
              << " num_coincidence " << get_num_coincidence()
              << " verbosity " << verbosity 
              ; 
}



/**
NNodeNudger::collect_coincidence(unsigned i, unsigned j)
---------------------------------------------------------

1. classify joins for all 4 pairings eg PAIR_MINMIN, PAIR_MINMAX, ...
2. collect JOIN_COINCIDENT pairs recording the pairing type 


::

              maxmax      +--+
               /          |  |
       +-----+--+---------+--+-----+
       |     |  |           \      |
       |     +--+          minmax  |
       |                           |
       |                           |
       |                           |
       |                           |
       |                           |
       |                           |
       |                           |
       |                           |
       |                           |
       |                           |
       |  maxmin   +--+            |
       |    \      |  |            |
       +---+--+----+--+------------+
           |  |      \
           +--+       minmin
           

      Label order just picks one of the
      pair as first, eg smaller box in above
      comparisons. 

**/

void NNodeNudger::collect_coincidence(unsigned i, unsigned j)
{
    LOG(LEVEL) << " collect_concidence " 
               << " treeidx : " << root->treeidx 
               << " prim pair (i,j) : (" << i << "," << j << ")" 
               ;
  
    for(unsigned p=0 ; p < 4 ; p++)
    {
        NNodePairType pair = (NNodePairType)p ; 

        float zi, zj ; 
        switch(pair)
        {
            case PAIR_MINMIN: { zi = bb[i].min.z ; zj = bb[j].min.z ; } ; break ;  
            case PAIR_MINMAX: { zi = bb[i].min.z ; zj = bb[j].max.z ; } ; break ;  
            case PAIR_MAXMIN: { zi = bb[i].max.z ; zj = bb[j].min.z ; } ; break ;  
            case PAIR_MAXMAX: { zi = bb[i].max.z ; zj = bb[j].max.z ; } ; break ;  
        }


        NNodeJoinType join = NNodeEnum::JoinClassify( zi, zj, epsilon );

        LOG(LEVEL) 
               << " pair: " << NNodeEnum::PairType(pair) 
               << " zi: " << std::fixed << std::setw(10) << std::setprecision(3) << zi
               << " zj: " << std::fixed << std::setw(10) << std::setprecision(3) << zj
               << " join: " << NNodeEnum::JoinType(join) 
               ;

        if(join == JOIN_COINCIDENT) 
        {
            switch(pair)
            {  
                case PAIR_MINMIN:  coincidence.push_back({ prim[i], prim[j], pair }); break ;
                case PAIR_MINMAX:  coincidence.push_back({ prim[j], prim[i], PAIR_MAXMIN }); break ;  // flip prim order of MINMAX to make a MAXMIN
                case PAIR_MAXMIN:  coincidence.push_back({ prim[i], prim[j], pair }); break ;
                case PAIR_MAXMAX:  coincidence.push_back({ prim[i], prim[j], pair }); break ;
            }
        } 
    }
}

unsigned NNodeNudger::get_num_coincidence() const 
{
   return coincidence.size();
}

void NNodeNudger::uncoincide()
{
   unsigned num_coincidence = coincidence.size();
   for(unsigned i=0 ; i < num_coincidence ; i++)
   {
       znudge(&coincidence[i]);
   }
}


void NNodeNudger::znudge(NNodeCoincidence* coin)
{
    // IS THIS MISSING A umaxmax ???

    if(!coin->fixed && can_znudge_union_maxmin(coin))      znudge_union_maxmin(coin);
    if(!coin->fixed && can_znudge_difference_minmin(coin)) znudge_difference_minmin(coin);
}

bool NNodeNudger::can_znudge_union_maxmin(const NNodeCoincidence* coin) const 
{
    const nnode* i = coin->i ; 
    const nnode* j = coin->j ; 
    const NNodePairType p = coin->p ; 
    return nnode::is_same_union(i,j) && p == PAIR_MAXMIN && i->is_znudge_capable() && j->is_znudge_capable() ;

    // requiring siblings is too restrictive... the binary splitup is an implemntation detail
    // what matters is that they are from the same union not the same pair 
    //
    // for z-sphere the ability to znudge depends on endcap existance on a side ... ? also radius contraints
    // due to this have removed from znudge capable
}

bool NNodeNudger::can_znudge_difference_minmin(const NNodeCoincidence* coin) const
{
    const nnode* i = coin->i ; 
    const nnode* j = coin->j ; 
    const NNodePairType p = coin->p ; 

    return i->parent && j->parent && 
           ( i->parent->type == CSG_DIFFERENCE ||  j->parent->type == CSG_DIFFERENCE ) 
           && p == PAIR_MINMIN && i->is_znudge_capable() && j->is_znudge_capable() ;
} 



void NNodeNudger::znudge_difference_minmin(NNodeCoincidence* coin)
{
    LOG(LEVEL) << "NNodeNudger::znudge_difference_minmin"
              << " coin " << coin->desc()
              << std::endl ; 
    LOG(LEVEL) << " NOT IMPLEMENTED " ;

    //std::raise(SIGINT);  
}


void NNodeNudger::znudge_union_maxmin(NNodeCoincidence* coin)
{
    assert(can_znudge_union_maxmin(coin));
    assert(coin->fixed == false);

    nnode* i = coin->i ; 
    nnode* j = coin->j ; 
    const NNodePairType p = coin->p ; 

    nbbox ibb = i->bbox();
    nbbox jbb = j->bbox();

    float dz(1.);

    assert( p == PAIR_MAXMIN );

    float zi = ibb.max.z ; 
    float zj = jbb.min.z ;
    float ri = i->r2() ; 
    float rj = j->r1() ; 

    NNodeJoinType join = NNodeEnum::JoinClassify( zi, zj, epsilon );
    assert(join == JOIN_COINCIDENT);

    if( ri > rj )  
    {
        j->decrease_z1( dz );   
        coin->n = NUDGE_J_DECREASE_Z1 ; 
    }
    else
    {
        i->increase_z2( dz ); 
        coin->n = NUDGE_I_INCREASE_Z2 ; 
    }

    nbbox ibb2 = i->bbox();
    nbbox jbb2 = j->bbox();

    float zi2 = ibb2.max.z ; 
    float zj2 = jbb2.min.z ;
 
    NNodeJoinType join2 = NNodeEnum::JoinClassify( zi2, zj2, epsilon );
    assert(join2 != JOIN_COINCIDENT);

    coin->fixed = true ; 

    nudges.push_back(*coin); 

}


bool NNodeNudger::can_znudge(const NNodeCoincidence* coin) const 
{
    bool can = false ; 

    can = can_znudge_union_maxmin(coin);
    if(can) return true ; 

    can = can_znudge_difference_minmin(coin);
    if(can) return true ; 

    return can ; 
}



/*

    +--------------+ .
    |              |
    |           . ++-------------+
    |             ||             |
    |         rb  ||  ra         |
    |             ||             | 
    |           . || .           |    
    |             ||             |
    |             ||          b  |
    |           . ++-------------+
    |  a           |
    |              |
    +--------------+ .

                  za  
                  zb                      

    ------> Z

*/





std::string NNodeNudger::brief() const 
{
    std::stringstream ss ; 
    ss
        << "NNodeNudger::brief"
        << " root.treeidx " << std::setw(3) << root->treeidx 
        << " num_prim " << std::setw(2) << prim.size() 
        << " num_coincidence " << std::setw(2) << coincidence.size()
        << " num_nudge " << std::setw(2) << nudges.size()
        << " " << ( listed ? "##LISTED" : "" )
        ;
    return ss.str();
}


std::string NNodeNudger::desc_coincidence() const 
{
    unsigned num_prim = prim.size() ;
    unsigned num_coincidence = coincidence.size() ;
    unsigned num_nudge = nudges.size() ;

    std::map<NNodePairType, unsigned> pair_counts ; 
    for(unsigned i=0 ; i < num_coincidence ; i++) pair_counts[coincidence[i].p]++ ; 

    assert( pair_counts[PAIR_MINMAX] == 0);


    std::stringstream ss ; 
    ss
        << " verbosity " << verbosity 
        << " root.treeidx " << std::setw(3) << root->treeidx 
        << " num_prim " << std::setw(2) << num_prim 
        << " num_coincidence " << std::setw(2) << num_coincidence
        << " num_nudge " << std::setw(2) << num_nudge
        << " MINMIN " << std::setw(2) << pair_counts[PAIR_MINMIN]
        << " MAXMIN " << std::setw(2) << pair_counts[PAIR_MAXMIN]
        << " MAXMAX " << std::setw(2) << pair_counts[PAIR_MAXMAX]
        ;

    if(verbosity > 2)
    {
        ss << std::endl ; 
        for(unsigned i=0 ; i < num_coincidence ; i++) ss << coincidence[i].desc() << std::endl ; 
    }
  
    return ss.str();
}


void NNodeNudger::dump(const char* msg)
{
      LOG(info) 
          << msg 
          << " treedir " << ( root->treedir ? root->treedir : "-" )
          << " typmsk " << root->get_type_mask_string() 
          << " nprim " << prim.size()
          << " nudges " << nudges.size()
          << " verbosity " << verbosity
           ; 

      dump_qty('R');
      dump_qty('Z');
      dump_qty('B');
      dump_joins();
}



void NNodeNudger::dump_qty(char qty, int wid)
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

void NNodeNudger::dump_joins()
{
     int wid = 10 ;
     std::cout << "dump_joins" << std::endl ; 

     for(unsigned i=1 ; i < prim.size() ; i++)
     {
         unsigned ja = zorder[i-1] ; 
         unsigned jb = zorder[i] ; 

         const nnode* a = prim[ja] ;
         const nnode* b = prim[jb] ;

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

