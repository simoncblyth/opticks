
#include <sstream>
#include <map>

#include "SSys.hh"
#include "PLOG.hh"

#include "OpticksCSG.h"
#include "OpticksCSGMask.h"

#include "NNode.hpp"
#include "NNodeNudger.hpp"



NNodeNudger::NNodeNudger(nnode* root_, float epsilon_, unsigned /*verbosity*/) 
     :
     root(root_),
     epsilon(epsilon_), 
     verbosity(SSys::getenvint("VERBOSITY",1)),
     znudge_count(0)
{
    init();
}

void NNodeNudger::init()
{
    root->collect_prim_for_edit(prim);
    update_prim_bb();
    collect_coincidence();
    uncoincide();
}

void NNodeNudger::update_prim_bb()
{
    zorder.clear();
    bb.clear(); 
    for(unsigned i=0 ; i < prim.size() ; i++)
    {
        const nnode* p = prim[i] ; 
        nbbox pbb = p->bbox(); 
        bb.push_back(pbb);
        zorder.push_back(i);
    }
    std::sort(zorder.begin(), zorder.end(), *this );
} 

bool NNodeNudger::operator()( int i, int j)  
{
    return bb[i].min.z < bb[j].min.z ;    // ascending bb.min.z
}  



/*

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





   * know how to handle siblings of union parent
     with minmax or maxmin pair coincidence

   * difference coincidence will often be non-siblings, eg 
     (cy-cy)-co when the base of the subtracted cone lines up with 
      the first cylinder ... perhaps should +ve-ize 


            -
         -    co
       cy cy

     +ve form:

            *
         *    !co
       cy !cy


     

     Consider (cy - co) with coincident base...
     solution is to grow co down, but how to 
     detect in code ? 

     When you get minmin coincidence ~~~
     (min means low edge... so direction to grow
      is clear ? Check parents of the pair and
      operate on one with the "difference" parent, 
      ie the one being subtracted) 

     Nope they could both be being subtracted ?


     A minmin coincidence after positivization, 
     can always pull down the one with the complement ?



                        +-----+
                       /       \
                      /         \
             +-------*-------+   \
             |      /        |    \
             |     /         |     \
             |    /          |      \
             |   /           |       \
             |  /            |        \
             | /             |         \
             |/              |          \
             *               |           \
            /|               |            \
           / |               |        B    \
          /  |  A            |              \
         /   |               |               \
        /    |               |                \
       +-----+~~~~~~~~~~~~~~~+-----------------+


*/





void NNodeNudger::collect_coincidence()
{
    if(root->treeidx < 0) return ; 

    coincidence.clear();

    unsigned num_prim = prim.size() ;

    for(unsigned i=0 ; i < num_prim ; i++){
    for(unsigned j=0 ; j < num_prim ; j++)
    {
        if( i < j ) collect_coincidence(i, j);  // all pairs once
    }
    }

    unsigned num_coincidence = coincidence.size() ;

    LOG(trace) << "NNodeNudger::collect_coincidence" 
              << " root.treeidx " << root->treeidx 
              << " num_prim " << num_prim 
              << " num_coincidence " << num_coincidence
              << " verbosity " << verbosity 
              ; 
}

void NNodeNudger::collect_coincidence(unsigned i, unsigned j)
{
    /*
     General collection of prim-prim coincidence is not useful
     for issue detection, because there are so many such coincidences
     that cause no problem.
   
     Nevertheless it may prove useful for classification of
     issues and automated decision making regards fixes. i.e.
     deciding which primitive to nudge and in which direction, 
     so as to avoid the issue and not change geometry.
    */

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




bool NNodeCoincidence::is_siblings() const
{
   return i->parent && j->parent && i->parent == j->parent ;
}

bool NNodeCoincidence::is_union_siblings() const
{
   return is_siblings() && i->parent->type == CSG_UNION ;
}

bool NNodeCoincidence::is_union_parents() const
{
   return i->parent && j->parent && i->parent->type == CSG_UNION && j->parent->type == CSG_UNION ;
}




std::string NNodeCoincidence::desc() const 
{
    std::stringstream ss ; 
    ss
        << "(" << std::setw(2) << i->idx
        << "," << std::setw(2) << j->idx
        << ")"
        << " " << NNodeEnum::PairType(p)
        << " " << i->tag()
        << " " << j->tag()
        << " sibs " << ( is_siblings() ? "Y" : "N" )
        << " u_sibs " << ( is_union_siblings() ? "Y" : "N" )
        << " u_par " << ( is_union_parents() ? "Y" : "N" )
        << " u_same " << ( nnode::is_same_union(i,j) ? "Y" : "N" )
        << " " << ( fixed ? "FIXED" : "" )
        ; 
  
    return ss.str();
}



unsigned NNodeNudger::get_num_coincidence() const 
{
   return coincidence.size();
}

std::string NNodeNudger::desc_coincidence() const 
{
    unsigned num_prim = prim.size() ;
    unsigned num_coincidence = coincidence.size() ;

    std::map<NNodePairType, unsigned> pair_counts ; 
    for(unsigned i=0 ; i < num_coincidence ; i++) pair_counts[coincidence[i].p]++ ; 

    assert( pair_counts[PAIR_MINMAX] == 0);


    std::stringstream ss ; 
    ss
        << " verbosity " << verbosity 
        << " root.treeidx " << std::setw(3) << root->treeidx 
        << " num_prim " << std::setw(2) << num_prim 
        << " num_coincidence " << std::setw(2) << num_coincidence
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








void NNodeNudger::uncoincide()
{
   unsigned num_coincidence = coincidence.size();
   for(unsigned i=0 ; i < num_coincidence ; i++)
   {
       znudge(&coincidence[i]);
   }


}


bool NNodeNudger::can_znudge(const NNodeCoincidence* coin) const 
{
    bool can = false ; 

    can = can_znudge_umaxmin(coin);
    if(can) return true ; 

    can = can_znudge_dminmin(coin);
    if(can) return true ; 

    return can ; 
}

void NNodeNudger::znudge(NNodeCoincidence* coin)
{
    if(!coin->fixed && can_znudge_umaxmin(coin)) znudge_umaxmin(coin);
    if(!coin->fixed && can_znudge_dminmin(coin)) znudge_dminmin(coin);
}


bool NNodeNudger::can_znudge_umaxmin(const NNodeCoincidence* coin) const 
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

bool NNodeNudger::can_znudge_dminmin(const NNodeCoincidence* coin) const
{
    const nnode* i = coin->i ; 
    const nnode* j = coin->j ; 
    const NNodePairType p = coin->p ; 

    return i->parent && j->parent && 
           ( i->parent->type == CSG_DIFFERENCE ||  j->parent->type == CSG_DIFFERENCE ) 
           && p == PAIR_MINMIN && i->is_znudge_capable() && j->is_znudge_capable() ;
} 



void NNodeNudger::znudge_dminmin(NNodeCoincidence* coin)
{
    std::cout << "NNodeNudger::znudge_dminmin"
              << " coin " << coin->desc()
              << std::endl ; 

}


void NNodeNudger::znudge_umaxmin(NNodeCoincidence* coin)
{
    assert(can_znudge_umaxmin(coin));
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
    }
    else
    {
        i->increase_z2( dz ); 
    }

    nbbox ibb2 = i->bbox();
    nbbox jbb2 = j->bbox();

    float zi2 = ibb2.max.z ; 
    float zj2 = jbb2.min.z ;
 
    NNodeJoinType join2 = NNodeEnum::JoinClassify( zi2, zj2, epsilon );
    assert(join2 != JOIN_COINCIDENT);

    coin->fixed = true ; 
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



void NNodeNudger::dump(const char* msg)
{
      LOG(info) 
          << msg 
          << " treedir " << ( root->treedir ? root->treedir : "-" )
          << " typmsk " << root->get_type_mask_string() 
          << " nprim " << prim.size()
          << " znudge_count " << znudge_count
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

