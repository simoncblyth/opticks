
#include "PLOG.hh"

#include "OpticksCSG.h"
#include "NNode.hpp"
#include "NNodeNudger.hpp"



NNodeNudger::NNodeNudger(nnode* root, float epsilon, unsigned verbosity) 
     :
     root(root),
     epsilon(epsilon), 
     verbosity(verbosity),
     znudge_count(0)
{
    init();
}

void NNodeNudger::init()
{
     root->collect_prim_for_edit(prim);
     update_bb();
}

void NNodeNudger::update_bb()
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


void NNodeNudger::znudge()
{
    znudge_lineup();
    //znudge_anypair();
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


void NNodeNudger::znudge_anypair(unsigned i, unsigned j)
{
   // VERBOSITY=3 NScanTest 85 

    LOG(info) << "NNodeNudger::znudge_anypair" 
              << " i " << i
              << " j " << j
              ; 

    int wid = 10 ; 
    bool are_sibling = prim[i]->parent == prim[j]->parent ;  
    OpticksCSG_t i_parent_type = prim[i]->parent->type ;
    OpticksCSG_t j_parent_type = prim[j]->parent->type ;

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
        
        if(verbosity > 2)
        {
             std::cout 
             << " i: " << std::setw(15) << prim[i]->tag()
             << " j: " << std::setw(15) << prim[j]->tag()
             << " are_sibling " << ( are_sibling ? "Y" : "N" )
             << " i_parent " << CSGName(i_parent_type) 
             << " j_parent " << CSGName(j_parent_type) 
             << " pair " << NNodeEnum::PairType(pair)
             << " zi " << std::setw(wid) << std::fixed << std::setprecision(3) << zi
             << " zj " << std::setw(wid) << std::fixed << std::setprecision(3) << zj
             << " join " << NNodeEnum::JoinType(join)
             << std::endl ; 
        }
    }
}


void NNodeNudger::znudge_anypair()
{
    unsigned num_prim = prim.size() ;
    LOG(info) << "NNodeNudger::znudge_anypair" 
              << " num_prim " << num_prim 
              << " verbosity " << verbosity 
              ; 
    for(unsigned i=0 ; i < num_prim ; i++){
    for(unsigned j=0 ; j < num_prim ; j++)
    {
        if( i < j ) znudge_anypair(i, j);  // all pairs once
    }
    }
}
   


void NNodeNudger::znudge_lineup()
{
     int wid = 10 ;
     float dz = 1.0f ; // perhaps should depend on z-range of prims ?   
     unsigned num_prim = prim.size() ;

     if(verbosity > 0)
     LOG(info) << " znudge over prim pairs " 
               << " verbosity " << verbosity 
               << " num_prim " << num_prim
               << " dz " << dz
                ; 

     for(unsigned i=1 ; i < num_prim ; i++)
     {
          unsigned ja = zorder[i-1] ; 
          unsigned jb = zorder[i] ; 

          nnode* a = prim[ja] ;
          nnode* b = prim[jb] ;

          float za = bb[ja].max.z ; 
          float ra = a->r2() ; 

          float zb = bb[jb].min.z ; 
          float rb = b->r1() ; 

          NNodeJoinType join = NNodeEnum::JoinClassify( za, zb, epsilon );

          if(verbosity > 2)
          std::cout 
                 << " ja: " << std::setw(15) << prim[ja]->tag()
                 << " jb: " << std::setw(15) << prim[jb]->tag()
                 << " za: " << std::setw(wid) << std::fixed << std::setprecision(3) << za 
                 << " zb: " << std::setw(wid) << std::fixed << std::setprecision(3) << zb 
                 << " join " << std::setw(2*wid) << NNodeEnum::JoinType(join)
                 << " ra: " << std::setw(wid) << std::fixed << std::setprecision(3) << ra 
                 << " rb: " << std::setw(wid) << std::fixed << std::setprecision(3) << rb 
                 << std::endl ; 

          if( join == JOIN_COINCIDENT )
          {
              // TODO: fix unjustified assumption that transforms dont swap the radii orderiing 
              // expand side with smaller radii into the other to make the join OVERLAP

              if( ra > rb )  
              {
                  b->decrease_z1( dz );   
              }
              else
              {
                  a->increase_z2( dz ); 
              }
              znudge_count++ ; 
          }  
     } 
     update_bb();
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


