
#include "PLOG.hh"

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
     int wid = 10 ;
     float dz = 1.0f ; // perhaps should depend on z-range of prims ?   

     if(verbosity > 0)
     LOG(info) << " znudge over prim pairs " 
               << " dz " << dz
                ; 

     for(unsigned i=1 ; i < prim.size() ; i++)
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


