#include "PLOG.hh"


#include "NOpenMeshBoundary.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMesh.hpp"
#include "NOpenMeshZipper.hpp"

#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"



template <typename T>
NOpenMeshZipper<T>::NOpenMeshZipper(
          const NOpenMesh<T>& lhs, 
          const NOpenMesh<T>& rhs
         )
   :
   lhs(lhs), 
   rhs(rhs)
{
    init();
}

template <typename T>
void NOpenMeshZipper<T>::init()
{
    LOG(info) << "NOpenMeshZipper::init"
              << " lhs " << lhs.brief()
              << " rhs " << rhs.brief()
              ;

    dump();
}



template <typename T>
void NOpenMeshZipper<T>::dump()
{
    unsigned n_lhs_inner = lhs.find.inner_loops.size() ;
    unsigned n_rhs_inner = rhs.find.inner_loops.size() ;
    unsigned n_lhs_outer = lhs.find.outer_loops.size() ;
    unsigned n_rhs_outer = rhs.find.outer_loops.size() ;

    std::cout 
         << " n_lhs_inner " << n_lhs_inner
         << " n_rhs_inner " << n_rhs_inner
         << " n_lhs_outer " << n_lhs_outer
         << " n_rhs_outer " << n_rhs_outer
         << std::endl 
         ; 

    for(unsigned i=0 ; i < n_lhs_inner ; i++)  
        dump_boundary( i, lhs.find.inner_loops[i], "lhs_inner" );

    for(unsigned i=0 ; i < n_lhs_outer ; i++)
        dump_boundary( i, lhs.find.outer_loops[i], "lhs_outer" );

    for(unsigned i=0 ; i < n_rhs_inner ; i++)  
        dump_boundary( i, rhs.find.inner_loops[i], "rhs_inner" );

    for(unsigned i=0 ; i < n_rhs_outer ; i++)  
        dump_boundary( i, rhs.find.outer_loops[i], "rhs_outer" );
}



template <typename T>
void NOpenMeshZipper<T>::dump_boundary(int index, const NOpenMeshBoundary<T>& loop, const char* msg)
{
    LOG(info) 
           << msg << " " 
           << std::setw(5) << index  
           << loop.desc()
            ; 

    dump_frontier(loop) ; 
}


template <typename T>
void NOpenMeshZipper<T>::dump_frontier(const NOpenMeshBoundary<T>& loop)
{
/*
In general the intersection between sub-objects frontier will not be a plane, 
nevertherless finding the best fit plane provides a way to simplify 
handling : can order loop edges and frontier points 
according to an angle after picking some basis vector that lies in the plane.

Hmm project spoke vectors from the cog onto the plane...
 


* http://www.ilikebigbits.com/blog/2015/3/2/plane-from-points

    Edit: as the commenter Paul pointed out, this method will minimize the squares
    of the residuals as perpendicular to the main axis, not the residuals
    perpendicular to the plane. If the residuals are small (i.e. your points all
    lie close to the resulting plane), then this method will probably suffice.
    However, if your points are more spread then this method may not be the best
    fit.

*/

    float xx(0);
    float xy(0);
    float xz(0);

    float yy(0);
    float yz(0);
    float zz(0);

    for(unsigned i=0 ; i < loop.frontier.size() ; i++)
    {
         P p = loop.frontier[i] ; 
         P d = p - loop.frontier_cog ; 

         xx += d[0] * d[0];
         xy += d[0] * d[1];
         xz += d[0] * d[2];
         yy += d[1] * d[1];
         yz += d[1] * d[2];
         zz += d[2] * d[2];


         std::cout 
             << std::setw(5) << i  
             << " p " << NOpenMeshDesc<T>::desc_point(p,8,2) 
             << " d " << NOpenMeshDesc<T>::desc_point(d,8,2) 
             << " dlen " << d.length()
             << std::endl 
             ;
    }

    float det_x = yy*zz - yz*yz ;
    float det_y = xx*zz - xz*xz ;
    float det_z = xx*yy - xy*xy ;

    float det_max = std::max( std::max(det_x, det_y), det_z ) ; 
    assert( det_max > 0);


   

    P dir ; 

    float a, b ; 

    if(det_max == det_x)
    {
        a = (xz*yz - xy*zz) / det_x;
        b = (xy*yz - xz*yy) / det_x;
        dir[0] = 1. ; 
        dir[1] = a ; 
        dir[2] = b ; 
    }
    else if (det_max == det_y) 
    {
        a = (yz*xz - xy*zz) / det_y;
        b = (xy*xz - yz*xx) / det_y;

        dir[0] = a ; 
        dir[1] = 1 ; 
        dir[2] = b ; 
    } 
    else
    {
        a = (yz*xy - xz*yy) / det_z;
        b = (xz*xy - yz*xx) / det_z;
        dir[0] = a ; 
        dir[1] = b ; 
        dir[2] = 1 ; 
    }

     std::cout 
             << " dir " << NOpenMeshDesc<T>::desc_point(dir,8,2) << std::endl ; 
 

}
 



template struct NOpenMeshZipper<NOpenMeshType> ;

