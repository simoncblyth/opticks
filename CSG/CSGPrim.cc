
#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include "scuda.h"
#include "sqat4.h"

#include "SMesh.h"
#include "SMeshGroup.h"
#include "ssys.h"

#include "CSGPrim.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <cstring>
#include <algorithm>


std::string CSGPrim::desc() const
{
    std::stringstream ss ;
    ss
      << "CSGPrim"
      << " numNode/node/tran/plan"
      << std::setw(4) << numNode() << " "
      << std::setw(4) << nodeOffset() << " "
      << std::setw(4) << tranOffset() << " "
      << std::setw(4) << planOffset() << " "
      << "sbtOffset/meshIdx/repeatIdx/primIdx "
      << std::setw(4) << sbtIndexOffset() << " "
      << std::setw(4) << meshIdx() << " "
      << std::setw(4) << repeatIdx() << " "
      << std::setw(4) << primIdx()
      << " mn " << mn()
      << " mx " << mx()
      ;
    std::string str = ss.str();
    return str ;
}

std::string CSGPrim::descRange() const
{
    float3 mn3 = mn();
    float3 mx3 = mx();
    float4 ce4 = ce();

    std::array<float,4> _ce = {ce4.x, ce4.y, ce4.z,ce4.w};
    std::array<float,6> _bb = {mn3.x, mn3.y, mn3.z, mx3.x, mx3.y, mx3.z};

    std::stringstream ss ;
    ss << "CSGPrim::descRange" ;
    ss << " ce " << s_bb::Desc_<float,4>( _ce.data() ) ;
    ss << " bb " << s_bb::Desc_<float,6>( _bb.data()) ;
    ss << " lvid " << std::setw(3) << meshIdx() ;
    ss << " ridx " << std::setw(4) << repeatIdx() ;
    ss << " pidx " << std::setw(5) << primIdx() ;
    std::string str = ss.str();
    return str ;
}


std::string CSGPrim::descRangeNumPy() const
{
    float3 mn3 = mn();
    float3 mx3 = mx();
    float4 ce4 = ce();

    std::array<float,4> _ce = {ce4.x, ce4.y, ce4.z,ce4.w};
    std::array<float,6> _bb = {mn3.x, mn3.y, mn3.z, mx3.x, mx3.y, mx3.z};

    std::stringstream ss ;
    ss << s_bb::DescNumPy_<float,4>( _ce.data(), "ce", false ) ;
    ss << s_bb::DescNumPy_<float,6>( _bb.data(), "bb", false ) ;
    ss << " # CSGPrim::descRangeNumPy " ;
    ss << " lvid " << std::setw(3) << meshIdx() ;
    ss << " ridx " << std::setw(4) << repeatIdx() ;
    ss << " pidx " << std::setw(5) << primIdx() ;
    std::string str = ss.str();
    return str ;
}




/**
CSGPrim::DescRange
-------------------

Displays coordinate ranges and extent of the analytic CSGPrim and triangulated SMesh.

**/


std::string CSGPrim::DescRange(const CSGPrim* prim, int primOffset, int numPrim, const std::vector<std::string>* soname, const SMeshGroup* mg ) // static
{
    int NUMPY = ssys::getenvint(CSGPrim__DescRange_NUMPY,0);
    bool NAMEONLY = ssys::getenvint(CSGPrim__DescRange_NAMEONLY,0) > 0;

    size_t num_so = soname ? soname->size() : 0 ;
    int mg_subs = mg ? mg->subs.size() : 0 ;
    float EXTENT_DIFF = ssys::getenvfloat(CSGPrim__DescRange_EXTENT_DIFF, 200.f );
    float EXTENT_MINI = ssys::getenvfloat(CSGPrim__DescRange_EXTENT_MINI,   0.f );

    std::vector<float>* CE_ZMIN_ZMAX = ssys::getenvfloatvec(CSGPrim__DescRange_CE_ZMIN_ZMAX, nullptr ); // nullptr when no envvar
    float CE_ZMIN = CE_ZMIN_ZMAX ? (*CE_ZMIN_ZMAX)[0] : 0.f ;
    float CE_ZMAX = CE_ZMIN_ZMAX ? (*CE_ZMIN_ZMAX)[1] : 0.f ;

    std::vector<float>* CE_YMIN_YMAX = ssys::getenvfloatvec(CSGPrim__DescRange_CE_YMIN_YMAX, nullptr ); // nullptr when no envvar
    float CE_YMIN = CE_YMIN_YMAX ? (*CE_YMIN_YMAX)[0] : 0.f ;
    float CE_YMAX = CE_YMIN_YMAX ? (*CE_YMIN_YMAX)[1] : 0.f ;

    std::vector<float>* CE_XMIN_XMAX = ssys::getenvfloatvec(CSGPrim__DescRange_CE_XMIN_XMAX, nullptr ); // nullptr when no envvar
    float CE_XMIN = CE_XMIN_XMAX ? (*CE_XMIN_XMAX)[0] : 0.f ;
    float CE_XMAX = CE_XMIN_XMAX ? (*CE_XMIN_XMAX)[1] : 0.f ;




    std::stringstream tt ;
    tt
       << " numPrim " << numPrim
       << " mg_subs " << mg_subs
       << "\n"
       << " EXTENT_MINI : " << std::setw(10) << std::fixed << std::setprecision(3) << EXTENT_MINI
       << " [" << CSGPrim__DescRange_EXTENT_MINI << "]"
       << "\n"
       << " EXTENT_DIFF : " << std::setw(10) << std::fixed << std::setprecision(3) << EXTENT_DIFF
       << " [" << CSGPrim__DescRange_EXTENT_DIFF << "]"
       << "\n"
       << " [" << CSGPrim__DescRange_CE_ZMIN_ZMAX << "]"
       << " CE_ZMIN : " << std::setw(10) << std::fixed << std::setprecision(3) << CE_ZMIN
       << " CE_ZMAX : " << std::setw(10) << std::fixed << std::setprecision(3) << CE_ZMAX
       << "\n"
       << " [" << CSGPrim__DescRange_CE_YMIN_YMAX << "]"
       << " CE_YMIN : " << std::setw(10) << std::fixed << std::setprecision(3) << CE_YMIN
       << " CE_YMAX : " << std::setw(10) << std::fixed << std::setprecision(3) << CE_YMAX
       << "\n"
       << " [" << CSGPrim__DescRange_CE_XMIN_XMAX << "]"
       << " CE_XMIN : " << std::setw(10) << std::fixed << std::setprecision(3) << CE_XMIN
       << " CE_XMAX : " << std::setw(10) << std::fixed << std::setprecision(3) << CE_XMAX
       << "\n"
       ;

    std::string ctx = tt.str();

    std::stringstream ss ;
    if(!NAMEONLY) ss << "[CSGPrim::Desc"
       << ctx
       ;


    // TODO: make the order controllable by envvar

    // order the prim SMesh by maximum z
    std::vector<unsigned> idx(numPrim);

    std::iota(idx.begin(), idx.end(), 0u);  // fill 0,1,2,...,numPrim-1
    //using order_functor = SMesh::zmax_functor ;
    //using order_functor = SMesh::extent_functor ;

    //using order_functor = CSGPrim::zmax_functor ;
    using order_functor = CSGPrim::extent_functor ;
    order_functor order_fn {} ;

    std::stable_sort(idx.begin(), idx.end(),
        [&](unsigned a, unsigned b) -> bool {
            return order_fn(prim[primOffset+a]) > order_fn(prim[primOffset+b]);
        });

    int count = 0 ;

    for(int j=0 ; j < numPrim ; j++ )
    {
        unsigned i = idx[j] ;
        const SMesh* sub = mg ? mg->subs[i] : nullptr ;
        const CSGPrim& pr = prim[primOffset + i];
        unsigned lvid = pr.meshIdx();

        if(sub)
        {
            assert( sub->lvid == int(lvid) );
        }

        float4 pr_ce = pr.ce();

        bool extent_in_range = EXTENT_MINI == 0.f ? true : pr_ce.w >= EXTENT_MINI ;
        bool cez_in_range =  CE_ZMIN_ZMAX ? ( pr_ce.z > CE_ZMIN && pr_ce.z < CE_ZMAX ) : true ;
        bool cey_in_range =  CE_YMIN_YMAX ? ( pr_ce.y > CE_YMIN && pr_ce.y < CE_YMAX ) : true ;
        bool cex_in_range =  CE_XMIN_XMAX ? ( pr_ce.x > CE_XMIN && pr_ce.x < CE_XMAX ) : true ;

        if(!extent_in_range) continue ;
        if(!cez_in_range) continue ;
        if(!cey_in_range) continue ;
        if(!cex_in_range) continue ;


        count += 1 ;

        const glm::tvec4<float>* sub_ce = sub ? &(sub->ce) : nullptr ;
        const glm::tmat4x4<double>* sub_tr0 = sub ? &(sub->tr0) : nullptr ;
        const char* sub_loaddir = sub ? sub->loaddir : nullptr ;

        float extent_diff = sub_ce ? (*sub_ce).w - pr_ce.w : 0. ;

        const char* so = lvid < num_so ? (*soname)[lvid].c_str() : nullptr ;

        if(NAMEONLY)
        {
            ss << ( so ? so : "-" ) << "\n" ;
        }
        else
        {

            ss
                << std::setw(4) << i
                << " : "
                << ( NUMPY ? pr.descRangeNumPy() : pr.descRange() )
                << " so[" << ( so ? so : "-" ) << "]"
                << "\n"
                ;

        }

        if(sub) ss
            << std::setw(4) << i
            << " : "
            << ( NUMPY ? sub->descRangeNumPy() : sub->descRange() )
            << " so[" << ( so ? so : "-" ) << "]"
            << " extent_diff " << std::setw(10) << std::fixed << std::setprecision(3) << extent_diff
            << "\n"
            ;

        if( std::abs(extent_diff) > EXTENT_DIFF )
        {
            ss
               << " extent_diff > EXTENT_DIFF : "
               << " extent_diff : " << std::setw(10) << std::fixed << std::setprecision(3) << extent_diff
               << "\n"
               ;
            if(sub_tr0) ss << " sub_tr0\n" << stra<double>::Desc(*sub_tr0) << "\n"  ;
            if(sub_loaddir) ss << " sub_loaddir [" << sub_loaddir << "]\n" ;
        }
    }

    if(!NAMEONLY) ss
        << ctx
        << "]CSGPrim::Desc\n"
        ;

    std::string str = ss.str() ;
    return count == 0 ? "" : str ;
}









bool CSGPrim::IsDiff( const CSGPrim& a , const CSGPrim& b )
{
    return false ;
}




/**
CSGPrim::MakeSpec
-------------------

Specification providing pointers to access all the AABB of *numPrim* CSGPrim,
canonically invoked by CSGFoundry::getPrimSpecHost and CSGFoundry::getPrimSpecDevice
which provide the CSGPrim bbox pointers for all CSGPrim within a CSGSolid.::

    1075     const CSGSolid* so = solid.data() + solidIdx ;  // get the primOffset from CPU side solid
    1076     SCSGPrimSpec ps = CSGPrim::MakeSpec( d_prim,  so->primOffset, so->numPrim ); ;

This can be done very simply for both host and device due to the contiguous storage
of the CSGPrim in the foundry and fixed strides.

SCSGPrimSpec::primitiveIndexOffset
    Primitive index bias, applied in optixGetPrimitiveIndex() so the primIdx
    obtained in closesthit__ch__ is absolute to the entire geometry
    instead of the default of being local to the compound solid.
    (see GAS_Builder::MakeCustomPrimitivesBI_11N)

    This primIdx obtained for each intersect is combined with the
    optixGetInstanceId() to give the intersect identity.


How to implement Prim selection ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applying Prim selection based on meshIdx/lvIdx of each
Prim still requires to iterate over them all.
Better to apply selection in one place only.
So where to apply prim selection ?

SCSGPrimSpec is too late as the prim array handled
there needs to be memory contiguous.
This suggests addition of selected_prim to CSGFoundry::

    std::vector<CSGPrim>  prim ;
    std::vector<CSGPrim>  selected_prim ;

Must also ensure no blind passing of primOffsets as they
will be invalid.





**/

SCSGPrimSpec CSGPrim::MakeSpec( const CSGPrim* prim0,  unsigned primIdx, unsigned numPrim ) // static
{
    const CSGPrim* prim = prim0 + primIdx ;

    SCSGPrimSpec ps ;
    ps.aabb = prim->AABB() ;
    ps.sbtIndexOffset = prim->sbtIndexOffsetPtr() ;
    ps.num_prim = numPrim ;
    ps.stride_in_bytes = sizeof(CSGPrim);
    ps.primitiveIndexOffset = primIdx ;

    return ps ;
}

#endif


