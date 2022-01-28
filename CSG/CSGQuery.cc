#include "PLOG.hh"

#include "CSGFoundry.h"
#include "CSGQuery.h"
#include "CSGGrid.h"

#include "OpticksCSG.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"


CSGQuery::CSGQuery( const CSGFoundry* fd_ ) 
    :
    fd(fd_),
    prim0(fd->getPrim(0)),
    node0(fd->getNode(0)),
    plan0(fd->getPlan(0)),
    itra0(fd->getItra(0)),
    select_prim(nullptr),
    select_numNode(0),
    select_root(nullptr)
{
   init(); 
}


void CSGQuery::init()
{
    selectPrim(0u,0u);
}


void CSGQuery::selectPrim(unsigned solidIdx, unsigned primIdxRel )
{
    // TODO: use MOI to identify the prim to select, here just picking the first 
    const CSGPrim* pr = fd->getSolidPrim(solidIdx, primIdxRel); 
    assert( pr );  
    selectPrim(pr); 
}

void CSGQuery::selectPrim( const CSGPrim* pr )
{
    select_prim = pr ; 
    select_numNode = pr->numNode() ; 
    select_root = node0 + pr->nodeOffset() ; 

    LOG(info) 
         << " select_prim " << select_prim
         << " select_numNode " << select_numNode
         << " select_root " << select_root 
         ;   
}


void CSGQuery::dumpPrim() const 
{
    const CSGPrim* pr = select_prim ;
    if( pr == nullptr ) return ;

    int numNode = select_numNode ;
    for(int nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset()+numNode ; nodeIdx++)
    {
        const CSGNode* nd = node0 + nodeIdx ;
        LOG(info) << "    nodeIdx " << std::setw(5) << nodeIdx << " : " << nd->desc() ;
    }
}

float CSGQuery::operator()(const float3& position ) const
{
    return distance_tree( position, select_numNode, select_root, plan0, itra0 ) ;
}

bool CSGQuery::intersect( quad4& isect,  float t_min, const quad4& p ) const 
{
    float3 ray_origin    = make_float3(  p.q0.f.x, p.q0.f.y, p.q0.f.z );  
    float3 ray_direction = make_float3(  p.q1.f.x, p.q1.f.y, p.q1.f.z );  
    return intersect( isect, t_min, ray_origin, ray_direction ); 
}

bool CSGQuery::intersect( quad4& isect,  float t_min, const float3& ray_origin, const float3& ray_direction ) const 
{
    isect.zero();
    bool valid_intersect = intersect_tree(isect.q0.f, select_numNode, select_root, plan0, itra0, t_min, ray_origin, ray_direction );  

    if( valid_intersect )
    {   
        float t = isect.q0.f.w ; 
        float3 ipos = ray_origin + t*ray_direction ;   
        isect.q1.f.x = ipos.x ;
        isect.q1.f.y = ipos.y ;
        isect.q1.f.z = ipos.z ;
        isect.q1.f.w = t ; 

        isect.q2.f.x = ray_origin.x ; 
        isect.q2.f.y = ray_origin.y ; 
        isect.q2.f.z = ray_origin.z ;
        isect.q2.f.w = t_min ; 

        isect.q3.f.x = ray_direction.x ; 
        isect.q3.f.y = ray_direction.y ; 
        isect.q3.f.z = ray_direction.z ;
    }   
    return valid_intersect ; 
}


CSGGrid* CSGQuery::scanPrim(int resolution) const 
{
    const CSGPrim* pr = select_prim ;
    if( pr == nullptr )
    {
        LOG(fatal) << " no prim is selected " ;
        return nullptr ;
    }

    const float4 ce =  pr->ce() ;
    CSGGrid* grid = new CSGGrid( ce, resolution, resolution, resolution );
    grid->scan(*this) ;
    return grid ;
}



