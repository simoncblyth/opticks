#include "SSys.hh"

#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"
#include "CSGGrid.h"

#include "OpticksCSG.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "OPTICKS_LOG.hh"


struct CSGSignedDistanceFieldTest
{
    const CSGFoundry& fd ; 
    const CSGNode* node0 ;  
    const float4*  plan0 ; 
    const qat4*    itra0 ; 

    const CSGPrim* select_prim ; 
    int            select_numNode ; 
    const CSGNode* select_root ; 
     

    CSGSignedDistanceFieldTest( const CSGFoundry& fd_ ); 

    void     selectPrim(const CSGPrim* pr ); 
    void     dumpPrim();
    CSGGrid* scanPrim(unsigned nn );

    float operator()(const float3& position) const ; 

};

CSGSignedDistanceFieldTest::CSGSignedDistanceFieldTest( const CSGFoundry& fd_ )
    :
    fd(fd_),
    node0(fd.node.data()),
    plan0(fd.plan.data()),
    itra0(fd.itra.data()),
    select_prim(nullptr),
    select_numNode(0),
    select_root(nullptr)
{
}

void CSGSignedDistanceFieldTest::selectPrim( const CSGPrim* pr )
{
    select_prim = pr ; 
    select_numNode = pr->numNode() ; 
    select_root = node0 + pr->nodeOffset() ; 
}

void CSGSignedDistanceFieldTest::dumpPrim()
{
    const CSGPrim* pr = select_prim ; 
    int numNode = select_numNode ; 
    for(int nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset()+numNode ; nodeIdx++)
    {    
        const CSGNode* nd = node0 + nodeIdx ; 
        LOG(info) << "    nodeIdx " << std::setw(5) << nodeIdx << " : " << nd->desc() ; 
    }
}
float CSGSignedDistanceFieldTest::operator()(const float3& position ) const 
{
    return distance_tree( position, select_numNode, select_root, plan0, itra0 ) ; 
}

CSGGrid* CSGSignedDistanceFieldTest::scanPrim( unsigned resolution )
{
    const CSGPrim* pr = select_prim ; 
    const float4 ce =  pr->ce() ; 
    CSGGrid* grid = new CSGGrid( ce, resolution, resolution, resolution ); 
    grid->scan(*this) ; 
    return grid ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* geom = SSys::getenvvar("GEOM", "UnionBoxSphere") ; 
    int resolution = SSys::getenvint("RESOLUTION", 25 ); 

    LOG(info) << " GEOM " << geom << " RESOLUTION " << resolution ; 
    
    CSGFoundry fd ;
    CSGMaker* mk = fd.maker ;
    CSGSolid* so = mk->make( geom ); 

    LOG(info) << " so " << so ; 
    LOG(info) << " so.desc " << so->desc() ; 
    LOG(info) << " fd.desc " << fd.desc() ; 

    CSGSignedDistanceFieldTest tst(fd); 

    // HMM: should use moi machinery perhaps, so can grab from the standard geometry in addition to CSGMaker demo prim 

    unsigned solidIdx = 0 ; 
    unsigned primIdxRel = 0 ; 
    const CSGPrim* pr = fd.getSolidPrim(solidIdx, primIdxRel); 
    tst.selectPrim(pr); 

    tst.dumpPrim(); 
    const CSGGrid* grid = tst.scanPrim(resolution); 
    grid->save(geom);  

    return 0 ; 
}
