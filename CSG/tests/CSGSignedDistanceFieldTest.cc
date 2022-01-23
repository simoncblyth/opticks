#include "SSys.hh"
#include "SPath.hh"

#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"
#include "CSGGrid.h"

#include "Opticks.hh"
#include "OpticksCSG.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "OPTICKS_LOG.hh"


struct CSGSignedDistanceFieldTest
{
    const char* default_geom ; 
    const char* geom ; 
    const char* cfbase ; 
    const char* name ; 
    int resolution ; 

    const CSGFoundry* fd ; 
    const CSGNode* node0 ;  
    const float4*  plan0 ; 
    const qat4*    itra0 ; 

    const CSGPrim* select_prim ; 
    int            select_numNode ; 
    const CSGNode* select_root ; 
     

    CSGSignedDistanceFieldTest();
    void     initFD();  
    void     initFD_geom();  
    void     initFD_cfbase();  

    void     selectPrim(const CSGPrim* pr ); 
    void     dumpPrim();
    CSGGrid* scanPrim();

    float operator()(const float3& position) const ; 

};

CSGSignedDistanceFieldTest::CSGSignedDistanceFieldTest()
    :
    //default_geom("UnionBoxSphere"),
    default_geom(nullptr),
    geom(SSys::getenvvar("GEOM", default_geom)),
    cfbase(SSys::getenvvar("CFBASE")), 
    name(nullptr),
    resolution(SSys::getenvint("RESOLUTION", 25)),
    fd(nullptr),
    node0(nullptr),
    plan0(nullptr),
    itra0(nullptr),
    select_prim(nullptr),
    select_numNode(0),
    select_root(nullptr)
{
    LOG(info) << " GEOM " << geom << " RESOLUTION " << resolution ; 
    initFD(); 
}

void  CSGSignedDistanceFieldTest::initFD_geom()
{
    fd = new CSGFoundry ; 
    node0 = fd->node.data() ; 
    plan0 = fd->plan.data() ;
    itra0 = fd->itra.data() ;

    name = strdup(geom); 
    LOG(info) << "init from GEOM " << geom << " name " << name ; 
    CSGMaker* mk = fd->maker ;
    CSGSolid* so = mk->make( geom ); 
    LOG(info) << " so " << so ; 
    LOG(info) << " so.desc " << so->desc() ; 
    LOG(info) << " fd.desc " << fd->desc() ; 
}
void  CSGSignedDistanceFieldTest::initFD_cfbase()
{
    name = SPath::Basename(cfbase); 
    LOG(info) << "init from CFBASE " << cfbase << " name " << name  ; 


    fd = CSGFoundry::Load(cfbase, "CSGFoundry");
    //fd->upload();

    node0 = fd->node.data() ; 
    plan0 = fd->plan.data() ;
    itra0 = fd->itra.data() ;
}

void  CSGSignedDistanceFieldTest::initFD()
{
    if( geom != nullptr && cfbase == nullptr)
    {
        initFD_geom(); 
    }
    else if(cfbase != nullptr)
    {
        initFD_cfbase();
    } 
    else
    {
        LOG(fatal) << " neither GEOM or CFBASE envvars are defined " ; 
        return ; 
    }

    // TODO: use MOI to identify the prim to select, here just picking the first 
    unsigned solidIdx = 0 ; 
    unsigned primIdxRel = 0 ; 
    const CSGPrim* pr = fd->getSolidPrim(solidIdx, primIdxRel); 
    selectPrim(pr); 
}


void CSGSignedDistanceFieldTest::selectPrim( const CSGPrim* pr )
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

void CSGSignedDistanceFieldTest::dumpPrim()
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
float CSGSignedDistanceFieldTest::operator()(const float3& position ) const 
{
    return distance_tree( position, select_numNode, select_root, plan0, itra0 ) ; 
}

CSGGrid* CSGSignedDistanceFieldTest::scanPrim()
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

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGSignedDistanceFieldTest tst ;
    tst.dumpPrim();
 
    const CSGGrid* grid = tst.scanPrim(); 

    if( grid )
    {
        grid->save(tst.name);  
    }
    else
    {
        LOG(fatal) << "failed to create grid " ; 
    }

    return 0 ; 
}
