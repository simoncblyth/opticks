#include "SSys.hh"
#include "SPath.hh"
#include "PLOG.hh"

#include "scuda.h"
#include "squad.h"
#include "SCenterExtentGenstep.hh"

#include "CSGQuery.h"
#include "CSGFoundry.h"
#include "CSGGeometry.h"
#include "CSGGrid.h"

CSGGeometry::CSGGeometry()
    :
    default_geom(nullptr),
    geom(SSys::getenvvar("GEOM", default_geom)),
    cfbase(SSys::getenvvar("CFBASE")), 
    name(nullptr),
    fd(nullptr),
    q(nullptr)
{
    LOG(info) << " GEOM " << geom  ; 
    init(); 
}

void  CSGGeometry::init()
{
    if( geom != nullptr && cfbase == nullptr)
    {
        init_geom(); 
    }
    else if(cfbase != nullptr)
    {
        init_cfbase();
    } 
    else
    {
        LOG(fatal) << " neither GEOM or CFBASE envvars are defined " ; 
        return ; 
    }
    q = new CSGQuery(fd); 
}

void  CSGGeometry::init_geom()
{
    name = strdup(geom); 
    LOG(info) << "init from GEOM " << geom << " name " << name ; 
    fd = CSGFoundry::Make(geom) ; 
}
void  CSGGeometry::init_cfbase()
{
    name = SPath::Basename(cfbase); 
    LOG(info) << "init from CFBASE " << cfbase << " name " << name  ; 
    fd = CSGFoundry::Load(cfbase, "CSGFoundry");
}

void CSGGeometry::saveSignedDistanceField() const 
{
    int resolution = SSys::getenvint("RESOLUTION", 25) ; 
    LOG(info) << " name " << name << " RESOLUTION " << resolution ; 
    q->dumpPrim();

    const CSGGrid* grid = q->scanPrim(resolution); 
    assert( grid );  
    grid->save(name);  
}

void CSGGeometry::saveCenterExtentGenstepIntersect() const 
{
    if(cfbase == nullptr || geom == nullptr)
    {
        LOG(fatal) << " require CFBASE and GEOM to control output dir " ; 
        return ;   
    }
    int create_dirs = 2 ;   
    const char* outdir = SPath::Resolve(cfbase, "CSGIntersectSolidTest", geom, create_dirs );  


    
    SCenterExtentGenstep* cegs = new SCenterExtentGenstep ; 

    const std::vector<quad4>& pp = cegs->pp ; 
    std::vector<quad4>& ii = cegs->ii ; 
    float t_min = 0.f ; 
    quad4 isect ;

    for(unsigned i=0 ; i < pp.size() ; i++)
    {   
        const quad4& p = pp[i]; 
        bool valid_intersect = q->intersect(isect, t_min, p );  
        if( valid_intersect )
        {   
            ii.push_back(isect);
        }   
    }   
    cegs->save(outdir); 
}



