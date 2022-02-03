#include <csignal>

#include "SSys.hh"
#include "SPath.hh"
#include "PLOG.hh"

#include "scuda.h"
#include "squad.h"
#include "sc4u.h"

#include "SCenterExtentGenstep.hh"
#include "NP.hh"

#include "CSGQuery.h"
#include "CSGDraw.h"
#include "CSGFoundry.h"
#include "CSGGeometry.h"
#include "CSGRecord.h"
#include "CSGGrid.h"



const char* CSGGeometry::OutDir(const char* cfbase, const char* geom)   // static  
{
    if(cfbase == nullptr || geom == nullptr)
    {
        LOG(fatal) << " require CFBASE and GEOM to control output dir " ; 
        return nullptr ;   
    }
    int create_dirs = 2 ;   
    const char* outdir = SPath::Resolve(cfbase, "CSGIntersectSolidTest", geom, create_dirs );  
    return outdir ;  
}


CSGGeometry::CSGGeometry(const CSGFoundry* fd_)
    :
    default_geom(nullptr),
    geom(SSys::getenvvar("GEOM", default_geom)),
    cfbase(SSys::getenvvar("CFBASE")), 
    outdir(OutDir(cfbase,geom)),
    name(nullptr),
    fd(fd_),
    q(nullptr),
    d(nullptr)
{
    LOG(info) << " GEOM " << geom  ; 
    init(); 
}

void  CSGGeometry::init()
{
    if( fd == nullptr )
    {
        if( geom != nullptr && cfbase == nullptr)
        {
            name = strdup(geom); 
            LOG(info) << "init from GEOM " << geom << " name " << name ; 
            fd = CSGFoundry::Make(geom) ; 
        }
        else if(cfbase != nullptr)
        {
            name = SPath::Basename(cfbase); 
            LOG(info) << "init from CFBASE " << cfbase << " name " << name  ; 
            fd = CSGFoundry::Load(cfbase, "CSGFoundry");
        }
        else
        {
            LOG(fatal) << " neither GEOM or CFBASE envvars are defined and fd pointer not provided : ABORT " ; 
            std::raise(SIGINT); 
        }
    }
    else
    {
        LOG(info) << " booting from provided CSGFoundry pointer " ; 
        cfbase = fd->getCFBase(); 
    }
    q = new CSGQuery(fd); 
    d = new CSGDraw(q) ; 
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


/**
CSGGeometry::centerExtentGenstepIntersect
-------------------------------------------

When SELECT_ISECT envvar is set to the path of an isect.npy 
those intersects are rerun. This is convenient for debugging 
a selection of intersects. 

**/

void CSGGeometry::centerExtentGenstepIntersect() 
{
    const char* path = SSys::getenvvar("SELECTED_ISECT"); 
    if(path == nullptr)
    {
        saveCenterExtentGenstepIntersect(); 
    }
    else
    {
        intersectSelected(path); 
    }

    draw("CSGGeometry::centerExtentGenstepIntersect"); 
}


void CSGGeometry::saveCenterExtentGenstepIntersect() const 
{
    SCenterExtentGenstep* cegs = new SCenterExtentGenstep ; 
    const std::vector<quad4>& pp = cegs->pp ; 
    std::vector<quad4>& ii = cegs->ii ; 

    float t_min = 0.f ; 
    quad4 isect ;
    for(unsigned i=0 ; i < pp.size() ; i++)
    {   
        const quad4& p = pp[i]; 
        if(q->intersect(isect, t_min, p )) ii.push_back(isect); 
    }   

    unsigned num_isect = ii.size() ;
    LOG(info) 
        << " pp.size " << pp.size()
        << " ii.size " << num_isect
        << ( num_isect == 0 ? " WARNING : NO INTERSECTS " : " " )
        ;

    cegs->save(outdir); 
}

void CSGGeometry::intersectSelected(const char* path)
{
    NP* a = NP::Load(path); 
    if( a == nullptr ) LOG(fatal) << " FAILED to load from path " << path ; 
    if( a == nullptr ) return ; 

    bool expected_shape = a->has_shape(-1,4,4) ;  
    assert(expected_shape); 
    LOG(info) << " load SELECTED_ISECT from " << path << " a.sstr " << a->sstr() ; 

    unsigned num_isect = a->shape[0] ;
    std::vector<quad4> isects(num_isect) ; 
    memcpy( isects.data(), a->bytes(),  sizeof(float)*16 );    

    for(unsigned i=0 ; i < num_isect ; i++)
    {
        const quad4& prev_isect = isects[i] ; 

        int4 gsid ; 
        C4U_decode(gsid, prev_isect.q3.u.w ); 

        quad4 isect ;
        bool valid_intersect = q->intersect_again(isect, prev_isect );  

        std::string gsid_name = C4U_name(gsid, "gsid", '_' ); 

#ifdef DEBUG_RECORD
        int create_dirs = 2 ; // 2:dirpath 
        const char* dir = SPath::Resolve(outdir, "intersectSelected", gsid_name.c_str(), create_dirs ); 
        CSGRecord::Save(dir); 
        if(i == 0) CSGRecord::Dump(dir); 
        CSGRecord::Clear(); 
#endif

        std::cout 
            << " gsid " << C4U_desc(gsid) 
            << " gsid_name " << gsid_name 
            << " valid_intersect " << valid_intersect  
            << " t:prev_isect.q0.f.w " 
            << std::setw(10) << std::fixed << std::setprecision(4) << prev_isect.q0.f.w 
            << " t:isect.q0.f.w " 
            << std::setw(10) << std::fixed << std::setprecision(4) << isect.q0.f.w 
            << std::endl 
            ;
    }
}

void CSGGeometry::dump(const char* msg) const 
{
    q->dump(msg); 
}
void CSGGeometry::draw(const char* msg) 
{
    d->draw(msg); 
}

void CSGGeometry::Draw( const CSGFoundry* fd, const char* msg  ) // static 
{
    CSGGeometry cgeo(fd); 
    cgeo.draw(msg); 
}
