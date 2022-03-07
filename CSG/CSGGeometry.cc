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

const plog::Severity CSGGeometry::LEVEL = PLOG::EnvLevel("CSGGeometry", "DEBUG") ; 


void CSGGeometry::operator()()
{
    LOG(info) ; 

    bool dump_ = SSys::getenvbool("DUMP"); 
    if(dump_)
    {
        LOG(info) << " dump  " ; 
        dump();  
        draw(); 
    }
    else
    {
        LOG(info) << " centerExtentGenstepIntersect " ; 
        centerExtentGenstepIntersect();  
    }
}


const char* CSGGeometry::OutDir(const char* cfbase, const char* geom, const char* sopr)   // static  
{
    if(cfbase == nullptr )
    {
        LOG(fatal) << " require CFBASE to control output dir " ; 
        return nullptr ;   
    }

    const char* reldir = geom ? geom : sopr ; 

    if(reldir == nullptr )
    {
        LOG(fatal) << " require geom or sopr to control reldir included in output dir " ; 
        return nullptr ;   
    }

    int create_dirs = 2 ;   
    const char* outdir = SPath::Resolve(cfbase, "CSGIntersectSolidTest", reldir, create_dirs );  
    return outdir ;  
}

/**
CSGGeometry::CSGGeometry
-------------------------

Can boot in three ways:

1. externally created CSGFoundry instance
2. GEOM envvar identifying a test geometry, and resulting in creation of a CSGFoundry instance  
3. CFBASE envvar identifying directory containing a persisted CSGFoundry geometry that is loaded

**/

CSGGeometry::CSGGeometry(const char* default_cfbase, const CSGFoundry* fd_)
    :
    default_geom(nullptr),
    default_sopr(nullptr),
    geom(SSys::getenvvar("GEOM", default_geom)),
    sopr(SSys::getenvvar("SOPR", default_sopr)),
    cfbase(SSys::getenvvar("CFBASE", default_cfbase)), 
    outdir(OutDir(cfbase,geom,sopr)),
    name(nullptr),
    fd(fd_),
    q(nullptr),
    ce(nullptr),
    d(nullptr),
    sxyzw(SSys::getenvintvec("SXYZW",',')),
    sxyz(SSys::getenvintvec("SXYZ",',')),
    no_selection(sxyzw == nullptr && sxyz == nullptr),
    sx(0),
    sy(0),
    sz(0),
    sw(0)
{
    LOG(info) << " GEOM " << geom  ; 
    init(); 
}

void CSGGeometry::init()
{
    init_fd(); 
    q = new CSGQuery(fd); 
    q->dumpPrim("CSGGeometry::init"); 
    ce = new float4(q->select_prim->ce()) ; 
    d = new CSGDraw(q, 'Z') ; 
    init_selection(); 
}

void CSGGeometry::init_fd()
{
    if( fd == nullptr )
    {
        if( geom != nullptr && cfbase == nullptr)
        {
            name = strdup(geom); 
            LOG(info) << "init from GEOM " << geom << " name " << name ; 
            fd = CSGFoundry::MakeGeom(geom) ; 
        }
        else if(cfbase != nullptr)
        {
            name = SPath::Basename(cfbase); 
            LOG(info) << "init from CFBASE " << cfbase << " name " << name  ; 
            fd = CSGFoundry::Load(cfbase, "CSGFoundry");
            LOG(info) << " fd.meta\n" << ( fd->hasMeta() ? fd->meta : " NO meta " ) ; 
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
}

void CSGGeometry::init_selection()
{
    if(sxyz)
    { 
        bool sxyz_expect = sxyz->size() == 3 ; 
        if(!sxyz_expect)
        {
            LOG(fatal) << "SXYZ envvar is provided but with size: " << sxyz->size() << " when 3 is expected" ; 
            assert(0); 
        }  
        sx = (*sxyz)[0] ; 
        sy = (*sxyz)[1] ; 
        sz = (*sxyz)[2] ; 

        LOG(info) << "SXYZ (sx,sy,sz) " << "(" << sx << "," << sy << "," << sz << ")" ; 
    }
    else if(sxyzw)
    {
        bool sxyzw_expect = sxyzw->size() == 4 ; 
        if(!sxyzw_expect)
        {
            LOG(fatal) << "SXYZW envvar is provided but with size: " << sxyzw->size() << " when 4 is expected" ; 
            assert(0); 
        }  
        sx = (*sxyzw)[0] ; 
        sy = (*sxyzw)[1] ; 
        sz = (*sxyzw)[2] ;
        sw = (*sxyzw)[3] ;
        LOG(info) << "SXYZW (sx,sy,sz,sw) " << "(" << sx << "," << sy << "," << sz << "," << sw << ")" ; 
    }
    else
    {
        LOG(info) << " no SXYZ or SXYZW selection " ; 
    }
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
        float t_min = SSys::getenvfloat("TMIN", 0.f ); 
        saveCenterExtentGenstepIntersect(t_min); 
    }
    else
    {
        intersectSelected(path); 
    }

    draw("CSGGeometry::centerExtentGenstepIntersect"); 
}


void CSGGeometry::saveCenterExtentGenstepIntersect(float t_min) const 
{
    //float4 ce = make_float4(0.f, 0.f, 0.f, 70.f ); 
    SCenterExtentGenstep* cegs = new SCenterExtentGenstep(ce) ; 
    const std::vector<quad4>& pp = cegs->pp ; 
    std::vector<quad4>& ii = cegs->ii ; 

    LOG(info) << "[ pp.size " << pp.size() << " t_min " << std::fixed << std::setw(10) << std::setprecision(4) << t_min  ; 

    C4U gsid ;
    quad4 isect ;
    unsigned num_ray(0); 

    for(unsigned i=0 ; i < pp.size() ; i++)
    {   
        const quad4& p = pp[i]; 
        gsid.u = p.q3.u.w ; 

        int ix = gsid.c4.x ; 
        int iy = gsid.c4.y ; 
        int iz = gsid.c4.z ; 
        int iw = gsid.c4.w ; 

        if( sxyz == nullptr && sxyzw == nullptr )   // no restriction 
        {
            num_ray += 1 ; 
            if(q->intersect(isect, t_min, p )) ii.push_back(isect); 
        }
        else if( sxyz && ix == sx && iy == sy && iz == sz )    // restrict to single genstep 
        {
            num_ray += 1 ; 
            if(q->intersect(isect, t_min, p )) ii.push_back(isect); 
        }
        else if( sxyzw && ix == sx && iy == sy && iz == sz && iw == sw )  // restrict to single photon 
        {
            num_ray += 1 ; 
            LOG(info) << "[ single photon selected" ; 
            bool valid_isect = q->intersect(isect, t_min, p ) ; 
            if(valid_isect) ii.push_back(isect); 
            LOG(info) << std::endl << CSGQuery::Desc(isect, "single photon selected", &valid_isect ) ;  

#ifdef DEBUG_RECORD
            std::string gsid_name = C4U_name( gsid.u , "gsid", '_' );  
            int create_dirs = 2 ; // 2:dirpath 
            const char* dir = SPath::Resolve(outdir, "saveCenterExtentGenstepIntersect", gsid_name.c_str(), create_dirs ); 
            LOG(info) << " DEBUG_RECORD saving to " << dir ; 
            CSGRecord::Save(dir); 
            CSGRecord::Dump(dir); 
            CSGRecord::Clear(); 
#else
            LOG(info) << " DEBUG_RECORD flag not enabled " ; 
#endif
            LOG(info) << "] single photon selected " ; 
        }
    }   

    unsigned num_isect = ii.size() ;
    LOG(info) 
        << " pp.size " << pp.size()
        << " num_ray " << num_ray 
        << " ii.size " << num_isect
        << ( num_isect == 0 ? " WARNING : NO INTERSECTS " : " " )
        ;

    if( no_selection )
    {
        LOG(info) << " saving to outdir " << outdir ; 
        cegs->save(outdir); 
    }
    else
    {
        LOG(info) << " NOT saving as an SXYZ or SXYZW selection is in use " ; 
        
    }

    LOG(info) << "]" ; 
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
    LOG(error) << "fd.dumpSolid" ; 
    fd->dumpSolid(); 
    LOG(error) << "fd.dumpPrim" ; 
    fd->dumpPrim(); 
    LOG(error) << "fd.dumpNode" ; 
    fd->dumpNode(); 
    LOG(error) << "q.dump" ; 
    q->dump(msg); 
}
void CSGGeometry::draw(const char* msg) 
{
    d->draw(msg); 
}

void CSGGeometry::Draw( const CSGFoundry* fd, const char* msg  ) // static 
{
    CSGGeometry cgeo(nullptr, fd); 
    cgeo.draw(msg); 
}
