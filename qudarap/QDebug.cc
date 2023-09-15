#include <cuda_runtime.h>

#include <sstream>

#include "scuda.h"
#include "squad.h"
#include "sscint.h"
#include "scerenkov.h"
#include "ssys.h"

#include "QBnd.hh"
#include "qbnd.h"

#include "QDebug.hh"
#include "QState.hh"
#include "qdebug.h"
#include "QU.hh"
#include "SLOG.hh"

const plog::Severity QDebug::LEVEL = SLOG::EnvLevel("QDebug", "DEBUG") ; 
const QDebug* QDebug::INSTANCE = nullptr ; 
const QDebug* QDebug::Get(){ return INSTANCE ; }

/**
QDebug::MakeInstance
------------------------

qdebug.h contains miscellaneous used by fill_state testing 

**/

qdebug* QDebug::MakeInstance()   // static
{
    qdebug* dbg = new qdebug ; 

    float cosTheta = 0.5f ; 
    dbg->wavelength = 500.f ; 
    dbg->cosTheta = cosTheta ; 
    qvals( dbg->normal , "DBG_NRM", "0,0,1" ); 
    qvals( dbg->direction , "DBG_DIR", "0,0,-1" ); 
    dbg->orient = 1.f ;  // orient -1.f flips the normal direction 
    dbg->value = ssys::getenvfloat("DBG_VALUE", 0.2f)  ;  // eg sigma_alpha or polish 

    // qstate: mocking result of fill_state 
    dbg->s = QState::Make(); 

    // quad2: mocking prd per-ray-data result of optix trace calls 
    dbg->prd = quad2::make_eprd() ;  // see qudarap/tests/eprd.sh 
    
    dbg->p.ephoton() ;   // sphoton::ephoton 
 
    sscint& scint_gs = dbg->scint_gs ; 
    sscint::FillGenstep( scint_gs, 0, 100 ); 

    scerenkov& cerenkov_gs = dbg->cerenkov_gs ; 

    const QBnd* qb = QBnd::Get() ; 

    unsigned cerenkov_matline = qb ? qb->qb->boundary_tex_MaterialLine_LS : 0 ;   

    LOG_IF(error, qb == nullptr) 
         << "AS NO QBnd at QDebug::MakeInstance the qdebug cerenkov genstep is using default matline of zero " << std::endl 
         << "THIS MEANS qdebug CERENKOV GENERATION WILL LIKELY INFINITE LOOP AND TIMEOUT " << std::endl 
         << " cerenkov_matline " << cerenkov_matline  << std::endl
         << " TO FIX THIS YOU PROBABLY NEED TO RERUN THE GEOMETRY CONVERSION TO UPDATE THE PERSISTED SSim IN CSGFoundry/SSim "
         ;

    scerenkov::FillGenstep( cerenkov_gs, cerenkov_matline, 100 ); 

    return dbg ; 
}

QDebug::QDebug()
    :
    dbg(MakeInstance()),
    d_dbg(QU::UploadArray<qdebug>(dbg, 1, "QDebug::QDebug/d_dbg" ))
{
    INSTANCE = this ; 
    LOG(LEVEL) << desc() ; 
}

qdebug* QDebug::getDevicePtr() const
{
    return d_dbg ; 
}
 
std::string QDebug::desc() const
{
    std::stringstream ss ; 
    ss << "QDebug::desc " 
       << " dbg " << dbg 
       << " d_dbg " << d_dbg 
       << std::endl 
       << " QState::Desc " << QState::Desc(dbg->s)  
       << std::endl 
       << " dbg.p.desc " << dbg->p.desc() 
       ;
    std::string s = ss.str(); 
    return s ; 
}


