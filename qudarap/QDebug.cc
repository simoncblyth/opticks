#include <cuda_runtime.h>

#include <sstream>

#include "scuda.h"
#include "squad.h"
#include "sscint.h"
#include "scerenkov.h"

#include "QDebug.hh"
#include "QState.hh"
#include "qdebug.h"
#include "QU.hh"
#include "PLOG.hh"

const plog::Severity QDebug::LEVEL = PLOG::EnvLevel("QDebug", "DEBUG") ; 
const QDebug* QDebug::INSTANCE = nullptr ; 
const QDebug* QDebug::Get(){ return INSTANCE ; }

/**
QDebug::QDebug
----------------

**/


qdebug* QDebug::MakeInstance()   // static
{
   // miscellaneous used by fill_state testing 

    qdebug* dbg = new qdebug ; 

    float cosTheta = 0.5f ; 
    dbg->wavelength = 500.f ; 
    dbg->cosTheta = cosTheta ; 
    qvals( dbg->normal , "DBG_NRM", "0,0,1" ); 
   
    // qstate: mocking result of fill_state 
    dbg->s = QState::Make(); 

    // quad2: mocking prd per-ray-data result of optix trace calls 
    dbg->prd = quad2::make_eprd() ;  // see qudarap/tests/eprd.sh 
     
    sscint& scint_gs = dbg->scint_gs ; 
    sscint::FillGenstep( scint_gs, 0, 100 ); 

    scerenkov& cerenkov_gs = dbg->cerenkov_gs ; 
    scerenkov::FillGenstep( cerenkov_gs, 0, 100 ); 

    return dbg ; 
}

QDebug::QDebug()
    :
    dbg(MakeInstance()),
    d_dbg(QU::UploadArray<qdebug>(dbg, 1 ))
{
    INSTANCE = this ; 
    LOG(info) << desc() ; 
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


