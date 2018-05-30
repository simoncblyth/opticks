#pragma once

// Umbrella header bringing in  headers from all linked Opticks subprojects 
// allowing the logging of each subproject to be individually controlled.

#include "PLOG.hh"

#ifdef OPTICKS_SYSRAP
#include "SYSRAP_LOG.hh"
#endif
#ifdef OPTICKS_BRAP
#include "BRAP_LOG.hh"
#endif
#ifdef OPTICKS_NPY
#include "NPY_LOG.hh"
#endif
#ifdef OPTICKS_OKCORE
#include "OKCORE_LOG.hh"
#endif
#ifdef OPTICKS_GGEO
#include "GGEO_LOG.hh"
#endif
#ifdef OPTICKS_ASIRAP
#include "ASIRAP_LOG.hh"
#endif
#ifdef OPTICKS_MESHRAP
#include "MESHRAP_LOG.hh"
#endif
#ifdef OPTICKS_OKGEO
#include "OKGEO_LOG.hh"
#endif
#ifdef OPTICKS_OGLRAP
#include "OGLRAP_LOG.hh"
#endif
#ifdef OPTICKS_OKGL
#include "OKGL_LOG.hh"
#endif
#ifdef OPTICKS_OK
#include "OK_LOG.hh"
#endif
#ifdef OPTICKS_OKG4
#include "OKG4_LOG.hh"
#endif
#ifdef OPTICKS_CUDARAP
#include "CUDARAP_LOG.hh"
#endif
#ifdef OPTICKS_THRAP
#include "THRAP_LOG.hh"
#endif
#ifdef OPTICKS_OXRAP
#include "OXRAP_LOG.hh"
#endif
#ifdef OPTICKS_OKOP
#include "OKOP_LOG.hh"
#endif
#ifdef OPTICKS_CFG4
#include "CFG4_LOG.hh"
#endif
#ifdef OPTICKS_G4OK
#include "G4OK_LOG.hh"
#endif


#include "SYSRAP_API_EXPORT.hh"

struct PLOG ; 

class SYSRAP_API OPTICKS_LOG {
   public:
       static void Initialize(PLOG* instance, void* app1, void* app2 );
       static void Check(const char* msg);
};

#define OPTICKS_LOG_COLOR__(argc, argv) \
{  \
    PLOG_COLOR(argc, argv); \
    OPTICKS_LOG::Initialize(PLOG::instance, plog::get(), NULL );  \
}  \


#define OPTICKS_LOG__(argc, argv) \
{  \
    PLOG_(argc, argv); \
    OPTICKS_LOG::Initialize(PLOG::instance, plog::get(), NULL ); \
} \



