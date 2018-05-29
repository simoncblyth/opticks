#pragma once

// Umbrella header bringing in  headers from all linked Opticks subprojects 
// allowing the logging of each subproject to be individually controlled.

#include "PLOG.hh"

#ifdef WITH_SYSRAP
#include "SYSRAP_LOG.hh"
#endif
#ifdef WITH_BRAP
#include "BRAP_LOG.hh"
#endif
#ifdef WITH_NPY
#include "NPY_LOG.hh"
#endif
#ifdef WITH_OKCORE
#include "OKCORE_LOG.hh"
#endif
#ifdef WITH_GGEO
#include "GGEO_LOG.hh"
#endif
#ifdef WITH_ASIRAP
#include "ASIRAP_LOG.hh"
#endif
#ifdef WITH_MESHRAP
#include "MESHRAP_LOG.hh"
#endif
#ifdef WITH_OKGEO
#include "OKGEO_LOG.hh"
#endif
#ifdef WITH_OGLRAP
#include "OGLRAP_LOG.hh"
#endif
#ifdef WITH_OKGL
#include "OKGL_LOG.hh"
#endif
#ifdef WITH_OK
#include "OK_LOG.hh"
#endif
#ifdef WITH_OKG4
#include "OKG4_LOG.hh"
#endif
#ifdef WITH_CUDARAP
#include "CUDARAP_LOG.hh"
#endif
#ifdef WITH_THRAP
#include "THRAP_LOG.hh"
#endif
#ifdef WITH_OXRAP
#include "OXRAP_LOG.hh"
#endif
#ifdef WITH_OKOP
#include "OKOP_LOG.hh"
#endif
#ifdef WITH_CFG4
#include "CFG4_LOG.hh"
#endif
#ifdef WITH_G4OK
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



