/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <iomanip>
#include <sstream>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

// sysrap-
#include "S_freopen_redirect.hh"
#include "SSys.hh"
#include "SStr.hh"

// brap-
#include "BStr.hh"
#include "BFile.hh"
#include "BTimeStamp.hh"
//#include "STimes.hh"
#include "BTimes.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NMeta.hpp"
#include "GLMFormat.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksEntry.hh"
#include "OpticksBufferControl.hh"

// optixrap-
#include "SPPM.hh"
#include "OConfig.hh"
#include "OContext.hh"
#include "OError.hh"

#include "PLOG.hh"
using namespace optix ; 

const char* OContext::COMPUTE_ = "COMPUTE" ; 
const char* OContext::INTEROP_ = "INTEROP" ; 

plog::Severity OContext::LEVEL = PLOG::EnvLevel("OContext","DEBUG") ; 


const char* OContext::getModeName() const 
{
    switch(m_mode)
    {
       case COMPUTE:return COMPUTE_ ; break ; 
       case INTEROP:return INTEROP_ ; break ; 
    }
    assert(0);
}
const char* OContext::getRunLabel() const 
{
    return m_runlabel ; 
}
const char* OContext::getRunResultsDir() const 
{
    return m_runresultsdir ; 
}

OpticksEntry* OContext::addEntry(char code, const char* from)
{
    LOG(LEVEL) << code << " : " << from ; 

    bool defer = true ; 
    unsigned index ;
    switch(code)
    { 
        case 'G': index = addEntry("generate.cu", "generate", "exception", defer) ; break ;
        case 'T': index = addEntry("generate.cu", "trivial",  "exception", defer) ; break ;
        case 'Z': index = addEntry("generate.cu", "zrngtest",  "exception", defer) ; break ;
        case 'N': index = addEntry("generate.cu", "nothing",  "exception", defer) ; break ;
        case 'R': index = addEntry("generate.cu", "tracetest",  "exception", defer) ; break ;
        case 'D': index = addEntry("generate.cu", "dumpseed", "exception", defer) ; break ;
        case 'S': index = addEntry("seedTest.cu", "seedTest", "exception", defer) ; break ;
        case 'P': index = addEntry("pinhole_camera.cu", "pinhole_camera" , "exception", defer);  break;
    }
    return new OpticksEntry(index, code) ; 
}


unsigned OContext::getDebugPhoton() const 
{
    return m_debug_photon ; 
}

Opticks* OContext::getOpticks() const 
{
    return m_ok ; 
}


// needed by rayleighTest
OConfig* OContext::getConfig() const 
{
    return m_cfg ; 
}



struct Device
{
   int index ; 
   char name[256] ;  
   int computeCaps[2];
   RTsize total_mem;
   int ordinal ;  

   Device(unsigned index_)
      :
      index(index_)
   {
        RT_CHECK_ERROR(rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
        RT_CHECK_ERROR(rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps));
        RT_CHECK_ERROR(rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(total_mem), &total_mem));
        RT_CHECK_ERROR(rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(ordinal), &ordinal));
   } 

   std::string desc() const 
   {
       std::stringstream ss ;  
       ss <<  "Device " << index << " " << std::setw(30) << name 
          << " ordinal " << ordinal 
          <<  " Compute Support: " << computeCaps[0] << " " << computeCaps[1] 
          <<  " Total Memory: " <<  (unsigned long long)total_mem
          ;
       return ss.str();
   }   
};


struct VisibleDevices 
{
    unsigned num_devices;
    unsigned version;
    std::vector<Device> devices ; 

    VisibleDevices()
    {
        RT_CHECK_ERROR(rtDeviceGetDeviceCount(&num_devices));
        RT_CHECK_ERROR(rtGetVersion(&version));
        for(unsigned i = 0; i < num_devices; ++i) 
        {
            Device d(i); 
            devices.push_back(d);     
        }  
    }

    std::string desc() const 
    {
       std::stringstream ss ;  
       for(unsigned i = 0; i < num_devices; ++i) ss << devices[i].desc() << std::endl ; 
       return ss.str();
    }

    std::string brief() const 
    {
       std::stringstream ss ;  
       for(unsigned i = 0; i < num_devices; ++i) 
       {
           std::string name = SStr::Replace( devices[i].name , ' ', '_' ); 
           ss << name ; 
           if( i < num_devices - 1 ) ss << " " ; 
       }
       return ss.str();
    }
};


void OContext::CheckDevices(Opticks* ok)
{
    VisibleDevices vdev ; 
    LOG(info) << std::endl << vdev.desc(); 

    NMeta* parameters = ok->getParameters(); 
    parameters->add<int>("NumDevices", vdev.num_devices );
    parameters->add<std::string>("VisibleDevices", vdev.brief() );

    const char* frame_renderer = Opticks::Instance()->getFrameRenderer();
    if( frame_renderer != NULL)
    {
        if(vdev.num_devices != 1) LOG(fatal) << "vdev.num_devices " << vdev.num_devices ;   
        assert( vdev.num_devices == 1 && "expecting only a single visible device, the one driving the display, in interop mode") ; 
        const char* optix_device = vdev.devices[0].name ;
        LOG(LEVEL) << " frame_renderer " << frame_renderer ; 
        LOG(LEVEL) << " optix_device " << optix_device  ; 
        bool interop_device_match = SStr::Contains( frame_renderer, optix_device )  ; 
        assert( interop_device_match && "OpenGL and OptiX must be talking to the same single device in interop mode"  ); 

        parameters->add<std::string>("FrameRenderer", frame_renderer );
    }
    else
    {
        LOG(LEVEL) << " NULL frame_renderer : compute mode ? " ;  
    }
}


OContext* OContext::Create(Opticks* ok, const char* cmake_target, const char* ptxrel )
{
    LOG(LEVEL) << "[" ;
    OKI_PROFILE("_OContext::Create");

    SetupOptiXCachePathEnvvar();

    NMeta* parameters = ok->getParameters(); 
    int rtxmode = ok->getRTX();
#if OPTIX_VERSION_MAJOR >= 6
    InitRTX( rtxmode ); 
#else
    assert( rtxmode == 0 && "Cannot use --rtx 1/2/-1 options prior to OptiX 6.0.0 " ) ;
#endif
    parameters->add<int>("RTXMode", rtxmode  );

    CheckDevices(ok);

    OKI_PROFILE("_optix::Context::create");
    optix::Context context = optix::Context::create();
    OKI_PROFILE("optix::Context::create");

    OContext* ocontext = new OContext(context, ok, cmake_target, ptxrel );

    OKI_PROFILE("OContext::Create");
    LOG(LEVEL) << "]" ;
    return ocontext ; 
}


/**
OContext::SetupOptiXCachePathEnvvar
-------------------------------------

Avoids permissions problems between multiple 
OptiX users on same node.

**/

const char* OContext::OPTIX_CACHE_PATH_KEY = "OPTIX_CACHE_PATH" ; 
const char* OContext::GetOptiXCachePathEnvvar()
{
    const char* dir = SSys::getenvvar(OPTIX_CACHE_PATH_KEY , NULL ); 
    return dir ;  
}

void OContext::SetupOptiXCachePathEnvvar()
{
    const char* key = OPTIX_CACHE_PATH_KEY ; 
    const char* dir = GetOptiXCachePathEnvvar() ; 
    if( dir == NULL )
    {
        dir = Opticks::OptiXCachePathDefault();    
        LOG(LEVEL) 
            << "envvar " << key 
            << " not defined "
            << "setting it internally to " << dir 
            ; 

        bool overwrite = true ; 
        SSys::setenvvar( key, dir, overwrite ); 

        const char* dir2 = GetOptiXCachePathEnvvar() ; 
        assert( strcmp( dir, dir2) == 0 );   
    }
    else
    {
        LOG(LEVEL) 
            << "envvar " << key 
            << " already defined " << dir 
            ;
    }
}



#if OPTIX_VERSION_MAJOR >= 6
void OContext::InitRTX(int rtxmode)  // static
{
    if(rtxmode == -1)
    {
        LOG(fatal) << " --rtx " << rtxmode << " leaving ASIS "  ;   
    }
    else
    { 
        int rtx0(-1) ;
        RT_CHECK_ERROR( rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx0), &rtx0) );
        assert( rtx0 == 0 );  // despite being zero performance suggests it is enabled

        int rtx = rtxmode > 0 ? 1 : 0 ;       
        LOG(info) << " --rtx " << rtxmode << " setting  " << ( rtx == 1 ? "ON" : "OFF" )  ; 
        RT_CHECK_ERROR( rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx), &rtx));

        int rtx2(-1) ; 
        RT_CHECK_ERROR(rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx2), &rtx2));
        assert( rtx2 == rtx );
    }
}
#endif





//frm optixMeshViewer/optixMeshViewer.cpp
void UsageReportLogger::log( int lvl, const char* tag, const char* msg )
{
    std::cout << "[" << lvl << "][" << std::left << std::setw( 12 ) << tag << "] " << msg;
}



OContext::OContext(optix::Context context, Opticks* ok, const char* cmake_target, const char* ptxrel ) 
    : 
    m_logger(new UsageReportLogger),
    m_context(context),
    m_ok(ok),
    m_cfg(new OConfig(m_context, cmake_target, ptxrel)),
    m_mode(m_ok->isCompute() ? COMPUTE : INTEROP),
    m_debug_photon(m_ok->getDebugIdx()),
    m_entry(0),
    m_closed(false),
    m_verbose(false),
    m_llogpath(NULL),
    m_launch_count(0),
    m_runlabel(m_ok->getRunLabel()),
    m_runresultsdir(m_ok->getRunResultsDir())
{
    init();
    initPrint();
    initDevices();
}


/**
OContext::getRawPointer
------------------------

Used by OCtx::Get to allow the new fangled OCtx context wrapper 
to be used with a preexisting context.

**/

void* OContext::getRawPointer()  
{
    optix::ContextObj* contextObj = m_context.get(); 
    RTcontext context_ptr = contextObj->get(); 
    void* ptr = (void*)context_ptr ; 
    return ptr ; 
}


/**

TODO: investigate new API for controlling stack in OptiX 6 with RTX enabled

* https://devtalk.nvidia.com/default/topic/1047558/optix/stack-overflow-with-optix6-0-0-rtx-2080ti/
* https://raytracing-docs.nvidia.com/optix6/api_6_5/html/group___context.html#gac9dbd3baa30e9f2ece268726862def0a

::

   rtContextSetMaxTraceDepth 
   rtContextSetMaxCallableProgramDepth

* https://raytracing-docs.nvidia.com/optix6/api_6_5/html/group___context.html#ga1da5629dbb8d0090e1ea5590d1e67206

  * default of both these depths is 5, 
    but i can probably set them to 1 !!! THIS MIGHT BE A BIG WIN 

**/



void usageReportCallback( int lvl, const char* tag, const char* msg, void* cbdata )
{
    //frm optixMeshViewer/optixMeshViewer.cpp
    // Route messages to a C++ object (the "logger"), as a real app might do.
    // We could have printed them directly in this simple case.

    UsageReportLogger* logger = reinterpret_cast<UsageReportLogger*>( cbdata );
    logger->log( lvl, tag, msg );  
}



void OContext::init()
{
    InitBufferNames(m_buffer_names); 
    InitDebugBufferNames(m_debug_buffer_names); 

    unsigned int num_ray_type = getNumRayType() ;
    m_context->setRayTypeCount( num_ray_type );   // more static than entry type count

    unsigned stacksize_bytes = m_ok->getStack() ;
    m_ok->set("stacksize", stacksize_bytes );

    m_context->setStackSize(stacksize_bytes);
    LOG(LEVEL) 
        << " mode " << getModeName()
        << " num_ray_type " << num_ray_type 
        << " stacksize_bytes " << stacksize_bytes
        ; 


#if OPTIX_VERSION_MAJOR >= 6
    int maxCallableProgramDepth = m_ok->getMaxCallableProgramDepth();
    if(maxCallableProgramDepth > 0)
    {
        unsigned _maxCallableProgramDepth = m_context->getMaxCallableProgramDepth() ; 
        m_context->setMaxCallableProgramDepth( maxCallableProgramDepth ) ;  
        assert( unsigned(maxCallableProgramDepth) == m_context->getMaxCallableProgramDepth()) ; 

        LOG(error) << " --maxCallableProgramDepth changed from " << _maxCallableProgramDepth << " to " << maxCallableProgramDepth ;  
    }
#endif

#if OPTIX_VERSION_MAJOR >= 6
    int maxTraceDepth = m_ok->getMaxTraceDepth();
    if(maxTraceDepth > 0)
    {
        unsigned _maxTraceDepth = m_context->getMaxTraceDepth() ; 
        m_context->setMaxTraceDepth( maxTraceDepth ) ;  
        assert( unsigned(maxTraceDepth) == m_context->getMaxTraceDepth()) ; 

        LOG(error) << " --maxTraceDepth changed from " << _maxTraceDepth << " to " << maxTraceDepth ;  
    }
#endif

    int usageReportLevel = m_ok->getUsageReportLevel();
    if(usageReportLevel > 0)
    {
        m_context->setUsageReportCallback( usageReportCallback, usageReportLevel, m_logger ); 
    } 

}


/**
OContext::initDevices
-----------------------

This re-implements device introspection CUDA-centrically
based on CDevice so it can work more generally, eg with OptiX7

OKTest without options defaults to writing the below::

    js.py $TMP/default_pfx/evt/dayabay/torch/0/parameters.json


**/

void OContext::initDevices()
{
    const char* dirpath = m_ok->getRuncacheDir(); 
    bool nosave = false ; 
    CDevice::Visible(m_visible_devices, dirpath, nosave ); 
    CDevice::Load(   m_all_devices, dirpath); 

    CDevice::Dump(   m_visible_devices, "Visible devices"); 
    CDevice::Dump(   m_all_devices, "All devices"); 

    NMeta* parameters = m_ok->getParameters(); 
    std::string cdb_all = CDevice::Brief(m_all_devices) ; 
    std::string cdb_vis = CDevice::Brief(m_visible_devices) ;
 
    parameters->add<std::string>("CDeviceBriefAll", cdb_all  ) ; 
    parameters->add<std::string>("CDeviceBriefVis", cdb_vis  ) ;
}


bool OContext::hasTopGroup() const 
{
    return m_top.get() != NULL ;  
}

void OContext::createTopGroup()
{
    if(hasTopGroup()) 
    {
        LOG(error) << " already have m_top Group " ; 
        return ; 
    }

    m_top = m_context->createGroup();
    m_context[ "top_object" ]->set( m_top );
}

optix::Group OContext::getTopGroup()
{
    if(!hasTopGroup())
    {
        createTopGroup();
    }
    return m_top ; 
}


void OContext::initPrint()
{
    LOG(LEVEL) << "[" ; 


    m_context->setPrintBufferSize(4096);
    //m_context->setPrintBufferSize(2*2*2*8192);

    m_context->setExceptionEnabled(RT_EXCEPTION_ALL, false );  
    // disable all exceptions 
    // this is different from the default of leaving STACKOVERFLOW exception enabled


    glm::ivec3 idx ; 
    if(m_ok->getPrintIndex(idx))   // --pindex 0  : 1st index
    {
        m_context->setPrintEnabled(true);
        m_context->setPrintLaunchIndex(idx.x, idx.y, idx.z);
        LOG(LEVEL) << "setPrintLaunchIndex "
                   << " idx.x " << idx.x
                   << " idx.y " << idx.y
                   << " idx.z " << idx.z
                   ; 
    } 
    else if( m_ok->isPrintEnabled()  )   // --printenabled 
    {    
         m_context->setPrintEnabled(true);
         assert( m_context->getPrintEnabled() == true );  
         LOG(info) << " --printenabled " ; 
    }
   /*
    else if( m_ok->hasMask() )   // --mask NNN
    {
         m_context->setPrintEnabled(true);
         assert( m_context->getPrintEnabled() == true );  
         LOG(info) << " --printenabled via the --mask setting " ; 
    }
   */
    else
    {
         return ;  
    }


    // only enable exceptions when print also enabled
    if( m_ok->isExceptionEnabled() )
    {
        m_context->setExceptionEnabled(RT_EXCEPTION_ALL, false );  
    }


    unsigned uindex = m_ok->hasMask() ? m_ok->getMaskIndex(idx.x) : idx.x ; 
    m_llogpath = m_ok->isPrintIndexLog() ?  LaunchLogPath(uindex) : NULL ; 

    LOG(LEVEL) 
        << " idx " << gformat(idx) 
        << " llogpath " << ( m_llogpath ? m_llogpath : "-" )
        ;  
    LOG(LEVEL) << "]" ; 
}



std::string OContext::printDesc() const 
{
    const char* llogpath = getPrintIndexLogPath() ; 

    optix::int3 pindex = m_context->getPrintLaunchIndex();
    bool printenabled = m_context->getPrintEnabled(); 

    std::stringstream ss ;  
    ss << ( printenabled ? " --printenabled " : " " )
       << " printLaunchIndex (" 
       << " " << pindex.x 
       << " " << pindex.y 
       << " " << pindex.z 
       << ") " 
       << ( llogpath ? llogpath : "-" )
       ;

   return ss.str(); 
} 






const char* OContext::getPrintIndexLogPath() const 
{
    return m_llogpath ;  
}


const char* OContext::LaunchLogPath(unsigned index)
{
    const char* name = BStr::concat<unsigned>("ox_", index, ".log" ); 
    std::string path = BFile::FormPath("$TMP", name ); 
    return strdup(path.c_str()); 
}




optix::Context OContext::getContext()
{
     return m_context ; 
}
optix::Context& OContext::getContextRef()
{
     return m_context ; 
}

unsigned int OContext::getNumRayType()
{
    return e_rayTypeCount ;
}

OContext::Mode_t OContext::getMode()
{
    return m_mode ; 
}

bool OContext::isCompute()
{
    return m_mode == COMPUTE ; 
}
bool OContext::isInterop()
{
    return m_mode == INTEROP ; 
}

OContext::~OContext()
{
    cleanUp(); 
}

void OContext::cleanUp()
{
    m_context->destroy();
    m_context = 0;

    // cleanUpCache();
}


/**
OContext::cleanUpCache
--------------------------

Now that control the OPTIX_CACHE_PATH envvar, 
setting it to a path with username such as /var/tmp/simon/OptiXCache
there is less need to delete the cache.

The cache directory on Linux uses a common
path for all users /var/tmp/OptixCache 
which presents a permissons problem on multi-user systems.

OContext::cleanUpCache is a workaround that deletes the 
cache directory at termination so subsequent users can create 
(and delete) their own such directory.

Note that crashes that prevent the running of cleanupCache 
will cause context creation to fail for subsequent users with::

    terminate called after throwing an instance of 'optix::Exception'
    what():  OptiX was unable to open the disk cache with sufficient privileges. Please make sure the database file is writeable by the current user.

**/
void OContext::cleanUpCache()
{
    const char* key = "OPTICKS_KEEPCACHE" ; 
    int keepcache = SSys::getenvint( key, 0 ); 
    const char* cachedir = GetOptiXCachePathEnvvar(); 
    if( keepcache > 0 ) 
    {
        LOG(fatal) << " keeping cache " << cachedir 
                  << " as envvar set " << key 
                  ;  
    }
    else
    {
        LOG(info) << " RemoveDir " << cachedir ; 
        BFile::RemoveDir( cachedir ); 
    }
}



optix::Program OContext::createProgram(const char* cu_filename, const char* progname )
{  
    LOG(LEVEL) << " cu_filename " << cu_filename << " progname " << progname ; 
    optix::Program prog = m_cfg->createProgram(cu_filename, progname);
    return prog ; 
}

unsigned int OContext::addEntry(const char* cu_filename, const char* raygen, const char* exception, bool defer)
{
    return m_cfg->addEntry(cu_filename, raygen, exception, defer ); 
}
unsigned int OContext::addRayGenerationProgram( const char* filename, const char* progname, bool defer)
{
    assert(0);
    return m_cfg->addRayGenerationProgram(filename, progname, defer);
}
unsigned int OContext::addExceptionProgram( const char* filename, const char* progname, bool defer)
{
    assert(0);
    return m_cfg->addExceptionProgram(filename, progname, defer);
}

void OContext::setMissProgram( unsigned int index, const char* filename, const char* progname, bool defer )
{
    m_cfg->setMissProgram(index, filename, progname, defer);
}



void OContext::close()
{
    if(m_closed) return ; 

    m_closed = true ; 

    unsigned int num = m_cfg->getNumEntryPoint() ;

    LOG(debug) << "numEntryPoint " << num ; 

    m_context->setEntryPointCount( num );

    LOG(debug) << "setEntryPointCount done." ;
 
    if(m_verbose) m_cfg->dump("OContext::close");

    m_cfg->apply();

    LOG(debug) << "m_cfg->apply() done." ;

}


void OContext::dump(const char* msg)
{
    m_cfg->dump(msg);
}
unsigned int OContext::getNumEntryPoint()
{
    return m_cfg->getNumEntryPoint();
}


/**
OContext::launch
------------------

Invoked by OPropagator::launch and prelaunch


**/

double OContext::launch(unsigned int lmode, unsigned int entry, unsigned int width, unsigned int height, BTimes* times )
{
    if(!m_closed) close();


    LOG(LEVEL)
              << " entry " << entry 
              << " width " << width 
              << " height " << height 
              << " "
              << printDesc()
              ;

    double dt(0.) ; 

    if(lmode & VALIDATE)
    {
        dt = validate_();
        LOG(LEVEL) << "VALIDATE time: " << dt ;
        if(times) times->add("validate", m_launch_count,  dt) ;
    }

    if(lmode & COMPILE)
    {
        dt = compile_();
        LOG(LEVEL) << "COMPILE time: " << dt ;
        if(times) times->add("compile", m_launch_count,  dt) ;
    }

    if(lmode & PRELAUNCH)
    {
        dt = launch_(entry, 0u, 0u );
        LOG(LEVEL) << "PRELAUNCH time: " << dt ;
        if(times) times->add("prelaunch", m_launch_count,  dt) ;
    }

    if(lmode & LAUNCH)
    {
        dt = m_llogpath ? launch_redirected_(entry, width, height ) : launch_(entry, width, height );
        LOG(LEVEL) << "LAUNCH time: " << dt  ;
        if(times) times->add("launch", m_launch_count,  dt) ;
    }

    m_launch_count += 1 ; 

    return dt ; 
}


double OContext::validate_()
{
    double t0, t1 ; 
    t0 = BTimeStamp::RealTime();

    m_context->validate(); 

    t1 = BTimeStamp::RealTime();
    return t1 - t0 ; 
}

double OContext::compile_()
{
    double t0, t1 ; 
    t0 = BTimeStamp::RealTime();

    m_context->compile(); 

    t1 = BTimeStamp::RealTime();
    return t1 - t0 ; 
}

double OContext::launch_(unsigned entry, unsigned width, unsigned height)
{
    double t0, t1 ; 
    t0 = BTimeStamp::RealTime();

    m_context->launch( entry, width, height ); 

    t1 = BTimeStamp::RealTime();
    return t1 - t0 ; 
}


double OContext::launch_redirected_(unsigned entry, unsigned width, unsigned height)
{
    assert( m_llogpath ) ;

    S_freopen_redirect sfr(stdout, m_llogpath );

    double dt = launch_( entry, width, height ) ;

    return dt ;  
}

/*

OContext::launch_redirected_ succeeds to write kernel rtPrintf 
logging to file BUT a subsequent same process "system" invokation 
of python has problems
indicating that the cleanup is not complete::

    Traceback (most recent call last):
      File "/Users/blyth/opticks/ana/tboolean.py", line 62, in <module>
        print ab
    IOError: [Errno 9] Bad file descriptor
    2017-12-13 15:33:13.436 INFO  [321569] [SSys::run@50] tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch   rc_raw : 256 rc : 1
    2017-12-13 15:33:13.436 WARN  [321569] [SSys::run@57] SSys::run FAILED with  cmd tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch  
    2017-12-13 15:33:13.436 INFO  [321569] [OpticksAna::run@79] OpticksAna::run anakey tboolean cmdline tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch   rc 1 rcmsg OpticksAna::run non-zero RC from ana script
    2017-12-13 15:33:13.436 FATAL [321569] [Opticks::dumpRC@186]  rc 1 rcmsg : OpticksAna::run non-zero RC from ana script
    2017-12-13 15:33:13.436 INFO  [321569] [SSys::WaitForInput@151] SSys::WaitForInput OpticksAna::run paused : hit RETURN to continue...

*/


/**
OContext::upload
------------------


**/

template <typename T>
void OContext::upload(optix::Buffer& buffer, NPY<T>* npy)
{
    unsigned long long numBytes = npy->getNumBytes(0) ;
    assert( sizeof(size_t) == sizeof(unsigned long long) ); 

    LOG(LEVEL) << " numBytes " << numBytes ;  

    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    bool verbose = ctrl("VERBOSE_MODE") || SSys::IsVERBOSE() ;

    if(ctrl(OpticksBufferControl::OPTIX_OUTPUT_ONLY_))
    { 
         LOG(error) 
             << "NOT PROCEEDING "
             << " name " << npy->getBufferName()
             << " as " << OpticksBufferControl::OPTIX_OUTPUT_ONLY_
             << " desc " << npy->description("skip-upload") 
             ;
     
    }
    else if(ctrl("UPLOAD_WITH_CUDA"))
    {
        if(verbose) LOG(LEVEL) << npy->description("UPLOAD_WITH_CUDA markDirty") ;

        void* d_ptr = NULL;
        rtBufferGetDevicePointer(buffer->get(), 0, &d_ptr);
        cudaMemcpy(d_ptr, npy->getBytes(), numBytes, cudaMemcpyHostToDevice);
        buffer->markDirty();
        if(verbose) LOG(LEVEL) << npy->description("UPLOAD_WITH_CUDA markDirty DONE") ;
    }
    else
    {
        if(verbose) LOG(LEVEL) << npy->description("standard OptiX UPLOAD") ;
        memcpy( buffer->map(), npy->getBytes(), numBytes );
        buffer->unmap(); 
    }
}


/**
OContext::download
--------------------

Downloading from GPU buffer to NPY array only proceeds 
for appropriate OpticksBufferControl settings associated 
with the named array.

**/

template <typename T>
void OContext::download(optix::Buffer& buffer, NPY<T>* npy)
{
    assert(npy);
    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    bool verbose = ctrl("VERBOSE_MODE") || SSys::IsVERBOSE() ;

    bool proceed = false ; 
    if(ctrl(OpticksBufferControl::OPTIX_INPUT_ONLY_))
    {
         proceed = false ; 
         LOG(error) 
             << "NOT PROCEEDING "
             << " name " << npy->getBufferName()
             << " as " << OpticksBufferControl::OPTIX_INPUT_ONLY_
             << " desc " << npy->description("skip-download") 
             ;
    }
    else if(ctrl(OpticksBufferControl::COMPUTE_MODE_))
    {
         proceed = true ; 
    }
    else if(ctrl(OpticksBufferControl::OPTIX_NON_INTEROP_))
    {   
         proceed = true ;
         LOG(info) << "PROCEED for " << npy->getBufferName() << " as " << OpticksBufferControl::OPTIX_NON_INTEROP_  ;
    }   
    else
    {
         proceed = false ; 
         LOG(fatal) << "NOT PROCEEDing for " << npy->getBufferName()   ;
    }

    
    if(proceed)
    {

        if(verbose)
             LOG(info) << " VERBOSE_MODE "  << " " << npy->description("download") ;

        void* ptr = buffer->map() ; 
        npy->read( ptr );
        buffer->unmap(); 
    }
    else
    {
        if(verbose)
             LOG(info)<< npy->description("download SKIPPED") ;

    }
}


void OContext::InitBufferNames(std::vector<std::string>& names)
{
    names.push_back("gensteps");
    names.push_back("seed");
    names.push_back("photon");
    names.push_back("source");
    names.push_back("record");
    names.push_back("sequence");
    names.push_back("debug");

    names.push_back("demo");
    names.push_back("axis");
}

void OContext::InitDebugBufferNames(std::vector<std::string>& names)
{
    names.push_back("record");
    names.push_back("sequence");
}


bool OContext::isAllowedBufferName(const char* name) const 
{
    return std::find( m_buffer_names.begin(), m_buffer_names.end() , name ) != m_buffer_names.end() ; 
}

bool OContext::isDebugBufferName(const char* name) const 
{
    return std::find( m_debug_buffer_names.begin(), m_debug_buffer_names.end() , name ) != m_debug_buffer_names.end() ; 
}





optix::Buffer OContext::createEmptyBufferF4() 
{
    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, 0);
    return emptyBuffer ;
}


/**
OContext::createBuffer
-----------------------

Workhorse, called for example from OEvent::createBuffers

For OpenGL visualized buffers the NPY array must have a valid bufferId 
indicating that the data was uploaded to an OpenGL buffer by Rdr::upload.
For buffers that are not visualized such as the "debug" buffer it is 
necessary for the OpticksBufferSpec/OpticksBufferControl tag of 
OPTIX_NON_INTEROP to be set to avoid assertions when running interactively.
See notes/issues/OKTest_CANNOT_createBufferFromGLBO_as_not_uploaded_name_debug.rst 

**/

template <typename T>
optix::Buffer OContext::createBuffer(NPY<T>* npy, const char* name)
{
    assert(npy);
    bool allowed_name = isAllowedBufferName(name); 
    if(!allowed_name) LOG(fatal) << " name " << name << " IS NOT ALLOWED " ; 
    assert(allowed_name);   

    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    bool compute = isCompute()  ; 

    LOG(LEVEL) 
        << std::setw(20) << name 
        << std::setw(20) << npy->getShapeString()
        << " mode : " << ( compute ? "COMPUTE " : "INTEROP " )
        << " BufferControl : " << ctrl.description(name)
        ;

    unsigned int type(0);
    bool noctrl = false ; 
    
    if(      ctrl("OPTIX_INPUT_OUTPUT") )  type = RT_BUFFER_INPUT_OUTPUT ;
    else if( ctrl("OPTIX_OUTPUT_ONLY")  )  type = RT_BUFFER_OUTPUT  ;
    else if( ctrl("OPTIX_INPUT_ONLY")   )  type = RT_BUFFER_INPUT  ;
    else                                   noctrl = true ; 
   
    if(noctrl) LOG(fatal) << "no buffer control for " << name << ctrl.description("") ;
    assert(!noctrl);
 
    if( ctrl("BUFFER_COPY_ON_DIRTY") )     type |= RT_BUFFER_COPY_ON_DIRTY ;
    // p170 of OptiX_600 optix-api 

    optix::Buffer buffer ; 

    if( compute )
    {
        buffer = m_context->createBuffer(type);
    }
    else if( ctrl("OPTIX_NON_INTEROP") )
    {
        buffer = m_context->createBuffer(type);
    }
    else
    {
        int buffer_id = npy ? npy->getBufferId() : -1 ;
        if(!(buffer_id > -1))
            LOG(fatal) 
                << "CANNOT createBufferFromGLBO as not uploaded  "
                << " name " << std::setw(20) << name
                << " buffer_id " << buffer_id 
                ; 
        assert(buffer_id > -1 );

        LOG(debug) 
            << "createBufferFromGLBO" 
            << " name " << std::setw(20) << name
            << " buffer_id " << buffer_id 
            ; 

        buffer = m_context->createBufferFromGLBO(type, buffer_id);
    } 

    configureBuffer<T>(buffer, npy, name );
    return buffer ; 
}


/**
OContext::determineBufferSize
-------------------------------

Mostly the size is the result of NPY::getNumQuads for RT_FORMAT_USER or
the seed buffer the size is ni*nj*nk the product of the first three dimensions.

**/

template <typename T>
unsigned OContext::determineBufferSize(NPY<T>* npy, const char* name)
{
    unsigned int ni = std::max(1u,npy->getShape(0));
    unsigned int nj = std::max(1u,npy->getShape(1));  
    unsigned int nk = std::max(1u,npy->getShape(2));  

    bool is_seed = strcmp(name, "seed")==0 ;

    RTformat format = getFormat(npy->getType(), is_seed);

    unsigned int size ; 

    if(format == RT_FORMAT_USER || is_seed)
    {
        size = ni*nj*nk ; 
    }
    else
    {
        size = npy->getNumQuads() ;  
    }
    return size ; 
}



template <typename T>
unsigned OContext::getBufferSize(NPY<T>* npy, const char* name)
{
    bool allowed_buffer_name = isAllowedBufferName(name); 
    if(!allowed_buffer_name)
    {
        LOG(fatal) << " name " << name << " IS NOT ALLOWED " ; 
        assert(0);   
    } 

    unsigned size = determineBufferSize( npy, name ); 

    bool is_debug_buffer_name = isDebugBufferName(name);
    bool is_production =  m_ok->isProduction() ; 
    bool is_forced_empty = is_production && is_debug_buffer_name ;     

    LOG(LEVEL) 
        << " name " << name 
        << " is_debug_buffer_name " << is_debug_buffer_name
        << " is_production " << is_production
        << " is_forced_empty " << is_forced_empty
        << " unforced size " << size
        << " npy " << npy->getShapeString()
        ;      

    return is_forced_empty ? 0 : size ; 
}





/**
OContext::configureBuffer
---------------------------

NB in interop mode, the OptiX buffer is just a reference to the 
OpenGL buffer object, however despite this the size
and format of the OptiX buffer still needs to be set as they control
the addressing of the buffer in the OptiX programs 

::

    79 rtBuffer<float4>               genstep_buffer;
    80 rtBuffer<float4>               photon_buffer;
    ..
    85 rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
    86 rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 

**/


template <typename T>
void OContext::configureBuffer(optix::Buffer& buffer, NPY<T>* npy, const char* name)
{
    bool is_seed = strcmp(name, "seed")==0 ;

    RTformat format = getFormat(npy->getType(), is_seed);
    buffer->setFormat(format);  // must set format, before can set ElementSize

    unsigned size = getBufferSize(npy, name);

    const char* label ; 
    if(     format == RT_FORMAT_USER) label = "USER";
    else if(is_seed)                  label = "SEED";
    else                              label = "QUAD";



    std::stringstream ss ; 
    ss 
       << std::setw(10) << name
       << std::setw(20) << npy->getShapeString()
       << " " << label 
       << " size " << size ; 
       ;
    std::string hdr = ss.str();

    if(format == RT_FORMAT_USER )
    {
        buffer->setElementSize(sizeof(T));
        LOG(LEVEL) << hdr
                  << " elementsize " << sizeof(T)
                  ;
    }
    else
    {
        LOG(LEVEL) << hdr ;
    }
    

    buffer->setSize(size); 

}

/**
OContext::resizeBuffer
-------------------------

Canonical usage is from OEvent::resizeBuffers

**/

template <typename T>
void OContext::resizeBuffer(optix::Buffer& buffer, NPY<T>* npy, const char* name)    // formerly static 
{
    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    bool verbose = ctrl("VERBOSE_MODE") ;

    unsigned size = getBufferSize(npy, name);
    buffer->setSize(size); 

    LOG(verbose ? info : LEVEL ) 
        << name 
        << " shape " << npy->getShapeString() 
        << " size " << size  
        ; 
}





RTformat OContext::getFormat(NPYBase::Type_t type, bool is_seed)
{
    RTformat format ; 
    switch(type)
    {
        case NPYBase::FLOAT:     format = RT_FORMAT_FLOAT4         ; break ; 
        case NPYBase::SHORT:     format = RT_FORMAT_SHORT4         ; break ; 
        case NPYBase::INT:       format = RT_FORMAT_INT4           ; break ; 
        case NPYBase::UINT:      format = RT_FORMAT_UNSIGNED_INT4  ; break ; 
        case NPYBase::CHAR:      format = RT_FORMAT_BYTE4          ; break ; 
        case NPYBase::UCHAR:     format = RT_FORMAT_UNSIGNED_BYTE4 ; break ; 
        case NPYBase::ULONGLONG: format = RT_FORMAT_USER           ; break ; 
        case NPYBase::DOUBLE:    format = RT_FORMAT_USER           ; break ; 
    }

    if(is_seed)
    {
         assert(type == NPYBase::UINT);
         format = RT_FORMAT_UNSIGNED_INT ;
         LOG(LEVEL) << "OContext::getFormat override format for seed " ; 
    }
    return format ; 
}


/**
OContext::snap
----------------

cu/pinhole_camera.cu::

    rtBuffer<uchar4, 2>  output_buffer;

**/


void OContext::snap(const char* path)
{
    if(m_ok->isNoSavePPM())
    {
        LOG(fatal) << " --nosaveppm " << path ; 
        return ;  
    }


    optix::Buffer output_buffer = m_context["output_buffer"]->getBuffer() ; 

    RTsize width, height, depth ;
    output_buffer->getSize(width, height, depth);

    bool yflip = true ; 
    LOG(LEVEL) 
         << " path " << path 
         << " width " << width
         << " height " << height
         << " depth " << depth
         << " yflip " << yflip
         ;   

    void* ptr = output_buffer->map() ; 

    int ncomp = 4 ;   
    SPPM::write(path,  (unsigned char*)ptr , width, height, ncomp, yflip );

    output_buffer->unmap(); 
}


void OContext::save(const char* path)
{
    optix::Buffer output_buffer = m_context["output_buffer"]->getBuffer() ;

    RTsize width, height, depth ;
    output_buffer->getSize(width, height, depth);

    LOG(info)
         << " width " << width
         << " width " << (int)width
         << " height " << height
         << " height " << (int)height
         << " depth " << depth
         ;

    NPY<unsigned char>* npy = NPY<unsigned char>::make(width, height, 4) ;
    npy->zero();

    void* ptr = output_buffer->map() ;
    npy->read( ptr );

    output_buffer->unmap();

    npy->save(path);
}







template OXRAP_API void OContext::upload<unsigned>(optix::Buffer&, NPY<unsigned>* );
template OXRAP_API void OContext::download<unsigned>(optix::Buffer&, NPY<unsigned>* );
template OXRAP_API void OContext::resizeBuffer<unsigned>(optix::Buffer&, NPY<unsigned>*, const char* );

template OXRAP_API void OContext::upload<float>(optix::Buffer&, NPY<float>* );
template OXRAP_API void OContext::download<float>(optix::Buffer&, NPY<float>* );
template OXRAP_API void OContext::resizeBuffer<float>(optix::Buffer&, NPY<float>*, const char* );

template OXRAP_API void OContext::upload<short>(optix::Buffer&, NPY<short>* );
template OXRAP_API void OContext::download<short>(optix::Buffer&, NPY<short>* );
template OXRAP_API void OContext::resizeBuffer<short>(optix::Buffer&, NPY<short>*, const char* );

template OXRAP_API void OContext::upload<unsigned long long>(optix::Buffer&, NPY<unsigned long long>* );
template OXRAP_API void OContext::download<unsigned long long>(optix::Buffer&, NPY<unsigned long long>* );
template OXRAP_API void OContext::resizeBuffer<unsigned long long>(optix::Buffer&, NPY<unsigned long long>*, const char* );

template OXRAP_API optix::Buffer OContext::createBuffer(NPY<float>*, const char* );
template OXRAP_API optix::Buffer OContext::createBuffer(NPY<short>*, const char* );
template OXRAP_API optix::Buffer OContext::createBuffer(NPY<unsigned long long>*, const char* );
template OXRAP_API optix::Buffer OContext::createBuffer(NPY<unsigned>*, const char* );


