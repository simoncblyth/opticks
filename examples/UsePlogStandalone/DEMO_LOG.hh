
#pragma once


// these macros are used from the main, so plog::get returns the logger from the main 
// and hands it to the logger from the shared lib 
//
// HMM : feels wring way around, shouldny the shared lib loggers be added as appenders to the main logger ?

#define DEMO_LOG_ {         DEMO_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), nullptr ); } 
#define _DEMO_LOG( IDX ) {  DEMO_LOG::Init<IDX>( info, plog::get<IDX>(), nullptr ) ; }


#define DEMO_API  __attribute__ ((visibility ("default")))

#include "plog/Severity.h"
#include "plog/Appenders/IAppender.h"

struct DEMO_API DEMO_LOG 
{
    static void Initialize(plog::Severity level, plog::IAppender* app );
};

