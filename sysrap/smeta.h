#pragma once
/**
smeta.h
==========

BASH_SOURCE fails to export, so as workaround export SCRIPT envvar
to capture the script name into metadata

Users::

    [lo] A[blyth@localhost sysrap]$ opticks-fl smeta::Collect
    ./CSG/CSGFoundry.cc
    ./sysrap/SEvt.cc
    ./sysrap/tests/SEvt_AddEnvMeta_Test.cc
    ./sysrap/tests/smeta_test.cc
    ./sysrap/smeta.h
    ./sysrap/sreportdb.h
    ./u4/U4Recorder.cc


Principal metadata collection from SEvt static, so it runs immediately - as the executable loads libSysRap::

    NP* SEvt::Init_RUN_META() // static
    {
        NP* run_meta = NP::Make<float>(1);

        const char* label = "SEvt__Init_RUN_META" ;
        SProf::Add(label);

        bool stamp = true ;
        smeta::Collect(run_meta->meta, label, stamp );

        return run_meta ;
    }

    NP* SEvt::RUN_META = Init_RUN_META() ;


Subsequent entries within CSGOptiX::init as GPU setup done::

    [lo] A[blyth@localhost CSGOptiX]$ opticks-f SEvt::SetRunMeta
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMetaString("GPUMeta", gm.c_str() );  // set CUDA_VISIBLE_DEVICES to control
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMetaString("QSim__Switches", switches.c_str() );
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMetaString("C4Version", c4.c_str());
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMetaString("C4Version", "NOT-WITH_CUSTOM4" );
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMetaString("optixpath", optixpath );
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMetaString("optixpath_mtime_str", str.c_str() );
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMeta<int64_t>("optixpath_mtime", mtime );
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMeta<int64_t>("optixpath_age_secs", age_secs );
    ./CSGOptiX/CSGOptiX.cc:    SEvt::SetRunMeta<int64_t>("optixpath_age_days", age_days );
    ./sysrap/SEvt.cc:void SEvt::SetRunMeta(const char* k, T v )
    ./sysrap/SEvt.cc:template void SEvt::SetRunMeta<int>(      const char*, int  );
    ./sysrap/SEvt.cc:template void SEvt::SetRunMeta<uint64_t>( const char*, uint64_t  );
    ./sysrap/SEvt.cc:template void SEvt::SetRunMeta<int64_t>(  const char*, int64_t  );
    ./sysrap/SEvt.cc:template void SEvt::SetRunMeta<unsigned>( const char*, unsigned  );
    ./sysrap/SEvt.cc:template void SEvt::SetRunMeta<float>(    const char*, float  );
    ./sysrap/SEvt.cc:template void SEvt::SetRunMeta<double>(   const char*, double  );
    ./sysrap/SEvt.cc:template void SEvt::SetRunMeta<std::string>( const char*, std::string  );
    ./sysrap/SEvt.cc:void SEvt::SetRunMetaString(const char* k, const char* v ) // static
    ./u4/U4Recorder.cc:    SEvt::SetRunMeta<int>("FAKES_SKIP", int(FAKES_SKIP) );
    [lo] A[blyth@localhost opticks]$






**/

#include "OKConf.h"

#ifdef WITH_CUDA
//#include "OKConf_CUDART.h"
//    DO NOT DO THIS HERE - DO IT FROM CSGOptiX TO AVOID ACCIDENTAL DEPENDENCY
//    ON CUDART FROM EVERY TEST THAT ENDS UP INCLUDING smeta.H

#endif

#include "sproc.h"
#include "ssys.h"
#include "sstamp.h"
#include "NP.hh"

struct smeta
{
    static constexpr const char* PREFIX = "OPTICKS_" ;
    static constexpr const char* VARS = R"(
CUDA_VISIBLE_DEVICES
HOME
USER
SCRIPT
SCRIPT_ARG
PWD
CMDLINE
CHECK
LAYOUT
TEST
TESTSCRIPT      # see oj/TEST.sh
TESTSCRIPT_ARG  # see oj/TEST.sh
VERSION
GEOM
EXECUTABLE
COMMANDLINE
DIRECTORY
${GEOM}_GEOMList
)" ;

    static constexpr const char* CI_VARS = R"(
CI_PIPELINE_ID      # The primary key/unique identifier for the overall run execution.
CI_JOB_ID	        # Each job within pipeline has unique CI_JOB_ID.
CI_PIPELINE_SOURCE	# Why the test ran: schedule, push, web
)" ;


// HIGHER ORDER KEYS WITH TOKENS ARE HANDLED IN ssys::_getenv
    static void Collect(std::string& meta, const char* source=nullptr, bool stamp=false) ;
    static void CollectEnv(std::string& meta ) ;
};

/**
smeta::Collect
----------------

This is used for example to populate (SEvt)sevt.meta by:

1. G4CXOpticks::init_SEvt for SEvt::EGPU meta
2. U4Recorder::init_SEvt for SEvt::ECPU meta
3. CSGFoundry::init for the CSGFoundry::meta

**/

inline void smeta::Collect(std::string& meta, const char* source, bool stamp )
{
    if(stamp)
    {
        int64_t t = sstamp::Now();
        std::string tf = sstamp::Format(t) ;
        NP::SetMeta<int64_t>(meta, "InitTimestamp", t);  // formerly _init_stamp
        NP::SetMeta<std::string>(meta, "InitTimestampFmt", tf);  // formerly _init_stamp_Fmt
    }

    if(source) NP::SetMeta<std::string>(meta, "source", source );
    NP::SetMeta<std::string>(meta, "ExecutableName", sproc::ExecutableName() );   // formerly "creator"
    NP::SetMeta<std::string>(meta, "OpticksGitHash", OKConf::OpticksGitHash() );
    NP::SetMeta<std::string>(meta, "uname", ssys::uname("-a"));


#ifdef WITH_CUDA
    NP::SetMeta<int>(meta, "OptiXVersion", OKConf::OptiXVersionInteger() );
    NP::SetMeta<int>(meta, "ComputeCapability", OKConf::ComputeCapabilityInteger() );
    NP::SetMeta<int>(meta, "CUDAVersion", OKConf::CUDAVersionInteger() );

    // NP::SetMeta<int>(meta, "CUDADriver", OKConf_CUDART::CUDADriverInteger() );
    //    DONT DO THIS HERE - DO IT FROM CSGOptiX.cc TO AVOID ACCIDENTAL DEPENDENCY
    //    ON CUDART FROM EVERY TEST THAT ENDS UP INCLUDING smeta.h

    NP::SetMeta<std::string>(meta, "NvidiaDriverVersion", OKConf::NvidiaDriverVersion() );
#endif
    NP::SetMeta<int>(meta, "Geant4Version", OKConf::Geant4VersionInteger() );
    NP::SetMeta<int>(meta, "OpticksVersion", OKConf::OpticksVersionInteger() );

    CollectEnv(meta);
}

/**
smeta::CollectEnv
-------------------

TODO: collect all envvars starting with prefix OPTICKS_

**/

inline void smeta::CollectEnv(std::string& meta )
{
    typedef std::pair<std::string, std::string> KV ;

    std::vector<KV> kvs0 ;
    ssys::getenv_(kvs0, VARS);
    NP::SetMetaKV(meta, kvs0);

    std::vector<KV> kvs1 ;
    ssys::getenv_(kvs1, CI_VARS);
    NP::SetMetaKV(meta, kvs1);

    std::vector<KV> kvs2 ;
    ssys::getenv_with_prefix(kvs2, PREFIX);
    NP::SetMetaKV(meta, kvs2);
}



