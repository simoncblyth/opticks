
#include <cstring>
#include "SArgs.hh"

#include "BOpticks.hh"
#include "BOpticksKey.hh"
#include "BOpticksResource.hh"

#include "PLOG.hh"

BOpticks::BOpticks(int argc, char** argv, const char* argforced )
    :    
    m_firstarg( argc > 1 ? argv[1] : nullptr ), 
    m_sargs(new SArgs(argc, argv, argforced)), 
    m_argc(m_sargs->argc),
    m_argv(m_sargs->argv),
    m_envkey(m_sargs->hasArg("--envkey") ? BOpticksKey::SetKey(nullptr) : false),
    m_testgeo(false),
    m_resource(new BOpticksResource(m_testgeo)),
    m_error(0)
{
    if(m_resource->hasKey())
    {
        m_resource->setupViaKey();  
    }
    else
    {
        m_error = 1 ;  
    }
}


const char* BOpticks::getPath(const char* rela, const char* relb, const char* relc ) const
{
    return m_resource->makeIdPathPath(rela, relb, relc );
}

int BOpticks::getError() const { 
    if( m_error > 0 ) LOG(fatal) << " MISSING OPTICKS_KEY " ; 
    return m_error ; 
}

const char* BOpticks::getFirstArg(const char* fallback) const 
{
    return m_firstarg ? m_firstarg : fallback ; 
}

const char* BOpticks::getArg(int n, const char* fallback) const   // argforce makes this problematic
{
    return n < m_argc ? m_argv[n] : fallback ; 
}
