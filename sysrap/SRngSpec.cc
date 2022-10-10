#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>

#include "SPath.hh"
#include "SRngSpec.hh"
#include "SEvt.hh"
#include "SLOG.hh"

const plog::Severity SRngSpec::LEVEL = SLOG::EnvLevel("SRngSpec", "DEBUG"); 

const char* SRngSpec::DefaultRngDir()  // static 
{
    return SPath::GetHomePath(".opticks/rngcache/RNG") ;
}

const char* SRngSpec::CURANDStatePath(const char* rngdir, unsigned rngmax, unsigned long long seed, unsigned long long offset) // static
{
    char buf[256];

    const char* oldpfx = "cuRANDWrapper" ; 
    const char* newpfx = "QCurandState" ; 
    const char* pfx = SEvt::Exists() ? newpfx : oldpfx ;  // rough heuristic to distinguish between old and new Opticks

    snprintf(buf, 256, "%s/%s_%u_%llu_%llu.bin", 
                 rngdir ? rngdir : DefaultRngDir(),
                 pfx, 
                 rngmax,
                 seed,
                 offset); 

    return strdup(buf) ; 
}

SRngSpec::SRngSpec(unsigned rngmax, unsigned long long seed, unsigned long long offset)
    :
    m_rngmax(rngmax),
    m_seed(seed), 
    m_offset(offset)
{
}

const char* SRngSpec::getCURANDStatePath(const char* rngdir) const 
{
    return CURANDStatePath(rngdir, m_rngmax, m_seed, m_offset ); 
}

bool SRngSpec::isValid(const char* rngdir) const 
{
    const char* path = getCURANDStatePath(rngdir); 
    bool readable = SPath::IsReadable(path); 
    LOG(LEVEL)
        << " path " << path 
        << " readable " << readable 
        ;
    return readable ;  
}

std::string SRngSpec::desc() const 
{
    std::stringstream ss ; 
    ss << "SRngSpec"
       << " rngmax " << m_rngmax
       << " seed " << m_seed
       << " offset " << m_offset
       ;
    return ss.str(); 
}


