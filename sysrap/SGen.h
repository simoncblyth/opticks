#pragma once
/**
SGen.h
==============

Used from SGLFW_Gen.h

**/

#include "NP.hh"

#include "spath.h"
#include "scerenkov.h"
#include "sstr.h"
#include "scuda.h"

struct SGen
{
    static constexpr const char* NAME = "genstep.npy" ;
    static constexpr const char* RPOS_SPEC = "4,GL_FLOAT,GL_FALSE,96,16,false";   // 6*4*4 = 96, 1*4*4 = 16
    static constexpr const char* RDEL_SPEC = "4,GL_FLOAT,GL_FALSE,96,32,false";   // 6*4*4 = 96, 2*4*4 = 32
    static constexpr const char* _level = "SGen__level" ;

    int level ;
    NP* genstep ;
    int genstep_first ;
    int genstep_count ;

    float4 mn = {} ;
    float4 mx = {} ;
    float4 ce = {} ;

    static NP*   LoadArray( const char* _fold, const char* _slice );
    static SGen* Load(      const char* _fold, const char* _slice=nullptr );

    SGen(NP* genstep);
    void init() ;

    const float* get_mn() const ;
    const float* get_mx() const ;
    const float* get_ce() const ;

    const float get_t0() const ;
    const float get_t1() const ;

    std::string desc() const ;
};

/**
SGen::LoadArray
-------------------


**/

inline NP* SGen::LoadArray(const char* _fold, const char* _slice )
{
    const char* path = spath::Resolve(_fold, NAME);

    bool looks_unresolved = spath::LooksUnresolved(path, _fold);
    if(looks_unresolved)
    {
        std::cout
            << "SGen::LoadArray"
            << " FAILED : DUE TO MISSING ENVVAR\n"
            << " _fold [" << ( _fold ? _fold : "-" ) << "]\n"
            << " path ["  << (  path ?  path : "-" ) << "]\n"
            << " looks_unresolved " << ( looks_unresolved ? "YES" : "NO " )
            << "\n"
            ;
        return nullptr ;
    }


    NP* a = nullptr ;

    if( _slice == nullptr )
    {
        a = NP::Load(path);
        a->set_meta<std::string>("SGen__LoadArray", "NP::Load");
    }
    else if(NP::LooksLikeWhereSelection(_slice))
    {
        a = NP::LoadThenSlice<float>(path, _slice);
        a->set_meta<std::string>("SGen__LoadArray_METHOD", "NP::LoadThenSlice");
        a->set_meta<std::string>("SGen__LoadArray_SLICE", _slice );
    }
    else
    {
        a = NP::LoadSlice(path, _slice);
        a->set_meta<std::string>("SGen__LoadArray_METHOD", "NP::LoadSlice");
        a->set_meta<std::string>("SGen__LoadArray_SLICE", _slice );
    }
    return a ;
}


inline SGen* SGen::Load(const char* _fold, const char* _slice )
{
    NP* _genstep = LoadArray(_fold, _slice);
    return _genstep ? new SGen(_genstep) : nullptr ;
}


inline SGen::SGen(NP* _genstep)
    :
    level(ssys::getenvint(_level,0)),
    genstep(_genstep),
    genstep_first(0),
    genstep_count(0)
{
    init();
}

/**
SGen::init
-------------------

Expected shape of genstep array like (10000, 6, 4 )

**/


inline void SGen::init()
{
    assert(genstep->shape.size() == 3);
    genstep_count = genstep->shape[0];

    // scerenkov and sscint first four quads have consistent form
    scerenkov::MinMaxPost(&mn.x, &mx.x, genstep );

    genstep->set_meta<float>("x0", mn.x );
    genstep->set_meta<float>("x1", mx.x );

    genstep->set_meta<float>("y0", mn.y );
    genstep->set_meta<float>("y1", mx.y );

    genstep->set_meta<float>("z0", mn.z );
    genstep->set_meta<float>("z1", mx.z );

    genstep->set_meta<float>("t0", mn.w );
    genstep->set_meta<float>("t1", mx.w );

    ce = scuda::center_extent( mn, mx );

    genstep->set_meta<float>("cx", ce.x );
    genstep->set_meta<float>("cy", ce.y );
    genstep->set_meta<float>("cz", ce.z );
    genstep->set_meta<float>("ce", ce.w );

    if(level > 0 ) std::cout
        << "[SGen::init\n"
        << desc()
        << "]SGen::init\n"
        ;
}



inline const float* SGen::get_mn() const
{
    return &mn.x ;
}
inline const float* SGen::get_mx() const
{
    return &mx.x ;
}
inline const float* SGen::get_ce() const
{
    return &ce.x ;
}

inline const float SGen::get_t0() const
{
    return mn.w ;
}
inline const float SGen::get_t1() const
{
    return mx.w ;
}


inline std::string SGen::desc() const
{
    const char* lpath = genstep ? genstep->lpath.c_str() : nullptr ;

    std::stringstream ss ;
    ss
        << "[SGen.desc\n"
        << " [" << _level << "] level " << level << "\n"
        << " lpath [" << ( lpath ? lpath : "-" ) << "]\n"
        << std::setw(20) << " mn " << mn
        << std::endl
        << std::setw(20) << " mx " << mx
        << std::endl
        << std::setw(20) << " ce " << ce
        << std::endl
        << std::setw(20) << " genstep.sstr " << genstep->sstr()
        << std::endl
        << std::setw(20) << " genstep_first " << genstep_first
        << std::endl
        << std::setw(20) << " genstep_count " << genstep_count
        << std::endl
        << ( genstep ? genstep->descMeta() : "-" )
        << std::endl
        << "]SGen.desc\n"
        ;
    std::string str = ss.str() ;
    return str ;
}

