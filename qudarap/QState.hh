#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>

#include "QUDARAP_API_EXPORT.hh"
#include "vector_functions.h"
#include "scuda.h"
#include "squad.h"
#include "qstate.h"
#include "NP.hh"

struct QUDARAP_API QState
{
    static qstate Make();     
    static void Convert(quad6& ss, const qstate& s); 
    static void Convert(qstate& s, const quad6&  ss); 
    static void Read( qstate& s,   const NP* a ); 
    static void Save(const qstate& s, const char* dir, const char* name ); 
    static void Load(      qstate& s, const char* dir, const char* name ); 
    static void Save(const qstate& s, const char* path ); 
    static void Load(      qstate& s, const char* path ); 
    static std::string Desc( const qstate& s ) ; 
};

inline qstate QState::Make()
{
    float m1_refractive_index = qenvfloat("M1_REFRACTIVE_INDEX", "1" ) ; 
    float m1_absorption_length = 1000.f ; 
    float m1_scattering_length = 1000.f ; 
    float m1_reemission_prob = 0.f ; 
    float m1_group_velocity = 300.f ; 

    float m2_refractive_index = qenvfloat("M2_REFRACTIVE_INDEX", "1.5" ) ; 
    float m2_absorption_length = 1000.f ; 
    float m2_scattering_length = 1000.f ; 
    float m2_reemission_prob = 0.f ; 

    float su_detect = 0.f ; 
    float su_absorb = 0.f ; 
    float su_reflect_specular = 0.f ; 
    float su_reflect_diffuse = 0.f ; 

    qstate s ; 
    s.material1 = make_float4( m1_refractive_index, m1_absorption_length, m1_scattering_length, m1_reemission_prob ); 
    s.material2 = make_float4( m2_refractive_index, m2_absorption_length, m2_scattering_length, m2_reemission_prob );  
    s.m1group2  = make_float4( m1_group_velocity, 0.f, 0.f, 0.f ); 
    s.surface   = make_float4( su_detect, su_absorb, su_reflect_specular, su_reflect_diffuse ); 
    s.optical   = make_uint4( 0u, 0u, 0u, 0u );  // x/y/z/w index/type/finish/value  
    s.index     = make_uint4( 0u, 0u, 0u, 0u );  // indices of m1/m2/surf/sensor
    return s ; 
}


inline void QState::Convert(quad6& ss, const qstate& s)
{
    ss.q0.f = s.material1 ; 
    ss.q1.f = s.material2 ; 
    ss.q2.f = s.m1group2 ; 
    ss.q3.f = s.surface ;
    ss.q4.u = s.optical ; 
    ss.q5.u = s.index ; 
}

inline void QState::Convert(qstate& s, const quad6& ss)
{
    s.material1 = ss.q0.f ; 
    s.material2 = ss.q1.f ; 
    s.m1group2  = ss.q2.f ; 
    s.surface   = ss.q3.f ; 
    s.optical   = ss.q4.u ; 
    s.index     = ss.q5.u ; 
}

inline void QState::Read( qstate& s, const NP* a )
{
    assert( a->has_shape(1,6,4) ); 
    quad6 ss ; 
    a->write((float*)&ss.q0.f.x); 
    Convert(s, ss); 
}

inline void QState::Save(const qstate& s, const char* dir, const char* name )
{
    quad6 ss ; 
    Convert(ss, s); 
    NP::Write( dir, name, (float*)&ss.q0.f.x,  1, 6, 4 ); 
}

inline void QState::Save(const qstate& s, const char* path )
{
    quad6 ss ; 
    Convert(ss, s); 
    NP::Write( path, (float*)&ss.q0.f.x,  1, 6, 4 ); 
}

inline void QState::Load(      qstate& s, const char* dir, const char* name )
{
    NP* a = NP::Load(dir, name); 
    Read(s, a ); 
    delete a ; 
}

inline void QState::Load(qstate& s, const char* path )
{
    NP* a = NP::Load(path); 
    Read(s, a ); 
    delete a ; 
}

inline std::string QState::Desc( const qstate& s ) 
{
    std::stringstream ss ; 
    ss << "QState::Desc" << std::endl
       << std::setw(10) << "material1 " <<  s.material1  << std::endl
       << std::setw(10) << "material2 " <<  s.material2  << std::endl
       << std::setw(10) << "m1group2  " <<  s.m1group2   << std::endl
       << std::setw(10) << "surface   " <<  s.surface    << std::endl
       << std::setw(10) << "optical   " <<  s.optical    << std::endl
       ;
    std::string repr = ss.str(); 
    return repr ; 
}


