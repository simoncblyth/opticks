#pragma once
#include <cstddef>
#include <cstring>
#include <optixu/optixpp_namespace.h>

class OAccel {
    public:
        OAccel(optix::Acceleration accel, const char* path);
    public:
        void import();
        void save();
    private:
        char* read(const char* path);

    private:
        optix::Acceleration    m_accel ; 
        const char*            m_path ;
        unsigned long long int m_size ;
        bool                   m_loaded ; 

};


inline OAccel::OAccel(optix::Acceleration accel, const char* path) :
   m_accel(accel),
   m_path( path ? strdup(path) : NULL ),
   m_size( 0ull ),
   m_loaded(false)
{
}


