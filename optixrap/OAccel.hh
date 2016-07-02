#pragma once

#include "OXPPNS.hh"

#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OAccel {
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


