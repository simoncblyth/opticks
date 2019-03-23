#pragma once
/**
OPropertyLib
===============

Base class of property libs providing buffer upload and dumping.
Subclasses include:

* :doc:`OBndLib`
* :doc:`OSourceLib`
* :doc:`OScintillatorLib`

**/


#include "OXPPNS.hh"
template <typename T> class NPY ;

#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OPropertyLib  {
    public:
        OPropertyLib(optix::Context& ctx, const char* name);
    public:
    protected:
        void upload(optix::Buffer& optixBuffer, NPY<float>* buffer);
        void dumpVals( float* vals, unsigned int n);
    protected:
        optix::Context       m_context ; 
        const char*          m_name ; 

};


