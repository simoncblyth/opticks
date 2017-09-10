#pragma once

class SLog ; 

// okc-
class Composition ; 

// okg-
class OpticksHub ;

// optixrap-
class OContext ;
class OTracer ;

//opop-
class OpEngine ; 

template <typename T> class NPY ;


#include "OKOP_API_EXPORT.hh"

#include "SRenderer.hh"

class OKOP_API OpTracer : public SRenderer {
    public:
       OpTracer(OpEngine* ope, OpticksHub* hub, bool immediate);
    public:
       void prepareTracer();
       void render();     // fulfils SRenderer protocol
       void snap();
    private:
       void init();
    private:
       SLog*            m_log ; 
       OpEngine*        m_ope ; 
       OpticksHub*      m_hub ; 
       bool             m_immediate ; 

       OContext*        m_ocontext ; 
       Composition*     m_composition ; 
       OTracer*         m_otracer ;

   //    NPY<unsigned char>*   m_npy ; 

};


