#pragma once

#include "plog/Severity.h"
#include <vector>
#include <string>

template <typename T> class NPY ;
struct GPt ; 
class GItemList ; 

/**
GPts
======= 

Canonical m_pts instances are residents of GMergedMesh and
are instanciated by GMergedMesh::GMergedMesh with GPts::Make.

Motivation for GPts is to allow postcache deferred creation of 
merged GParts instances.  This capability is needed in order
to reconcile the different Opticks/Geant4 requirements 
regarding balanced/unbalanced CSG trees, see notes/issues/x016.rst

Can think of GPts as gathering and persisting the arguments needed 
for deferred GParts creation and merging. 

**/


#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GPts { 
    public: 
        static const plog::Severity LEVEL ; 
        static GPts* Make(); 
        static GPts* Load(const char* path); 
        static const char* GPTS_LIST ; 
    private:
        template<typename T> static NPY<T>* LoadBuffer(const char* dir, const char* tag); 
        static const char* BufferName(const char* tag) ;
    public: 
        GPts(NPY<int>* pts, NPY<float>* plc, GItemList* specs); 
        void export_(); 
        void import(); 
    public: 
        unsigned     getNumPt() const ; 
        const GPt*   getPt(unsigned i) const ; 
        void         add(GPt*  other);
        void         dump(const char* msg="GPts::dump") const ;  
        void         save(const char* path); 
        std::string  brief() const ; 
    private:
        std::vector<GPt*>  m_pts ; 
    private:
        NPY<int>*          m_ipt_buffer ; 
        NPY<float>*        m_plc_buffer ; 
        GItemList*         m_specs ; 
 
};

#include "GGEO_TAIL.hh"

