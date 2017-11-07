#pragma once

#include <vector>
class NMeta ; 
class Opticks ; 
class GItemList ;
class GSource ; 
template <typename T> class GPropertyMap ;

#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GSourceLib : public GPropertyLib {
    public:
        static const unsigned int icdf_length ; 
        static const char* radiance_ ; 
        void save();
        static GSourceLib* load(Opticks* cache);
    public:
        GSourceLib(Opticks* cache);
    public:
        void add(GSource* source);
        unsigned int getNumSources();
    public:
        void generateBlackBodySample(unsigned int n=500000);
    public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
       void import();
       void sort();
       NPY<float>* createBuffer();
       NMeta*      createMeta();
       GItemList*  createNames();
    private:
        void init();
    public:
        GProperty<float>* constructSourceCDF(GPropertyMap<float>* pmap);
        GProperty<float>* constructInvertedSourceCDF(GPropertyMap<float>* pmap);
    private:
        std::vector<GPropertyMap<float>*> m_source ; 

};

#include "GGEO_TAIL.hh"

