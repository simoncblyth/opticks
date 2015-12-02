#pragma once

#include <vector>
class GCache ; 
class GItemList ;
class GSource ; 
template <typename T> class GPropertyMap ;

#include "GPropertyLib.hh"

class GSourceLib : public GPropertyLib {
    public:
        static const unsigned int icdf_length ; 
        static const char* radiance_ ; 
        void save();
        static GSourceLib* load(GCache* cache);
    public:
        GSourceLib(GCache* cache);
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
       GItemList*  createNames();
    private:
        void init();
    public:
        GProperty<float>* constructSourceCDF(GPropertyMap<float>* pmap);
        GProperty<float>* constructInvertedSourceCDF(GPropertyMap<float>* pmap);
    private:
        std::vector<GPropertyMap<float>*> m_source ; 

};

inline GSourceLib::GSourceLib( GCache* cache) 
    :
    GPropertyLib(cache, "GSourceLib")
{
    init();
}

inline unsigned int GSourceLib::getNumSources()
{
    return m_source.size();
}


