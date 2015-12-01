#include "GSource.hh"
#include "GPropertyLib.hh"

#include <cassert>



GSource::GSource(GSource* other) : GPropertyMap<float>(other)
{
}

GSource::GSource(const char* name, unsigned int index) : GPropertyMap<float>(name, index, "source")
{
   init();
}

GSource::~GSource()
{
}

void GSource::Summary(const char* msg )
{
    GPropertyMap<float>::Summary(msg);
}


void GSource::init()
{
    GDomain<float>* sd = GPropertyLib::getDefaultDomain();
    setStandardDomain(sd);
}


GSource* GSource::make_blackbody_source(const char* name, unsigned int index, float kelvin)
{
    GSource* source = new GSource(name, index);

    GProperty<float>* radiance = GProperty<float>::planck_spectral_radiance( source->getStandardDomain(), 6500.f );

    assert(radiance) ;

    source->addProperty("radiance", radiance );

    return source ; 
}

