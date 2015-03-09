#include "GSubstanceLib.hh"
#include "GSubstance.hh"
#include "GPropertyMap.hh"

#include "assert.h"
#include "stdio.h"
#include "limits.h"

GSubstanceLib::GSubstanceLib()
{
}

GSubstanceLib::~GSubstanceLib()
{
}

void GSubstanceLib::Summary(const char* msg)
{
    printf("%s\n", msg );
    GSubstance* substance = NULL ; 

    char buf[128];
    for(unsigned int i=0 ; i<g_registry.size() ; i++ )
    {
         substance = g_registry[i] ;
         snprintf(buf, 128, "%s substance index %u ", msg, i );
         substance->Summary(buf);
    } 
}




GSubstance* GSubstanceLib::get(GPropertyMap* imaterial, GPropertyMap* isurface, GPropertyMap* osurface )
{ 
    printf("GSubstanceLib::get imaterial %p isurface %p osurface %p \n", imaterial, isurface, osurface );

    GSubstance* tmp = new GSubstance(imaterial, isurface, osurface);

    unsigned int index = UINT_MAX ;
    for(unsigned int i=0 ; i<g_registry.size() ; i++ )
    {
         if(g_registry[i]->matches(tmp)) index = i ;
    } 

    if(index == UINT_MAX )
    {
        g_registry.push_back(tmp);
        index = g_registry.size() ;
    }
    else
    {
        delete tmp ; 
    } 

    printf("GSubstanceLib::get index %u \n", index); 
    return g_registry[index] ; 
}


