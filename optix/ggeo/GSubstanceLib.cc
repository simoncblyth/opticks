#include "GSubstanceLib.hh"
#include "GSubstanceLibMetadata.hh"
#include "GSubstance.hh"
#include "GPropertyMap.hh"
#include "GBuffer.hh"
#include "GEnums.hh"

#include <sstream>
#include "assert.h"
#include "stdio.h"
#include "limits.h"


float        GSubstanceLib::DOMAIN_LOW  = 60.f ; 
float        GSubstanceLib::DOMAIN_HIGH = 810.f ; 
float        GSubstanceLib::DOMAIN_STEP = 20.f ; 
unsigned int GSubstanceLib::DOMAIN_LENGTH = 39  ; 


GSubstanceLib::GSubstanceLib() : m_defaults(NULL), m_meta(NULL)
{
    // chroma/chroma/geometry.py
    //
    //    In [4]: np.arange(60., 810., 20.).astype(np.float32)
    //    Out[4]: 
    //    array([  60.,   80.,  100.,  120.,  140.,  160.,  180.,  200.,  220.,
    //            240.,  260.,  280.,  300.,  320.,  340.,  360.,  380.,  400.,
    //            420.,  440.,  460.,  480.,  500.,  520.,  540.,  560.,  580.,
    //            600.,  620.,  640.,  660.,  680.,  700.,  720.,  740.,  760.,
    //            780.,  800.], dtype=float32)
    //
    //    In [5]: len(np.arange(60., 810., 20.).astype(np.float32))
    //    Out[5]: 38
    //
    //    TODO: get standard domain to match python OR maybe move to 32 or 64 slots ?
    //

    setKeyMap(NULL);
    GDomain<double>* domain = new GDomain<double>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
    setStandardDomain( domain );


    GPropertyMap* defaults = new GPropertyMap("defaults", UINT_MAX, "defaults");
    defaults->setStandardDomain(getStandardDomain());

    // TODO: find way to avoid detector specific keys and default values being hardcoded ? 
    // or at least move up to a higher level of the code 

    defineDefaults(defaults);
    setDefaults(defaults);

    m_ramp = GProperty<double>::ramp( domain->getLow(), domain->getStep(), domain->getValues(), domain->getLength() );
    //m_ramp->Summary("GSubstanceLib::GSubstanceLib ramp", 20 );

}

GSubstanceLib::~GSubstanceLib()
{
}

unsigned int GSubstanceLib::getNumSubstances()
{
   return m_keys.size();
}

GProperty<double>* GSubstanceLib::getRamp()
{
   return m_ramp ;
}

void GSubstanceLib::setDefaults(GPropertyMap* defaults)
{
    m_defaults = defaults ;
}
GPropertyMap* GSubstanceLib::getDefaults()
{
    return m_defaults ;
}

void GSubstanceLib::setStandardDomain(GDomain<double>* standard_domain)
{
    m_standard_domain = standard_domain ; 

    assert(getStandardDomainLength() == DOMAIN_LENGTH );
    assert(m_standard_domain->getLow()  == DOMAIN_LOW );
    assert(m_standard_domain->getHigh() == DOMAIN_HIGH );
    assert(m_standard_domain->getStep() == DOMAIN_STEP );

}
GDomain<double>* GSubstanceLib::getStandardDomain()
{
    return m_standard_domain ;
}

unsigned int GSubstanceLib::getStandardDomainLength()
{
    return m_standard_domain ? m_standard_domain->getLength() : 0 ;
}



GProperty<double>* GSubstanceLib::getDefaultProperty(const char* name)
{
    return m_defaults ? m_defaults->getProperty(name) : NULL ;
}

GSubstance* GSubstanceLib::getSubstance(unsigned int index)
{
    GSubstance* substance = NULL ;
    if(index < m_keys.size())
    {
        std::string key = m_keys[index] ;  
        substance = m_registry[key];
        assert(substance->getIndex() == index );
    }
    return substance ; 
}


void GSubstanceLib::Summary(const char* msg)
{
    printf("%s\n", msg );
    char buf[128];
    for(unsigned int i=0 ; i < getNumSubstances() ; i++)
    {
         GSubstance* substance = getSubstance(i);
         snprintf(buf, 128, "%s substance ", msg );
         substance->Summary(buf);
    } 
}



GSubstance* GSubstanceLib::get(GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface )
{ 
    //printf("GSubstanceLib::get imaterial %p omaterial %p isurface %p osurface %p \n", imaterial, omaterial, isurface, osurface );

    GSubstance* tmp = new GSubstance(imaterial, omaterial, isurface, osurface);
    std::string key = tmp->digest();

    if(m_registry.count(key) == 0) // not yet registered, identity based on the digest 
    { 
        tmp->setIndex(m_keys.size());
        m_keys.push_back(key);  // for simple ordering  
        m_registry[key] = tmp ; 
    }
    else
    {
        delete tmp ; 
    } 

    GSubstance* substance = m_registry[key] ;
    //printf("GSubstanceLib::get key %s index %u \n", key.c_str(), substance->getIndex()); 
    return substance ; 
}




const char* GSubstanceLib::inner = "inner_" ;
const char* GSubstanceLib::outer = "outer_" ;

const char* GSubstanceLib::refractive_index = "refractive_index" ;
const char* GSubstanceLib::absorption_length = "absorption_length" ;
const char* GSubstanceLib::scattering_length = "scattering_length" ;
const char* GSubstanceLib::reemission_prob = "reemission_prob" ;

const char* GSubstanceLib::detect = "detect" ;
const char* GSubstanceLib::absorb = "absorb" ;
const char* GSubstanceLib::reflect_specular = "reflect_specular" ;
const char* GSubstanceLib::reflect_diffuse  = "reflect_diffuse" ;



const char* GSubstanceLib::keymap = 
"refractive_index:RINDEX,"
"absorption_length:ABSLENGTH,"
"scattering_length:RAYLEIGH,"
"reemission_prob:REEMISSIONPROB," 
"detect:EFFICIENCY," 
"absorb:DUMMY," 
"reflect_specular:REFLECTIVITY," 
"reflect_diffuse:REFLECTIVITY," 
;
// hmm how to convey specular/diffuse


void GSubstanceLib::setKeyMap(const char* spec)
{
    m_keymap.clear();
    const char* kmap = spec ? spec : keymap ; 

    char delim = ',' ;
    std::istringstream f(kmap);
    std::string s;
    while (getline(f, s, delim)) 
    {
        std::size_t colon = s.find(":");
        if(colon == std::string::npos)
        {
            printf("GSubstanceLib::setKeyMap SKIPPING ENTRY WITHOUT COLON %s\n", s.c_str());
            continue ;
        }
        
        std::string dk = s.substr(0, colon);
        std::string lk = s.substr(colon+1);
        //printf("GSubstanceLib::setKeyMap dk [%s] lk [%s] \n", dk.c_str(), lk.c_str());
        m_keymap[dk] = lk ; 
    }
}

const char* GSubstanceLib::getLocalKey(const char* dkey) // mapping between standard keynames and local key names, eg refractive_index -> RINDEX
{
    return m_keymap[dkey].c_str();
}




GPropertyMap* GSubstanceLib::createStandardProperties(const char* pname, GSubstance* substance)
{
    // hmm combining all 4-sets into one PropertyMap
    GPropertyMap* ptex = new GPropertyMap(pname);

    addMaterialProperties(ptex, substance->getInnerMaterial(), inner);
    addMaterialProperties(ptex, substance->getOuterMaterial(), outer);
    addSurfaceProperties( ptex, substance->getInnerSurface(),  inner);
    addSurfaceProperties( ptex, substance->getOuterSurface(),  outer);

    checkMaterialProperties(ptex,  0 , inner);
    checkMaterialProperties(ptex,  4 , outer);
    checkSurfaceProperties( ptex,  8 , inner);
    checkSurfaceProperties( ptex, 12 , outer);

    return ptex ; 
}




void GSubstanceLib::defineDefaults(GPropertyMap* defaults)
{

    defaults->addConstantProperty( refractive_index,      1.f  );
    defaults->addConstantProperty( absorption_length,     1e6  );
    defaults->addConstantProperty( scattering_length,     1e6  );
    defaults->addConstantProperty( reemission_prob,       0.f  );

    defaults->addConstantProperty( detect          ,       0.f );
    defaults->addConstantProperty( absorb          ,       0.f );
    defaults->addConstantProperty( reflect_specular,       0.f );
    defaults->addConstantProperty( reflect_diffuse ,       0.f );
}

void GSubstanceLib::addMaterialProperties(GPropertyMap* ptex, GPropertyMap* pmap, const char* prefix)
{
    assert(pmap);  // materials must always be defined
    assert(pmap->isMaterial());
    assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));

    if(ptex->hasStandardDomain())
    {
        assert(ptex->getStandardDomain()->isEqual(pmap->getStandardDomain()));
    } 
    else
    {
        ptex->setStandardDomain(pmap->getStandardDomain());
    }

    ptex->addProperty(refractive_index, getPropertyOrDefault( pmap, refractive_index ), prefix);
    ptex->addProperty(absorption_length,getPropertyOrDefault( pmap, absorption_length ), prefix);
    ptex->addProperty(scattering_length,getPropertyOrDefault( pmap, scattering_length ), prefix);
    ptex->addProperty(reemission_prob  ,getPropertyOrDefault( pmap, reemission_prob ), prefix);
}

void GSubstanceLib::addSurfaceProperties(GPropertyMap* ptex, GPropertyMap* pmap, const char* prefix)
{
    if(pmap) // surfaces often not defined
    { 
        assert(pmap->isSkinSurface() || pmap->isBorderSurface());
        assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));
    }

    ptex->addProperty(detect,           getPropertyOrDefault( pmap, detect ), prefix);
    ptex->addProperty(absorb,           getPropertyOrDefault( pmap, absorb ), prefix);
    ptex->addProperty(reflect_specular, getPropertyOrDefault( pmap, reflect_specular ), prefix);
    ptex->addProperty(reflect_diffuse,  getPropertyOrDefault( pmap, reflect_diffuse  ), prefix);
}


GPropertyD* GSubstanceLib::getPropertyOrDefault(GPropertyMap* pmap, const char* dkey)
{
    const char* lkey = getLocalKey(dkey); assert(lkey);  // missing local key mapping 

    GPropertyD* fallback = getDefaultProperty(dkey);  assert(fallback);

    GPropertyD* prop = pmap ? pmap->getProperty(lkey) : NULL ;

    return prop ? prop : fallback ;
}

void GSubstanceLib::checkMaterialProperties(GPropertyMap* ptex, unsigned int offset, const char* _prefix)
{
    std::string prefix = _prefix ; 
    std::string xname, pname ;

    xname = prefix + refractive_index ; 
    pname = ptex->getPropertyNameByIndex(offset+e_refractive_index);
    assert(xname == pname);

    xname = prefix + absorption_length  ; 
    pname = ptex->getPropertyNameByIndex(offset+e_absorption_length);
    assert(xname == pname);

    xname = prefix + scattering_length  ; 
    pname = ptex->getPropertyNameByIndex(offset+e_scattering_length);
    assert(xname == pname);

    xname = prefix + reemission_prob  ; 
    pname = ptex->getPropertyNameByIndex(offset+e_reemission_prob);
    assert(xname == pname);

}

void GSubstanceLib::checkSurfaceProperties(GPropertyMap* ptex, unsigned int offset, const char* _prefix)
{
    std::string prefix = _prefix ; 
    std::string xname, pname ;

    xname = prefix + detect ; 
    pname = ptex->getPropertyNameByIndex(offset+e_detect);
    assert(xname == pname);

    xname = prefix + absorb ; 
    pname = ptex->getPropertyNameByIndex(offset+e_absorb);
    assert(xname == pname);

    xname = prefix + reflect_specular ; 
    pname = ptex->getPropertyNameByIndex(offset+e_reflect_specular);
    assert(xname == pname);

    xname = prefix + reflect_diffuse ; 
    pname = ptex->getPropertyNameByIndex(offset+e_reflect_diffuse);
    assert(xname == pname);
}


GBuffer* GSubstanceLib::createWavelengthBuffer()
{
    //
    // 4 sets of 4 props  
    //
    //  The 4 sets for: 
    //          (inner material, outer material, inner surface, outer surface) 
    //
    //  and the 4 props:
    //         material: (refractive_index, absorption_length, scattering_length, reemission_prob)
    //         surface:  (detect, absorb, reflect_specular, reflect_diffuse)    
    //

    GBuffer* buffer(NULL) ;
    float* data(NULL) ;

    if(!m_meta) m_meta = new GSubstanceLibMetadata ; 

    unsigned int numSubstance = getNumSubstances() ;

    for(unsigned int isub=0 ; isub < numSubstance ; isub++)
    {
        GSubstance* substance = getSubstance(isub);
        unsigned int substanceIndex = substance->getIndex();
        assert(substanceIndex < numSubstance);

        assert(isub == substanceIndex);
        //printf("GSubstanceLib::createWavelengthBuffer isub %u substanceIndex  %u \n", isub, substanceIndex );

        char* ishortname = substance->getInnerMaterial()->getShortName("__dd__Materials__") ; 
        char* oshortname = substance->getOuterMaterial()->getShortName("__dd__Materials__") ; 
     
        const char* kfmt_subs = "lib.substance.%d.%s.%s" ;
        m_meta->add(kfmt_subs, isub, "imat", substance->getInnerMaterial() );
        m_meta->add(kfmt_subs, isub, "omat", substance->getOuterMaterial() );
        m_meta->add(kfmt_subs, isub, "isur", substance->getInnerSurface() );
        m_meta->add(kfmt_subs, isub, "osur", substance->getOuterSurface() );

        GPropertyMap* ptex = createStandardProperties("ptex", substance);
        unsigned int numProp = ptex->getNumProperties() ;
        assert(numProp == 16);
        assert(numProp % 4 == 0);

        GDomain<double>* domain = ptex->getStandardDomain();
        const unsigned int domainLength = domain->getLength();

        unsigned int subOffset = domainLength*numProp*substanceIndex ; 

        if( buffer == NULL )
        {
            unsigned int numFloat = domainLength*numProp*numSubstance ; 
            buffer = new GBuffer( sizeof(float)*numFloat, new float[numFloat], sizeof(float), 1 );
            data = (float*)buffer->getPointer();
        }

        for( unsigned int p = 0; p < numProp/4 ; ++p )  // over 4 different sets (imat,omat,isur,osur)  
        {   
            unsigned int propOffset = p*numProp/4 ;  // 0, 4, 8, 12
            
            // 4 properties of the set 
            GPropertyD* p0 = ptex->getPropertyByIndex(propOffset+0) ;
            GPropertyD* p1 = ptex->getPropertyByIndex(propOffset+1) ;
            GPropertyD* p2 = ptex->getPropertyByIndex(propOffset+2) ;
            GPropertyD* p3 = ptex->getPropertyByIndex(propOffset+3) ;

            // record standard property digest into metadata
            std::vector<GPropertyD*> props ; 
            props.push_back(p0);
            props.push_back(p1);
            props.push_back(p2);
            props.push_back(p3);
            char* pdig = digest(props);
            switch(p)
            {
               case 0:
                      m_meta->addDigest("lib.substance.%d.%s.%s", isub, "imat", pdig ); 
                      m_meta->addMaterial(isub, "imat", ishortname, pdig );
                      break ;
               case 1:
                      m_meta->addDigest("lib.substance.%d.%s.%s", isub, "omat", pdig ); 
                      m_meta->addMaterial(isub, "omat", oshortname, pdig );
                      break ;
               case 2:
                      m_meta->addDigest("lib.substance.%d.%s.%s", isub, "isur", pdig ); 
                      break ;
               case 3:
                      m_meta->addDigest("lib.substance.%d.%s.%s", isub, "osur", pdig ); 
                      break ;
            }
            free(pdig);

            for( unsigned int d = 0; d < domainLength; ++d ) 
            {   
                unsigned int dataOffset = ( p*domainLength + d )*4;  
                data[subOffset+dataOffset+0] = p0->getValue(d) ;
                data[subOffset+dataOffset+1] = p1->getValue(d) ;
                data[subOffset+dataOffset+2] = p2->getValue(d) ;
                data[subOffset+dataOffset+3] = p3->getValue(d) ;
            }       
        }   
        free(ishortname); 
        free(oshortname); 
    }
    m_meta->createMaterialMap();
    return buffer ; 
}


char* GSubstanceLib::digest(std::vector<GPropertyD*>& props)
{
    MD5Digest dig ;            
    for(unsigned int i=0 ; i < props.size() ; ++i)
    {
        GPropertyD* p = props[i]; 
        char* pdig = p->digest();
        dig.update(pdig, strlen(pdig));
        free(pdig);
    }
    return dig.finalize();
}



