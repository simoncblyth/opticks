#include "GSubstanceLib.hh"
#include "GSubstanceLibMetadata.hh"
#include "GSubstance.hh"
#include "GPropertyMap.hh"
#include "GBuffer.hh"
#include "GEnums.hh"


#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;


#include <sstream>
#include "assert.h"
#include "stdio.h"
#include "limits.h"
#include "string.h"

float        GSubstanceLib::DOMAIN_LOW  = 60.f ; 
float        GSubstanceLib::DOMAIN_HIGH = 810.f ; 
float        GSubstanceLib::DOMAIN_STEP = 20.f ; 
unsigned int GSubstanceLib::DOMAIN_LENGTH = 39  ; 

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

GSubstanceLib::GSubstanceLib() : m_defaults(NULL), m_meta(NULL), m_standard(true), m_num_prop(16)
{
    setKeyMap(NULL);
    GDomain<float>* domain = new GDomain<float>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
    setStandardDomain( domain );

    GPropertyMap<float>* defaults = new GPropertyMap<float>("defaults", UINT_MAX, "defaults");
    defaults->setStandardDomain(getStandardDomain());

    defineDefaults(defaults);
    setDefaults(defaults);

    m_ramp = GProperty<float>::ramp( domain->getLow(), domain->getStep(), domain->getValues(), domain->getLength() );
}

GSubstanceLib::~GSubstanceLib()
{
}

unsigned int GSubstanceLib::getNumSubstances()
{
   return m_keys.size();
}

GDomain<float>* GSubstanceLib::getDefaultDomain()
{
   return new GDomain<float>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
}

void GSubstanceLib::setStandardDomain(GDomain<float>* standard_domain)
{
    m_standard_domain = standard_domain ; 

    assert(getStandardDomainLength() == DOMAIN_LENGTH );
    assert(m_standard_domain->getLow()  == DOMAIN_LOW );
    assert(m_standard_domain->getHigh() == DOMAIN_HIGH );
    assert(m_standard_domain->getStep() == DOMAIN_STEP );
}

unsigned int GSubstanceLib::getStandardDomainLength()
{
    return m_standard_domain ? m_standard_domain->getLength() : 0 ;
}

GProperty<float>* GSubstanceLib::getDefaultProperty(const char* name)
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
        if(substance->getIndex() != index )
        {
            printf("GSubstanceLib::getSubstance WARNING substance index mismatch request %u substance %u key %s \n", index, substance->getIndex(), key.c_str() ); 
        } 
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

GSubstance* GSubstanceLib::get(GPropertyMap<float>* imaterial, GPropertyMap<float>* omaterial, GPropertyMap<float>* isurface, GPropertyMap<float>* osurface )
{ 
    // this "get" pulls the GSubstance into existance and populates the registry
    //printf("GSubstanceLib::get imaterial %p omaterial %p isurface %p osurface %p \n", imaterial, omaterial, isurface, osurface );

    GSubstance raw(imaterial, omaterial, isurface, osurface);
    GSubstance* standard = createStandardSubstance(&raw) ;
    std::string key = standard->pdigest(0,4);  // standard digest based identity 

    if(m_registry.count(key) == 0) // not yet registered
    { 
        standard->setIndex(m_keys.size());
        m_keys.push_back(key);  // for simple ordering  
        m_registry[key] = standard ; 
    }
    else
    {
        delete standard ; 
    } 

    GSubstance* substance = m_registry[key] ;
    //printf("GSubstanceLib::get key %s index %u \n", key.c_str(), substance->getIndex()); 
    return substance ; 
}



GSubstance* GSubstanceLib::createStandardSubstance(GSubstance* substance)
{
    GPropertyMap<float>* imat = substance->getInnerMaterial();
    GPropertyMap<float>* omat = substance->getOuterMaterial();
    GPropertyMap<float>* isur = substance->getInnerSurface();
    GPropertyMap<float>* osur = substance->getOuterSurface();

    GPropertyMap<float>* s_imat = new GPropertyMap<float>(imat);
    GPropertyMap<float>* s_omat = new GPropertyMap<float>(omat);
    GPropertyMap<float>* s_isur = new GPropertyMap<float>(isur);
    GPropertyMap<float>* s_osur = new GPropertyMap<float>(osur);

    addMaterialProperties( s_imat, imat, inner );
    addMaterialProperties( s_omat, omat, outer );
    addSurfaceProperties(  s_isur, isur, inner );
    addSurfaceProperties(  s_osur, osur, outer );

    GSubstance* s_substance = new GSubstance( s_imat , s_omat, s_isur, s_osur );
    return s_substance ; 
}


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



GPropertyMap<float>* GSubstanceLib::createStandardProperties(const char* pname, GSubstance* substance)
{
    // combining all 4-sets into one PropertyMap : for insertion into wavelengthBuffer

    GPropertyMap<float>* ptex = new GPropertyMap<float>(pname);
    
    ptex->add(substance->getInnerMaterial(), inner);
    ptex->add(substance->getOuterMaterial(), outer);
    ptex->add(substance->getInnerSurface(),  inner);
    ptex->add(substance->getOuterSurface(),  outer);

    checkMaterialProperties(ptex,  0 , inner);
    checkMaterialProperties(ptex,  4 , outer);
    checkSurfaceProperties( ptex,  8 , inner);
    checkSurfaceProperties( ptex, 12 , outer);

    return ptex ; 
}


void GSubstanceLib::defineDefaults(GPropertyMap<float>* defaults)
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

void GSubstanceLib::addMaterialProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix)
{
    assert(pmap);  // materials must always be defined
    assert(pmap->isMaterial());
    assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));

    if(pstd->hasStandardDomain())
    {
        assert(pstd->getStandardDomain()->isEqual(pmap->getStandardDomain()));
    } 
    else
    {
        pstd->setStandardDomain(pmap->getStandardDomain());
    }

    pstd->addProperty(refractive_index, getPropertyOrDefault( pmap, refractive_index ), prefix);
    pstd->addProperty(absorption_length,getPropertyOrDefault( pmap, absorption_length ), prefix);
    pstd->addProperty(scattering_length,getPropertyOrDefault( pmap, scattering_length ), prefix);
    pstd->addProperty(reemission_prob  ,getPropertyOrDefault( pmap, reemission_prob ), prefix);
}

void GSubstanceLib::addSurfaceProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix)
{
    if(pmap) // surfaces often not defined
    { 
        assert(pmap->isSkinSurface() || pmap->isBorderSurface());
        assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));
    }

    pstd->addProperty(detect,           getPropertyOrDefault( pmap, detect ), prefix);
    pstd->addProperty(absorb,           getPropertyOrDefault( pmap, absorb ), prefix);
    pstd->addProperty(reflect_specular, getPropertyOrDefault( pmap, reflect_specular ), prefix);
    pstd->addProperty(reflect_diffuse,  getPropertyOrDefault( pmap, reflect_diffuse  ), prefix);
}


GProperty<float>* GSubstanceLib::getPropertyOrDefault(GPropertyMap<float>* pmap, const char* dkey)
{
    const char* lkey = getLocalKey(dkey); assert(lkey);  // missing local key mapping 

    GProperty<float>* fallback = getDefaultProperty(dkey);  assert(fallback);

    GProperty<float>* prop = pmap ? pmap->getProperty(lkey) : NULL ;

    return prop ? prop : fallback ;
}



void GSubstanceLib::checkMaterialProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* _prefix)
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
void GSubstanceLib::checkSurfaceProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* _prefix)
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

const char* GSubstanceLib::getDigest(unsigned int index)
{
    return m_keys[index].c_str();
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

    GDomain<float>* domain = getStandardDomain();
    const unsigned int domainLength = domain->getLength();
    unsigned int numSubstance = getNumSubstances() ;
    unsigned int numProp = getNumProp() ; 
    unsigned int numFloat = domainLength*numProp*numSubstance ; 

    for(unsigned int isub=0 ; isub < numSubstance ; isub++)
    {
        GSubstance* substance = getSubstance(isub);
        unsigned int substanceIndex = substance->getIndex();

        //printf("GSubstanceLib::createWavelengthBuffer isub %u/%u substanceIndex  %u \n", isub, numSubstance, substanceIndex );
        assert(substanceIndex < numSubstance);
        assert(isub == substanceIndex);

        const char* dig = getDigest(isub);
        {
            char* ckdig = substance->pdigest(0,4);
            assert(strcmp(ckdig, dig) == 0);
        }

        unsigned int subOffset = domainLength*numProp*substanceIndex ; 
        const char* kfmt = "lib.substance.%d.%s.%s" ;
        m_meta->addDigest(kfmt, isub, "substance", (char*)dig ); 

        char* ishortname = substance->getInnerMaterial()->getShortName("__dd__Materials__") ; 
        char* oshortname = substance->getOuterMaterial()->getShortName("__dd__Materials__") ; 
        {
            m_meta->add(kfmt, isub, "imat", substance->getInnerMaterial() );
            m_meta->add(kfmt, isub, "omat", substance->getOuterMaterial() );
            m_meta->add(kfmt, isub, "isur", substance->getInnerSurface() );
            m_meta->add(kfmt, isub, "osur", substance->getOuterSurface() );
        }

        if( buffer == NULL )
        {
            buffer = new GBuffer( sizeof(float)*numFloat, new float[numFloat], sizeof(float), 1 );
            data = (float*)buffer->getPointer();
        }

        for( unsigned int p = 0; p < numProp/4 ; ++p )  // over 4 different sets (imat,omat,isur,osur)  
        { 
            // 4 properties of the set 
            GPropertyF *p0,*p1,*p2,*p3 ; 
            GPropertyMap<float>* psrc = substance->getConstituentByIndex(p) ; 

            p0 = psrc->getPropertyByIndex(0);
            p1 = psrc->getPropertyByIndex(1);
            p2 = psrc->getPropertyByIndex(2);
            p3 = psrc->getPropertyByIndex(3);

            // record standard 4-property digest into metadata

            std::vector<GPropertyF*> props ; 
            props.push_back(p0);
            props.push_back(p1);
            props.push_back(p2);
            props.push_back(p3);
            char* pdig = digest(props);

            {
               std::string ckdig = psrc->getPDigestString(0,4);
               assert(strcmp(pdig, ckdig.c_str())==0);
            }

            switch(p)
            {
               case 0:
                      m_meta->addDigest(kfmt, isub, "imat", pdig ); 
                      m_meta->addMaterial(isub, "imat", ishortname, pdig );
                      break ;
               case 1:
                      m_meta->addDigest(kfmt, isub, "omat", pdig ); 
                      m_meta->addMaterial(isub, "omat", oshortname, pdig );
                      break ;
               case 2:
                      m_meta->addDigest(kfmt, isub, "isur", pdig ); 
                      break ;
               case 3:
                      m_meta->addDigest(kfmt, isub, "osur", pdig ); 
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


char* GSubstanceLib::digest(std::vector<GProperty<float>*>& props)
{
    MD5Digest dig ;            
    for(unsigned int i=0 ; i < props.size() ; ++i)
    {
        GProperty<float>* p = props[i]; 
        char* pdig = p->digest();
        dig.update(pdig, strlen(pdig));
        free(pdig);
    }
    return dig.finalize();
}


const char* GSubstanceLib::materialPropertyName(unsigned int i)
{
    assert(i < 4);
    if(i == 0) return refractive_index ;
    if(i == 1) return absorption_length ;
    if(i == 2) return scattering_length ;
    if(i == 3) return reemission_prob ;
    return "?" ;
}

const char* GSubstanceLib::surfacePropertyName(unsigned int i)
{
    assert(i < 4);
    if(i == 0) return detect ;
    if(i == 1) return absorb ;
    if(i == 2) return reflect_specular;
    if(i == 3) return reflect_diffuse ;
    return "?" ;
}


char* GSubstanceLib::propertyName(unsigned int p, unsigned int i)
{
    assert(p < 4);
    char name[64];
    switch(p)
    {
       case 0: snprintf(name, 64, "%s%s", inner, materialPropertyName(i) ); break;
       case 1: snprintf(name, 64, "%s%s", outer, materialPropertyName(i) ); break;
       case 2: snprintf(name, 64, "%s%s", inner, surfacePropertyName(i) ); break;
       case 3: snprintf(name, 64, "%s%s", outer, surfacePropertyName(i) ); break;
    }
    return strdup(name);
}

std::string GSubstanceLib::propertyNameString(unsigned int p, unsigned int i)
{
    return propertyName(p,i);
}

GSubstance* GSubstanceLib::loadSubstance(float* subData, unsigned int isub)
{
    GSubstance* substance = new GSubstance ; 
    GDomain<float>* domain = GSubstanceLib::getDefaultDomain();
    unsigned int domainLength = domain->getLength(); 
    unsigned int numProp = getNumProp();
    assert(numProp % 4 == 0 && numProp/4 == 4);

    std::string mdig = m_meta->getSubstanceQty(isub, "substance", "digest");

    // property scrunch into float4 is the cause of the gymnastics
    for(unsigned int p=0 ; p < numProp/4 ; ++p ) 
    {
         std::string mapName = m_meta->getSubstanceQtyByIndex(isub, p, "name");
         GPropertyMap<float>* pmap = new GPropertyMap<float>(mapName.c_str(), isub, "recon"); 

         float* pdata = subData + p*domainLength*4 ; 

         for(unsigned int l=0 ; l < 4 ; ++l ) // un-interleaving the 4 properties
         {
             float* v = new float[domainLength];
             for(unsigned int d=0 ; d < domainLength ; ++d ) v[d] = pdata[d*4+l];  

             std::string pname = propertyNameString(p, l);
             pmap->addProperty(pname.c_str(), v, domain->getValues(), domainLength );
             delete v;
         }

         switch(p)
         {
            case 0:substance->setInnerMaterial(pmap);break;
            case 1:substance->setOuterMaterial(pmap);break;
            case 2:substance->setInnerSurface(pmap);break;
            case 3:substance->setOuterSurface(pmap);break;
         }
     }

     std::string sdig = substance->getPDigestString(0,4);

     if(strcmp(sdig.c_str(), mdig.c_str()) != 0)
     {
         printf("GSubstanceLib::loadSubstance digest mismatch %u : %s %s \n", isub, sdig.c_str(), mdig.c_str());
         digestDebug(substance, isub);
     }
     assert(strcmp(sdig.c_str(), mdig.c_str()) == 0); 
     return substance ; 
}

void GSubstanceLib::digestDebug(GSubstance* substance, unsigned int isub)
{
    for(unsigned int i=0 ; i<4 ; i++)
    {
        GPropertyMap<float>* q = substance->getConstituentByIndex(i);
        std::string qdig   = q->getPDigestString(0,4);
        std::string qdigm  = m_meta->getSubstanceQtyByIndex(isub, i, "digest");

        if(strcmp(qdigm.c_str(), qdig.c_str())!=0)
        {
            switch(i)
            {
                case 0:
                case 1:
                        printf("xmat %20s %s %s \n", q->getShortNameString().c_str(), qdig.c_str(), qdigm.c_str());
                        break ; 
                case 2:
                case 3:
                        printf("xsur %20s %s %s \n", "", qdig.c_str(), qdigm.c_str());
                        break ; 
            }
            GProperty<float>* prop ; 
            for(unsigned int p=0 ; p < 4 ; ++p)
            {
                prop = q->getPropertyByIndex(p);
                std::string pdig = prop->getDigestString();
                printf(" %u prop digest %s \n", p, pdig.c_str());
            }
         }
     } 
}




GSubstanceLib* GSubstanceLib::load(const char* dir)
{
    GSubstanceLib* lib = new GSubstanceLib();

    GSubstanceLibMetadata* meta = GSubstanceLibMetadata::load(dir);
    lib->setMetadata(meta); 

    GBuffer* buffer = GBuffer::load<float>(dir, "wavelength.npy");
    buffer->Summary("wavelength buffer");
    //lib->dumpWavelengthBuffer(buffer);
    lib->loadWavelengthBuffer(buffer);

    return lib ; 
}


void GSubstanceLib::loadWavelengthBuffer(GBuffer* buffer)
{
    if(!buffer) return ;
    float* data = (float*)buffer->getPointer();

    unsigned int numElementsTotal = buffer->getNumElementsTotal();

    GDomain<float>* domain = GSubstanceLib::getDefaultDomain();
    unsigned int domainLength = domain->getLength(); 
    unsigned int numProp = getNumProp();
    unsigned int numSubstance = numElementsTotal/(numProp*domainLength);
    assert(numSubstance == 54);

    for(unsigned int isub=0 ; isub < numSubstance ; ++isub )
    {
        unsigned int subOffset = domainLength*numProp*isub ;
        GSubstance* substance = loadSubstance(data + subOffset, isub); 
        //substance->Summary("GSubstanceLib::loadWavelengthBuffer",1);

        std::string key = substance->pdigest(0,4);  
        assert(m_registry.count(key) == 0); // there should be no digest duplicates in wavelengthBuffer

        substance->setIndex(m_keys.size());
        m_keys.push_back(key);  // for simple ordering  
        m_registry[key] = substance ; 

        // use metadata to reacreate the names 
        // find way to do roundtrip test (maybe via global digest of the lib) 
        
    }
}

void GSubstanceLib::dumpWavelengthBuffer(GBuffer* buffer)
{
    dumpWavelengthBuffer(buffer, getNumSubstances(), getNumProp(), getStandardDomainLength());  
}

void GSubstanceLib::dumpWavelengthBuffer(GBuffer* buffer, unsigned int numSubstance, unsigned int numProp, unsigned int domainLength)
{
    if(!buffer) return ;

    float* data = (float*)buffer->getPointer();
    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    assert(numElementsTotal == numSubstance*numProp*domainLength);
    GDomain<float>* domain = GSubstanceLib::getDefaultDomain();
    assert(domain->getLength() == domainLength);

    std::cout << "GSubstanceLib::dumpWavelengthBuffer " 
              << " numSubstance " << numSubstance
              << " numProp " << numProp
              << " domainLength " << domainLength
              << std::endl ; 

    assert(numProp % 4 == 0);

    for(unsigned int isub=0 ; isub < numSubstance ; ++isub )
    {
        unsigned int subOffset = domainLength*numProp*isub ;
        for(unsigned int p=0 ; p < numProp/4 ; ++p ) // property scrunch into float4 is the cause of the gymnastics
        {
             unsigned int offset = subOffset + ( p*domainLength*4 ) ;
             for(unsigned int l=0 ; l < 4 ; ++l )
             {
                 for(unsigned int d=0 ; d < domainLength ; ++d )
                 {
                     if(d%5 == 0) printf(" %15.3f", data[offset+d*4+l] );  // too many numbers so display one in every 5
                 }
                 printf("\n");
             }
        }
    }
}




