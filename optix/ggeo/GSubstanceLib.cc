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

GSubstanceLib::GSubstanceLib() : m_defaults(NULL), m_meta(NULL), m_standard(true)
{
    setKeyMap(NULL);
    GDomain<double>* domain = new GDomain<double>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
    setStandardDomain( domain );

    GPropertyMap* defaults = new GPropertyMap("defaults", UINT_MAX, "defaults");
    defaults->setStandardDomain(getStandardDomain());

    defineDefaults(defaults);
    setDefaults(defaults);

    m_ramp = GProperty<double>::ramp( domain->getLow(), domain->getStep(), domain->getValues(), domain->getLength() );
}

GSubstanceLib::~GSubstanceLib()
{
}

unsigned int GSubstanceLib::getNumSubstances()
{
   return m_keys.size();
}

GDomain<double>* GSubstanceLib::getDefaultDomain()
{
   return new GDomain<double>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
}

void GSubstanceLib::setStandardDomain(GDomain<double>* standard_domain)
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
        if(substance->getIndex() != index )
        {
            printf("GSubstanceLib::getSubstance WARNING substance index mismatch request %u substance %u key %s \n", index, substance->getIndex(), key.c_str() ); 
        } 
        //assert(substance->getIndex() == index );
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
    // this "get" pulls the GSubstance into existance and populates the registry
    //printf("GSubstanceLib::get imaterial %p omaterial %p isurface %p osurface %p \n", imaterial, omaterial, isurface, osurface );

    GSubstance raw(imaterial, omaterial, isurface, osurface);
    GSubstance* standard = createStandardSubstance(&raw) ;
    std::string key = standard->digest();  // standard digest based identity 

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
    GPropertyMap* imat = substance->getInnerMaterial();
    GPropertyMap* omat = substance->getOuterMaterial();
    GPropertyMap* isur = substance->getInnerSurface();
    GPropertyMap* osur = substance->getOuterSurface();

    GPropertyMap* s_imat = new GPropertyMap(imat);
    GPropertyMap* s_omat = new GPropertyMap(omat);
    GPropertyMap* s_isur = new GPropertyMap(isur);
    GPropertyMap* s_osur = new GPropertyMap(osur);

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


#ifdef TRANSITIONAL
GSubstanceLib* GSubstanceLib::createStandardizedLib()
{
    // hmm are preventing it for now, as using the index for the key for
    // transitional comparison, but once resume digest keying this
    // standardization will shrink the number of substances (expect by 4, from 58 to 54)

    GSubstanceLib* s_lib = new GSubstanceLib(); 
    for(unsigned int isub=0 ; isub < getNumSubstances() ; isub++)
    {
        GSubstance* a = getSubstance(isub);
        GSubstance* b = createStandardSubstance(a);
        s_lib->add(b);
    }
    s_lib->setStandard(true);
    return s_lib ;
}

void GSubstanceLib::add(GSubstance* substance)
{
    // GSubstanceLib::add this only used for standardized lib formation
    //
    // transitionally using substance digest keys yielding duplicate keys 
    // avoid that whilst still comparing against unstandardized by using the count 
    //

    //std::string key = substance->digest();   
    std::stringstream ss ; 
    ss << m_keys.size() ;
    std::string key = ss.str() ;  
     
    if(m_registry.count(key) == 0) // not yet registered, identity based on GSubstance digest 
    { 
        substance->setIndex(m_keys.size());
        m_keys.push_back(key);     // for simple ordering  
        m_registry[key] = substance  ; 
    }
    else
    {
        printf("GSubstanceLib::add WARNING adding duplicate substance key %s \n", key.c_str());
        assert(0);
    } 
}
#endif



#ifdef LEGACY
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
#endif


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

void GSubstanceLib::addMaterialProperties(GPropertyMap* pstd, GPropertyMap* pmap, const char* prefix)
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

void GSubstanceLib::addSurfaceProperties(GPropertyMap* pstd, GPropertyMap* pmap, const char* prefix)
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


GPropertyD* GSubstanceLib::getPropertyOrDefault(GPropertyMap* pmap, const char* dkey)
{
    const char* lkey = getLocalKey(dkey); assert(lkey);  // missing local key mapping 

    GPropertyD* fallback = getDefaultProperty(dkey);  assert(fallback);

    GPropertyD* prop = pmap ? pmap->getProperty(lkey) : NULL ;

    return prop ? prop : fallback ;
}



#ifdef LEGACY
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
#endif


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

        printf("GSubstanceLib::createWavelengthBuffer isub %u/%u substanceIndex  %u \n", isub, numSubstance, substanceIndex );
        assert(substanceIndex < numSubstance);
        //assert(isub == substanceIndex);

        char* ishortname = substance->getInnerMaterial()->getShortName("__dd__Materials__") ; 
        char* oshortname = substance->getOuterMaterial()->getShortName("__dd__Materials__") ; 
        {
            const char* kfmt = "lib.substance.%d.%s.%s" ;
            m_meta->add(kfmt, isub, "imat", substance->getInnerMaterial() );
            m_meta->add(kfmt, isub, "omat", substance->getOuterMaterial() );
            m_meta->add(kfmt, isub, "isur", substance->getInnerSurface() );
            m_meta->add(kfmt, isub, "osur", substance->getOuterSurface() );
        }

        GPropertyMap* ptex(NULL);
        GDomain<double>* domain(NULL);
        unsigned int numProp = 16 ; 


#ifdef TRANSITIONAL
        if(!isStandard())
        {
            ptex = createStandardProperties("ptex", substance);
            domain = ptex->getStandardDomain();
            assert(numProp == ptex->getNumProperties());
        }
        else
#endif
        {
            domain = getStandardDomain();
        }

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
            // 4 properties of the set 
            GPropertyD *p0,*p1,*p2,*p3 ; 
      //      if(isStandard())
      //      {
                GPropertyMap* psrc ; 
                switch(p)
                {
                    case 0:psrc = substance->getInnerMaterial() ; break ; 
                    case 1:psrc = substance->getOuterMaterial() ; break ; 
                    case 2:psrc = substance->getInnerSurface()  ; break ; 
                    case 3:psrc = substance->getOuterSurface()  ; break ; 
                } 
                p0 = psrc->getPropertyByIndex(0);
                p1 = psrc->getPropertyByIndex(1);
                p2 = psrc->getPropertyByIndex(2);
                p3 = psrc->getPropertyByIndex(3);
      //      }
      /*
            else 
            {
                unsigned int propOffset = p*numProp/4 ;  // 0, 4, 8, 12
                p0 = ptex->getPropertyByIndex(propOffset+0) ;
                p1 = ptex->getPropertyByIndex(propOffset+1) ;
                p2 = ptex->getPropertyByIndex(propOffset+2) ;
                p3 = ptex->getPropertyByIndex(propOffset+3) ;
            }
      */
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


void GSubstanceLib::dumpWavelengthBuffer(GBuffer* buffer)
{
    dumpWavelengthBuffer(buffer, getNumSubstances(), 16, getStandardDomainLength());  
}

void GSubstanceLib::dumpWavelengthBuffer(GBuffer* buffer, unsigned int numSubstance, unsigned int numProp, unsigned int domainLength)
{
    if(!buffer) return ;

    float* data = (float*)buffer->getPointer();

    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    assert(numElementsTotal == numSubstance*numProp*domainLength);

    GDomain<double>* domain = GSubstanceLib::getDefaultDomain();
    assert(domain->getLength() == domainLength);

    std::cout << "GMergedMesh::dumpWavelengthBuffer " 
              << " numSubstance " << numSubstance
              << " numProp " << numProp
              << " domainLength " << domainLength
              << std::endl ; 

    assert(numProp % 4 == 0);


    std::vector<GPropertyMap*> pmaps ; 

    for(unsigned int isub=0 ; isub < numSubstance ; ++isub )
    {
        unsigned int subOffset = domainLength*numProp*isub ;

        GPropertyMap* pmap = new GPropertyMap("humpty", isub, "reconstructed");
        pmaps.push_back(pmap);

        for(unsigned int p=0 ; p < numProp/4 ; ++p ) // property scrunch into float4 is the cause of the gymnastics
        {
             unsigned int offset = subOffset + ( p*domainLength*4 ) ;

             // un-interleaving the 4 properties
             for(unsigned int l=0 ; l < 4 ; ++l )
             {
                 double* v = new double[domainLength];
                 for(unsigned int d=0 ; d < domainLength ; ++d ) v[d] = data[offset+d*4+l];  
                 char pname[32];
                 snprintf(pname, 32, "p%ul%u", p, l);
                 pmap->addProperty(pname, v, domain->getValues(), domainLength );
                 delete v;
             }

             char* dig4 = pmap->pdigest(-4,0);  // last four properties digest-of-digest
             printf("sub %u/%u  prop %u/%u offset %u dig4 %s \n", isub, numSubstance, p, numProp/4, offset, dig4 );
             free(dig4);

             for(unsigned int l=0 ; l < 4 ; ++l )
             {
                 for(unsigned int d=0 ; d < domainLength ; ++d )
                 {
                     if(d%5 == 0) printf(" %15.3f", data[offset+d*4+l] );  // too many numbers so display one in every 5
                 }
                 printf("\n");

                 //GProperty<double>* prop = pmap->getPropertyByIndex(l-4);
                 //char* dig = prop->digest();
                 //printf(" %s\n", dig);          // single property digest
                 //free(dig);
             }


        }
    }
}




