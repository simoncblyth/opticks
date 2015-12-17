#include "GBndLib.hh"
#include "GPropertyMap.hh"


#include "NPY.hpp"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GItemList.hh"

#include <climits>
#include <cassert>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

void GBndLib::save()
{
    // NB bndlib exists for deferred boundary buffer creation,  
    saveIndexBuffer();  
}

GBndLib* GBndLib::load(GCache* cache, bool constituents)
{
    GBndLib* blib = new GBndLib(cache);
    blib->loadIndexBuffer();
    blib->importIndexBuffer();

    if(constituents)
    {
        GMaterialLib* mlib = GMaterialLib::load(cache);
        GSurfaceLib* slib = GSurfaceLib::load(cache);
        blib->setMaterialLib(mlib);
        blib->setSurfaceLib(slib);
    }
    return blib ; 
}

void GBndLib::loadIndexBuffer()
{
    std::string dir = getCacheDir(); 
    std::string name = getBufferName("Index");
    setIndexBuffer(NPY<unsigned int>::load(dir.c_str(), name.c_str())); 
}

void GBndLib::saveIndexBuffer()
{
    NPY<unsigned int>* index_buffer = createIndexBuffer();
    setIndexBuffer(index_buffer);
    saveToCache(index_buffer, "Index") ; 
}

void GBndLib::saveOpticalBuffer()
{
    NPY<unsigned int>* optical_buffer = createOpticalBuffer();
    setOpticalBuffer(optical_buffer);
    saveToCache(optical_buffer, "Optical") ; 
}

void GBndLib::createDynamicBuffers()
{
    NPY<float>* buf = createBuffer();
    setBuffer(buf);

    NPY<unsigned int>* optical_buffer = createOpticalBuffer();
    setOpticalBuffer(optical_buffer);
}



NPY<unsigned int>* GBndLib::createIndexBuffer()
{
    return createUint4Buffer(m_bnd);
}

void GBndLib::importIndexBuffer()
{
    LOG(debug) << "GBndLib::importIndexBuffer" ; 

    NPY<unsigned int>* ibuf = getIndexBuffer();
    importUint4Buffer(m_bnd, ibuf );
}


void GBndLib::init()
{
    assert(UNSET == GItemList::UNSET);
}

bool GBndLib::contains(const char* spec, bool flip)
{
    guint4 bnd = parse(spec, flip);
    return contains(bnd);
}

guint4 GBndLib::parse( const char* spec, bool flip)
{
    std::vector<std::string> elem ;
    boost::split(elem, spec, boost::is_any_of("/"));

    unsigned int nelem = elem.size();
    assert(nelem == 4);

    const char* omat_ = elem[0].c_str() ;
    const char* osur_ = elem[1].c_str() ;
    const char* isur_ = elem[2].c_str() ;
    const char* imat_ = elem[3].c_str() ;

    unsigned int omat = m_mlib->getIndex(omat_) ;
    unsigned int osur = m_slib->getIndex(osur_) ;
    unsigned int isur = m_slib->getIndex(isur_) ;
    unsigned int imat = m_mlib->getIndex(imat_) ;

    return flip ? 
              guint4(imat, isur, osur, omat)
                :
              guint4(omat, osur, isur, imat) 
                ;   
}



unsigned int GBndLib::addBoundary( const char* spec, bool flip)
{
    guint4 bnd = parse(spec, flip);
    add(bnd);
    return index(bnd) ; 
}

unsigned int GBndLib::addBoundary( const char* omat, const char* osur, const char* isur, const char* imat)
{
    guint4 bnd = add(omat, osur, isur, imat);
    return index(bnd) ; 
}

guint4 GBndLib::add( const char* spec, bool flip)
{
    guint4 bnd = parse(spec, flip);
    add(bnd);
    return bnd ; 
}

guint4 GBndLib::add( const char* omat_ , const char* osur_, const char* isur_, const char* imat_ )
{
    unsigned int omat = m_mlib->getIndex(omat_) ;
    unsigned int osur = m_slib->getIndex(osur_) ;
    unsigned int isur = m_slib->getIndex(isur_) ;
    unsigned int imat = m_mlib->getIndex(imat_) ;
    return add(omat, osur, isur, imat);
}

guint4 GBndLib::add( unsigned int omat , unsigned int osur, unsigned int isur, unsigned int imat )
{
    guint4 bnd = guint4(omat, osur, isur, imat);
    add(bnd);
    return bnd ; 
}

void GBndLib::add(const guint4& bnd)
{
    if(!contains(bnd)) m_bnd.push_back(bnd);
}

bool GBndLib::contains(const guint4& bnd)
{
    typedef std::vector<guint4> G ;  
    G::const_iterator b = m_bnd.begin() ;
    G::const_iterator e = m_bnd.end() ;
    G::const_iterator i = std::find(b, e, bnd) ;
    return i != e ;
}

unsigned int GBndLib::index(const guint4& bnd)
{
    typedef std::vector<guint4> G ;  
    G::const_iterator b = m_bnd.begin() ;
    G::const_iterator e = m_bnd.end() ;
    G::const_iterator i = std::find(b, e, bnd) ;
    return i == e ? UNSET : std::distance(b, i) ; 
}




std::string GBndLib::description(const guint4& bnd)
{
    unsigned int idx = index(bnd) ;
    std::string tag = idx == UNSET ? "-" : boost::lexical_cast<std::string>(idx) ; 

    std::stringstream ss ; 
    ss 
       << " ("   << std::setw(3) << tag << ")" 
       << " om:" << std::setw(25) << m_mlib->getName(bnd[OMAT]) 
       << " os:" << std::setw(25) << (bnd[OSUR] == UNSET ? "" : m_slib->getName(bnd[OSUR]))  
       << " is:" << std::setw(25) << (bnd[ISUR] == UNSET ? "" : m_slib->getName(bnd[ISUR])) 
       << " im:" << std::setw(25) << m_mlib->getName(bnd[IMAT]) 
       ;
    return ss.str();
}

std::string GBndLib::shortname(unsigned int boundary)
{
    guint4 bnd = getBnd(boundary);
    return shortname(bnd);
}


std::string GBndLib::shortname(const guint4& bnd)
{
    std::stringstream ss ; 
    ss 
       << m_mlib->getName(bnd[OMAT]) 
       << "/"
       << (bnd[OSUR] == UNSET ? "" : m_slib->getName(bnd[OSUR])) 
       << "/" 
       << (bnd[ISUR] == UNSET ? "" : m_slib->getName(bnd[ISUR]))  
       << "/" 
       << m_mlib->getName(bnd[IMAT]) 
       ;
    return ss.str();
}



guint4 GBndLib::getBnd(unsigned int boundary)
{
    unsigned int ni = getNumBnd();
    assert(boundary < ni);
    const guint4& bnd = m_bnd[boundary];
    return bnd ;  
}

unsigned int GBndLib::getInnerMaterial(unsigned int boundary)
{
    guint4 bnd = getBnd(boundary);
    return bnd[IMAT] ; 
}
unsigned int GBndLib::getOuterMaterial(unsigned int boundary)
{
    guint4 bnd = getBnd(boundary);
    return bnd[OMAT] ; 
}
unsigned int GBndLib::getInnerSurface(unsigned int boundary)
{
    guint4 bnd = getBnd(boundary);
    return bnd[ISUR] ; 
}
unsigned int GBndLib::getOuterSurface(unsigned int boundary)
{
    guint4 bnd = getBnd(boundary);
    return bnd[OSUR] ; 
}



const char* GBndLib::getOuterMaterialName(unsigned int boundary)
{
    unsigned int omat = getOuterMaterial(boundary);
    return m_mlib->getName(omat);
}
const char* GBndLib::getInnerMaterialName(unsigned int boundary)
{
    unsigned int imat = getInnerMaterial(boundary);
    return m_mlib->getName(imat);
}



const char* GBndLib::getOuterSurfaceName(unsigned int boundary)
{
    unsigned int osur = getOuterSurface(boundary);
    return m_slib->getName(osur);
}
const char* GBndLib::getInnerSurfaceName(unsigned int boundary)
{
    unsigned int isur = getInnerSurface(boundary);
    return m_slib->getName(isur);
}






const char* GBndLib::getOuterMaterialName(const char* spec)
{
    unsigned int boundary = addBoundary(spec);
    return getOuterMaterialName(boundary);
}
const char* GBndLib::getInnerMaterialName(const char* spec)
{
    unsigned int boundary = addBoundary(spec);
    return getInnerMaterialName(boundary);
}
const char* GBndLib::getOuterSurfaceName(const char* spec)
{
    unsigned int boundary = addBoundary(spec);
    return getOuterSurfaceName(boundary);
}
const char* GBndLib::getInnerSurfaceName(const char* spec)
{
    unsigned int boundary = addBoundary(spec);
    return getInnerSurfaceName(boundary);
}







GItemList* GBndLib::createNames()
{
    unsigned int ni = getNumBnd();
    GItemList* names = new GItemList(getType());
    for(unsigned int i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;
        names->add(shortname(bnd).c_str()); 
    }
    return names ; 
}



void GBndLib::dumpMaterialLineMap(std::map<std::string, unsigned int>& msu, const char* msg)
{ 
    LOG(debug) << msg ; 
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::const_iterator it = msu.begin() ; it != msu.end() ; it++)
        LOG(debug) << std::setw(5) << it->second 
                   << std::setw(30) << it->first 
                   ;
}

void GBndLib::fillMaterialLineMap( std::map<std::string, unsigned int>& msu)
{
    // first occurence of a material within the boundaries
    // has its material line recorded in the MaterialLineMap

    for(unsigned int i=0 ; i < getNumBnd() ; i++)    
    {
        const guint4& bnd = m_bnd[i] ;
        const char* omat = m_mlib->getName(bnd[OMAT]);
        const char* imat = m_mlib->getName(bnd[IMAT]);
        assert(imat && omat);
        if(msu.count(imat) == 0) msu[imat] = getLine(i, 0) ;
        if(msu.count(omat) == 0) msu[omat] = getLine(i, 1) ;
    }
    dumpMaterialLineMap(msu, "GBndLib::fillMaterialLineMap");
}




unsigned int GBndLib::getMaterialLine(const char* shortname)
{
    unsigned int ni = getNumBnd();
    for(unsigned int i=0 ; i < ni ; i++)    
    {
        const guint4& bnd = m_bnd[i] ;
        const char* omat = m_mlib->getName(bnd[OMAT]);
        const char* imat = m_mlib->getName(bnd[IMAT]);
        if(strncmp(imat, shortname, strlen(shortname))==0) return getLine(i, 0);
        if(strncmp(omat, shortname, strlen(shortname))==0) return getLine(i, 1);
    }
    return 0 ;
}
unsigned int GBndLib::getLine(unsigned int ibnd, unsigned int iquad)
{
    assert(iquad < NUM_QUAD);
    return ibnd*NUM_QUAD + iquad ;   
}
unsigned int GBndLib::getLineMin()
{
    unsigned int lineMin = getLine(0, 0);
    return lineMin ; 
}
unsigned int GBndLib::getLineMax()
{
    unsigned int numBnd = getNumBnd() ; 
    unsigned int lineMax = getLine(numBnd - 1, NUM_QUAD-1);
    return lineMax ; 
}

NPY<float>* GBndLib::createBuffer()
{
    NPY<float>* mat = m_mlib->getBuffer();
    NPY<float>* sur = m_slib->getBuffer();

    unsigned int ni = getNumBnd();
    unsigned int nj = NUM_QUAD ;       // im-om-is-os
    unsigned int nk = DOMAIN_LENGTH ; 
    unsigned int nl = NUM_PROP ;       // 4 interweaved props

    assert(nk = getStandardDomainLength()) ;
    assert(mat->getShape(1) == sur->getShape(1) && sur->getShape(1) == nk );
    assert(mat->getShape(2) == sur->getShape(2) && sur->getShape(2) == nl );

    NPY<float>* wav = NPY<float>::make( ni, nj, nk, nl ) ;
    wav->fill(-1.0f);   // match GBoundaryLib::SURFACE_UNSET

    float* mdat = mat->getValues();
    float* sdat = sur->getValues();
    float* wdat = wav->getValues();

    for(unsigned int i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;
        for(unsigned int j=0 ; j < nj ; j++)  // over imat/omat/isur/osur
        {
            unsigned int wof = nj*nk*nl*i + nk*nl*j ;

            if(j == IMAT || j == OMAT)    
            {
                unsigned int midx = bnd[j] ;
                assert(midx != UNSET);
                unsigned int mof = nk*nl*midx ;  
                memcpy( wdat+wof, mdat+mof, sizeof(float)*nk*nl );  
            }
            else if(j == ISUR || j == OSUR)  // isur/osur
            {
                unsigned int sidx = bnd[j] ;
                if(sidx != UNSET)
                {
                    unsigned int sof = nk*nl*sidx ;  
                    memcpy( wdat+wof, sdat+sof, sizeof(float)*nk*nl );  
                }
            }
        } 
    }
    return wav ; 
}


NPY<unsigned int>* GBndLib::createOpticalBuffer()
{
    bool one_based = true ; // surface and material indices 1-based, so 0 can stand for unset
    unsigned int ni = getNumBnd();
    unsigned int nj = NUM_QUAD ;    // im-om-is-os
    unsigned int nk = NUM_PROP ;      

    NPY<unsigned int>* optical = NPY<unsigned int>::make( ni, nj, nk) ;
    optical->zero(); 
    unsigned int* odat = optical->getValues();

    for(unsigned int i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;
        for(unsigned int j=0 ; j < nj ; j++)  // over imat/omat/isur/osur
        {
            unsigned int offset = nj*nk*i+nk*j ;
            if(j == IMAT || j == OMAT)    
            {
                unsigned int midx = bnd[j] ;
                assert(midx != UNSET);
                odat[offset+0] = one_based ? midx + 1 : midx  ; 
                odat[offset+1] = 0u ; 
                odat[offset+2] = 0u ; 
                odat[offset+3] = 0u ; 
            }
            else if(j == ISUR || j == OSUR)  
            {
                unsigned int sidx = bnd[j] ;
                if(sidx != UNSET)
                {
      
                    guint4 os = m_slib->getOpticalSurface(sidx) ;
                    odat[offset+0] = one_based ? sidx + 1 : sidx  ; 
                 // TODO: enum these
                    odat[offset+1] = os.y ; 
                    odat[offset+2] = os.z ; 
                    odat[offset+3] = os.w ; 
                }
            }
        } 
    }
    return optical ; 

    // bnd indices originate during cache creation from the AssimpGGeo call of GBndLib::add with the shortnames 
    // this means that for order preferences ~/.opticks/GMaterialLib/order.json and ~/.opticks/GSurfaceLib/order.json
    // to be reflected need to rebuild the cache with ggv -G first 
}


void GBndLib::import()
{
    LOG(debug) << "GBndLib::import" ; 
}
void GBndLib::sort()
{
    LOG(debug) << "GBndLib::sort" ; 
}
void GBndLib::defineDefaults(GPropertyMap<float>* defaults)
{
    LOG(debug) << "GBndLib::defineDefaults" ; 
}
 
void GBndLib::Summary(const char* msg)
{
    unsigned int ni = getNumBnd();
    LOG(info) << msg << " NumBnd:" << ni ; 
}

void GBndLib::dump(const char* msg)
{
    LOG(info) << msg ; 
    unsigned int ni = getNumBnd();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        const guint4& bnd = m_bnd[i] ;
        //bnd.Summary(msg);
        std::cout << description(bnd) << std::endl ; 
    } 
}



void GBndLib::dumpBoundaries(std::vector<unsigned int>& boundaries, const char* msg)
{
    LOG(info) << msg ; 
    unsigned int nb = boundaries.size() ;
    for(unsigned int i=0 ; i < nb ; i++)
    {
        unsigned int boundary = boundaries[i];
        guint4 bnd = getBnd(boundary);
        std::cout << std::setw(3) << i 
                  << std::setw(5) << boundary 
                  << std::setw(20) << bnd.description() 
                  << " : " 
                  << description(bnd) << std::endl ; 
    }
}



 
