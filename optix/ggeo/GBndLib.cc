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

GBndLib* GBndLib::load(GCache* cache)
{
    GBndLib* blib = new GBndLib(cache);
    blib->loadIndexBuffer();
    blib->importIndexBuffer();
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
    NPY<unsigned int>* ibuf = createIndexBuffer();
    saveToCache(ibuf, "Index") ; 
    setIndexBuffer(ibuf);
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
    assert(nelem > 1);

    const char* imat_ = nelem > 0 ? elem[0].c_str() : NULL ;
    const char* omat_ = nelem > 1 ? elem[1].c_str() : NULL ;
    const char* isur_ = nelem > 2 ? elem[2].c_str() : NULL ;
    const char* osur_ = nelem > 3 ? elem[3].c_str() : NULL ;

    unsigned int imat = m_mlib->getIndex(imat_) ;
    unsigned int omat = m_mlib->getIndex(omat_) ;
    unsigned int isur = m_slib->getIndex(isur_) ;
    unsigned int osur = m_slib->getIndex(osur_) ;

    return flip ? 
              guint4(omat, imat, osur, isur)
                :
              guint4(imat, omat, isur, osur) 
                ;   
}

guint4 GBndLib::add( const char* spec, bool flip)
{
    guint4 bnd = parse(spec, flip);
    add(bnd);
    return bnd ; 
}

guint4 GBndLib::add( const char* imat_ , const char* omat_, const char* isur_, const char* osur_ )
{
    unsigned int imat = m_mlib->getIndex(imat_) ;
    unsigned int omat = m_mlib->getIndex(omat_) ;
    unsigned int isur = m_slib->getIndex(isur_) ;
    unsigned int osur = m_slib->getIndex(osur_) ;
    return add(imat, omat, isur, osur);
}

guint4 GBndLib::add( unsigned int imat , unsigned int omat, unsigned int isur, unsigned int osur )
{
    guint4 bnd = guint4(imat, omat, isur, osur);
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
       << " im:" << std::setw(25) << m_mlib->getName(bnd.x) 
       << " om:" << std::setw(25) << m_mlib->getName(bnd.y) 
       << " is:" << std::setw(25) << (bnd.z == UNSET ? "" : m_slib->getName(bnd.z)) 
       << " os:" << std::setw(25) << (bnd.w == UNSET ? "" : m_slib->getName(bnd.w))  
       ;
    return ss.str();
}

std::string GBndLib::shortname(const guint4& bnd)
{
    std::stringstream ss ; 
    ss 
       << m_mlib->getName(bnd.x) 
       << "/"
       << m_mlib->getName(bnd.y) 
       << "/" 
       << (bnd.z == UNSET ? "" : m_slib->getName(bnd.z)) 
       << "/" 
       << (bnd.w == UNSET ? "" : m_slib->getName(bnd.w))  
       ;
    return ss.str();
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
    wav->zero(); 

    float* mdat = mat->getValues();
    float* sdat = sur->getValues();
    float* wdat = wav->getValues();

    for(unsigned int i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;
        for(unsigned int j=0 ; j < nj ; j++)  // over imat/omat/isur/osur
        {
            unsigned int wof = nj*nk*nl*i + nk*nl*j ;

            if(j == 0 || j == 1)      // imat/omat
            {
                unsigned int midx = bnd[j] ;
                assert(midx != UNSET);
                unsigned int mof = nk*nl*midx ;  
                memcpy( wdat+wof, mdat+mof, sizeof(float)*nk*nl );  
            }
            else if(j == 2 || j == 3)  // isur/osur
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


NPY<unsigned int>* GBndLib::createIndexBuffer()
{
    return createUint4Buffer(m_bnd);

    /*
    unsigned int ni = getNumBnd();
    unsigned int nj = 4 ;  
    NPY<unsigned int>* ibuf = NPY<unsigned int>::make( ni, nj) ;
    ibuf->zero();

    unsigned int* idat = ibuf->getValues();

    for(unsigned int i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;
        for(unsigned int j=0 ; j < nj ; j++)  // over imat/omat/isur/osur
           idat[i*nj+j] = bnd[j] ;  
    }
    return ibuf ; 
    */

}

void GBndLib::importIndexBuffer()
{
    LOG(info) << "GBndLib::importIndexBuffer" ; 

    NPY<unsigned int>* ibuf = getIndexBuffer();
    importUint4Buffer(m_bnd, ibuf );

/*

    unsigned int* idat = ibuf->getValues();

    unsigned int ni = ibuf->getShape(0);
    unsigned int nj = ibuf->getShape(1);
    assert(nj == 4); 

    for(unsigned int i=0 ; i < ni ; i++)
    {
        guint4 bnd(UNSET,UNSET,UNSET,UNSET);

        bnd.x = idat[i*nj+0] ;
        bnd.y = idat[i*nj+1] ;
        bnd.z = idat[i*nj+2] ;
        bnd.w = idat[i*nj+3] ;

        add(bnd);
    }
*/

}





void GBndLib::import()
{
    LOG(info) << "GBndLib::import" ; 
}

void GBndLib::defineDefaults(GPropertyMap<float>* defaults)
{
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

 
