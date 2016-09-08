
#include <climits>
#include <cassert>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "NGLM.hpp"
#include "NPY.hpp"
#include "Opticks.hh"

#include "GVector.hh"
#include "GItemList.hh"
#include "GAry.hh"
#include "GDomain.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"

#include "PLOG.hh"

void GBndLib::save()
{
    // NB bndlib exists for deferred boundary buffer creation,  
    //
    // NB only the Index buffer (persisted m_bnd vector of guint4 with omat-osur-isur-imat indices)
    //    is "-G" saved into the geocache
    //
    //    The float and optical buffers are regarded as dynamic 
    //    (although they are still persisted for debugging/record keeping)
    // 

    saveIndexBuffer();  
}

GBndLib* GBndLib::load(Opticks* cache, bool constituents)
{
    GBndLib* blib = new GBndLib(cache);

    LOG(trace) << "GBndLib::load" ; 

    blib->loadIndexBuffer();

    LOG(trace) << "GBndLib::load indexBuffer loaded" ; 
    blib->importIndexBuffer();

    if(constituents)
    {
        GMaterialLib* mlib = GMaterialLib::load(cache);
        GSurfaceLib* slib = GSurfaceLib::load(cache);
        blib->setMaterialLib(mlib);
        blib->setSurfaceLib(slib);
    }

    LOG(trace) << "GBndLib::load DONE" ; 

    return blib ; 
}

void GBndLib::loadIndexBuffer()
{
    LOG(trace) << "GBndLib::loadIndexBuffer" ; 

    std::string dir = getCacheDir(); 
    std::string name = getBufferName("Index");

    LOG(trace) << "GBndLib::loadIndexBuffer" 
               << " dir " << dir
               << " name " << name 
                ; 


    NPY<unsigned int>* indexBuf = NPY<unsigned int>::load(dir.c_str(), name.c_str()); 


    LOG(trace) << "GBndLib::loadIndexBuffer" 
               << " indexBuf " << indexBuf
               ;


    setIndexBuffer(indexBuf); 

    if(indexBuf == NULL) 
    {
        LOG(warning) << "GBndLib::loadIndexBuffer setting invalid " ; 
        setValid(false);
    }
    else
    {
        LOG(debug) << "GBndLib::loadIndexBuffer"
                  << " shape " << indexBuf->getShapeString() ;
    }
}

void GBndLib::saveIndexBuffer()
{
    NPY<unsigned int>* indexBuf = createIndexBuffer();
    setIndexBuffer(indexBuf);

    saveToCache(indexBuf, "Index") ; 
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

    LOG(debug) << "GBndLib::createDynamicBuffers" 
              << " buf " << buf->getShapeString()
              << " optical_buffer  " << optical_buffer->getShapeString()
               ;

    // declare closed here ? 

}


NPY<unsigned int>* GBndLib::createIndexBuffer()
{
    NPY<unsigned int>* idx = NULL ;  
    if(m_bnd.size() > 0)
    { 
       idx = createUint4Buffer(m_bnd);
    } 
    else
    {
        LOG(error) << "GBndLib::createIndexBuffer"
                   << " BUT SIZE IS ZERO "
                   ;
  
    } 
    return idx ;
}

void GBndLib::importIndexBuffer()
{
    LOG(trace) << "GBndLib::importIndexBuffer" ; 
    NPY<unsigned int>* ibuf = getIndexBuffer();

    if(ibuf == NULL)
    {
         LOG(warning) << "GBndLib::importIndexBuffer NULL buffer setting invalid" ; 
         setValid(false);
         return ;
    }

    LOG(debug) << "GBndLib::importIndexBuffer BEFORE IMPORT" 
              << " ibuf " << ibuf->getShapeString()
              << " m_bnd.size() " << m_bnd.size()
             ; 

    importUint4Buffer(m_bnd, ibuf );

    LOG(debug) << "GBndLib::importIndexBuffer AFTER IMPORT" 
              << " ibuf " << ibuf->getShapeString()
              << " m_bnd.size() " << m_bnd.size()
             ; 
}


GBndLib::GBndLib(Opticks* cache) 
   :
    GPropertyLib(cache, "GBndLib"),
    m_mlib(NULL),
    m_slib(NULL),
    m_index_buffer(NULL),
    m_optical_buffer(NULL)
{
    init();
}

void GBndLib::setMaterialLib(GMaterialLib* mlib)
{
    m_mlib = mlib ;  
}
void GBndLib::setSurfaceLib(GSurfaceLib* slib)
{
    m_slib = slib ;  
}
GMaterialLib* GBndLib::getMaterialLib()
{
    return m_mlib ; 
}
GSurfaceLib* GBndLib::getSurfaceLib()
{
    return m_slib ; 
}


unsigned int GBndLib::getNumBnd()
{
    return m_bnd.size() ; 
}

NPY<unsigned int>* GBndLib::getIndexBuffer()
{
    return m_index_buffer ;
}

bool GBndLib::hasIndexBuffer()
{
    return m_index_buffer != NULL ; 
}

void GBndLib::setIndexBuffer(NPY<unsigned int>* index_buffer)
{
    m_index_buffer = index_buffer ;
}

NPY<unsigned int>* GBndLib::getOpticalBuffer()
{
    return m_optical_buffer ;
}
void GBndLib::setOpticalBuffer(NPY<unsigned int>* optical_buffer)
{
    m_optical_buffer = optical_buffer ;
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


    if(nelem != 4)
    {      
        LOG(fatal) << "GBndLib::parse"
                   << " bad boundary spec, expecting 4 elements"
                   << " spec " << spec
                   << " nelem " << nelem
                   ;

    }

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
       << " om:" << std::setw(25) << (bnd[OMAT] == UNSET ? "OMAT-unset-ERROR" : m_mlib->getName(bnd[OMAT]))
       << " os:" << std::setw(25) << (bnd[OSUR] == UNSET ? "" : m_slib->getName(bnd[OSUR]))  
       << " is:" << std::setw(25) << (bnd[ISUR] == UNSET ? "" : m_slib->getName(bnd[ISUR])) 
       << " im:" << std::setw(25) << (bnd[IMAT] == UNSET ? "IMAT-unset-ERROR" : m_mlib->getName(bnd[IMAT])) 
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
       << (bnd[OMAT] == UNSET ? "OMAT-unset-error" : m_mlib->getName(bnd[OMAT])) 
       << "/"
       << (bnd[OSUR] == UNSET ? "" : m_slib->getName(bnd[OSUR])) 
       << "/" 
       << (bnd[ISUR] == UNSET ? "" : m_slib->getName(bnd[ISUR]))  
       << "/" 
       << (bnd[IMAT] == UNSET ? "IMAT-unset-error" : m_mlib->getName(bnd[IMAT]))
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
        if(msu.count(imat) == 0) msu[imat] = getLine(i, IMAT) ; // incorrectly 0 until 2016/3/13
        if(msu.count(omat) == 0) msu[omat] = getLine(i, OMAT) ; // incorrectly 1 until 2016/3/13
    }
    dumpMaterialLineMap(msu, "GBndLib::fillMaterialLineMap");
}

const std::map<std::string, unsigned int>& GBndLib::getMaterialLineMap()
{
    if(m_materialLineMap.size() == 0) fillMaterialLineMap(m_materialLineMap) ;
    return m_materialLineMap ;
}
void GBndLib::dumpMaterialLineMap(const char* msg)
{
    dumpMaterialLineMap(m_materialLineMap, msg );
}


unsigned int GBndLib::getMaterialLine(const char* shortname_)
{
    // used by App::loadGenstep for setting material line in TorchStep
    unsigned int ni = getNumBnd();
    unsigned int line = 0 ; 
    for(unsigned int i=0 ; i < ni ; i++)    
    {
        const guint4& bnd = m_bnd[i] ;
        const char* omat = m_mlib->getName(bnd[OMAT]);
        const char* imat = m_mlib->getName(bnd[IMAT]);

        if(strncmp(imat, shortname_, strlen(shortname_))==0)
        { 
            line = getLine(i, IMAT);  // was incorrectly 0 until 2016/3/13
            break ;
        }
        if(strncmp(omat, shortname_, strlen(shortname_))==0) 
        { 
            line=getLine(i, OMAT);  // was incorrectly 1 until 2016/3/13
            break ;
        } 
    }

    LOG(trace) << "GBndLib::getMaterialLine"
              << " shortname_ " << shortname_ 
              << " line " << line 
              ; 

    return line ;
}
unsigned int GBndLib::getLine(unsigned int ibnd, unsigned int imatsur)
{
    assert(imatsur < NUM_MATSUR);  // NUM_MATSUR canonically 4
    return ibnd*NUM_MATSUR + imatsur ;   
}
unsigned int GBndLib::getLineMin()
{
    unsigned int lineMin = getLine(0, 0);
    return lineMin ; 
}
unsigned int GBndLib::getLineMax()
{
    unsigned int numBnd = getNumBnd() ; 
    unsigned int lineMax = getLine(numBnd - 1, NUM_MATSUR-1);   
    assert(lineMax == numBnd*NUM_MATSUR - 1 );
    return lineMax ; 
}

NPY<float>* GBndLib::createBuffer()
{
    return createBufferForTex2d() ;
}

NPY<float>* GBndLib::createBufferForTex2d()
{
    /*

    GBndLib float buffer is a memcpy zip of the MaterialLib and SurfaceLib buffers
    pulling together data based on the indices for the materials and surfaces 
    from the m_bnd guint4 buffer

    Typical dimensions : (128, 4, 2, 39, 4)   

               128 : boundaries, 
                 4 : mat-or-sur for each boundary  
                 2 : payload-categries corresponding to NUM_FLOAT4
                39 : wavelength samples
                 4 : float4-values

     The only dimension that can easily be extended is the middle payload-categories one, 
     the low side is constrained by layout needed to OptiX tex2d<float4> as this 
     buffer is memcpy into the texture buffer
     high side is constained by not wanting to change texture line indices 

    */

    LOG(trace) << "GBndLib::createBufferForTex2d" ;

    NPY<float>* mat = m_mlib->getBuffer();
    NPY<float>* sur = m_slib->getBuffer();

    LOG(trace) << "GBndLib::createBufferForTex2d" 
               << " mat " << mat 
               << " sur " << sur
               ; 

    if(mat == NULL || sur == NULL)
    {
        LOG(error) << "GBndLib::createBufferForTex2d" 
                   << " NULL BUFFERS "
                   << " mat " << mat 
                   << " sur " << sur
                   ; 
        return NULL ; 
    }


    unsigned int ni = getNumBnd();
    unsigned int nj = NUM_MATSUR ;    // om-os-is-im

    // the klm matches the Materials and Surface buffer layouts, so can memcpy in 
    unsigned int nk = NUM_FLOAT4 ;    
    unsigned int nl = Opticks::DOMAIN_LENGTH ; 
    unsigned int nm = 4 ; 


    assert(nl == getStandardDomainLength()) ;
    assert(mat->getShape(1) == sur->getShape(1) && sur->getShape(1) == nk );
    assert(mat->getShape(2) == sur->getShape(2) && sur->getShape(2) == nl );

    NPY<float>* wav = NPY<float>::make( ni, nj, nk, nl, nm) ;
    wav->fill( GSurfaceLib::SURFACE_UNSET ); 

    LOG(debug) << "GBndLib::createBufferForTex2d"
               << " mat " << mat->getShapeString()
               << " sur " << sur->getShapeString()
               << " wav " << wav->getShapeString()
               ; 

    float* mdat = mat->getValues();
    float* sdat = sur->getValues();
    float* wdat = wav->getValues(); // destination

    for(unsigned int i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;
        for(unsigned int j=0 ; j < nj ; j++)     // over imat/omat/isur/osur species
        {
            unsigned int wof = nj*nk*nl*nm*i + nk*nl*nm*j ;

            if(j == IMAT || j == OMAT)    
            {
                unsigned int midx = bnd[j] ;
                if(midx != UNSET)
                { 
                    unsigned int mof = nk*nl*nm*midx ; 
                    memcpy( wdat+wof, mdat+mof, sizeof(float)*nk*nl*nm );  
                }
                else
                {
                    LOG(warning) << "GBndLib::createBufferForTex2d"
                                 << " ERROR IMAT/OMAT with UNSET MATERIAL "
                                 << " i " << i  
                                 << " j " << j 
                                 ; 
                }
            }
            else if(j == ISUR || j == OSUR) 
            {
                unsigned int sidx = bnd[j] ;
                if(sidx != UNSET)
                {
                    unsigned int sof = nk*nl*nm*sidx ;  
                    memcpy( wdat+wof, sdat+sof, sizeof(float)*nk*nl*nm );  
                }
            }

         }     // j
    }          // i
    return wav ; 
}




NPY<float>* GBndLib::createBufferOld()
{

    NPY<float>* mat = m_mlib->getBuffer();
    NPY<float>* sur = m_slib->getBuffer();

    unsigned int ni = getNumBnd();
    unsigned int nj = NUM_MATSUR ;       // im-om-is-os 
    unsigned int nk = Opticks::DOMAIN_LENGTH ; 
    unsigned int nl = NUM_PROP ;       // 4/8 interweaved props

    assert( nl == 4 || nl == 8);

    assert(nk == getStandardDomainLength()) ;
    assert(mat->getShape(1) == sur->getShape(1) && sur->getShape(1) == nk );
    assert(mat->getShape(2) == sur->getShape(2) && sur->getShape(2) == nl );

    NPY<float>* wav = NPY<float>::make( ni, nj, nk, nl ) ;
    wav->fill( GSurfaceLib::SURFACE_UNSET ); 

    float* mdat = mat->getValues();
    float* sdat = sur->getValues();
    float* wdat = wav->getValues(); // destination

    for(unsigned int i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;
        for(unsigned int j=0 ; j < nj ; j++)  // over imat/omat/isur/osur species
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
    /*
    m_bnd vector of guint4 is persisted in the optical buffer

    Optical buffer contains omat-osur-isur-imat info, 
    for materials just the material one based index, for 
    surfaces the one based surface index and other optical 
    surface parameters. 

    */
 
    bool one_based = true ; // surface and material indices 1-based, so 0 can stand for unset
    unsigned int ni = getNumBnd();
    unsigned int nj = NUM_MATSUR ;    // om-os-is-im
    unsigned int nk = 4 ;           // THIS 4 IS NOT RELATED TO NUM_PROP

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
    // does nothing as GBndLib needs dynamic buffers
}
void GBndLib::sort()
{
    LOG(debug) << "GBndLib::sort" ; 
}
void GBndLib::defineDefaults(GPropertyMap<float>* /*defaults*/)
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
    LOG(info) << msg 
              << " ni " << ni 
               ; 

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



 
