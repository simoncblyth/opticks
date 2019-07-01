
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
#include "GBnd.hh"
#include "GBndLib.hh"

#include "PLOG.hh"


const plog::Severity GBndLib::LEVEL = PLOG::EnvLevel("GBndLib", "DEBUG") ; 

const GBndLib* GBndLib::INSTANCE = NULL ; 
const GBndLib* GBndLib::GetInstance(){ return INSTANCE ; }

unsigned GBndLib::MaterialIndexFromLine( unsigned line ) 
{
    assert( INSTANCE ) ; 
    return INSTANCE->getMaterialIndexFromLine(line) ;
}


void GBndLib::save()
{
    saveIndexBuffer();  
}

GBndLib* GBndLib::load(Opticks* ok, bool constituents)
{
    LOG(LEVEL) << "[" ; 
    GBndLib* blib = new GBndLib(ok);

    LOG(verbose) ;

    blib->loadIndexBuffer();

    LOG(verbose) << "indexBuffer loaded" ; 
    blib->importIndexBuffer();


    if(constituents)
    {
        GMaterialLib* mlib = GMaterialLib::load(ok);
        GSurfaceLib* slib = GSurfaceLib::load(ok);
        GDomain<float>* finedom = ok->hasOpt("finebndtex") 
                            ?
                                mlib->getStandardDomain()->makeInterpolationDomain(Opticks::FINE_DOMAIN_STEP) 
                            :
                                NULL 
                            ;

        //assert(0); 

        if(finedom)
        {
            LOG(warning) << "--finebndtex option triggers interpolation of material and surface props "  ;
            GMaterialLib* mlib2 = new GMaterialLib(mlib, finedom );    
            GSurfaceLib* slib2 = new GSurfaceLib(slib, finedom );    

            mlib2->setBuffer(mlib2->createBuffer());
            slib2->setBuffer(slib2->createBuffer());

            blib->setStandardDomain(finedom);
            blib->setMaterialLib(mlib2);
            blib->setSurfaceLib(slib2);

            blib->setBuffer(blib->createBuffer()); 
        }
        else
        {
            blib->setMaterialLib(mlib);
            blib->setSurfaceLib(slib);
        } 
    }

    LOG(LEVEL) << "]" ; 

    return blib ; 
}

void GBndLib::loadIndexBuffer()
{
    LOG(LEVEL) ; 

    std::string dir = getCacheDir(); 
    std::string name = getBufferName("Index");

    LOG(verbose) << "GBndLib::loadIndexBuffer" 
               << " dir " << dir
               << " name " << name 
                ; 


    NPY<unsigned int>* indexBuf = NPY<unsigned int>::load(dir.c_str(), name.c_str()); 


    LOG(verbose) << "GBndLib::loadIndexBuffer" 
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




void GBndLib::dumpOpticalBuffer() const 
{
    LOG(error) << "." ; 


    if( m_optical_buffer )
    {
        m_optical_buffer->dump("dumpOpticalBuffer");     
    }

}



NMeta* GBndLib::createMeta()
{
    return NULL ; 
}



void GBndLib::createDynamicBuffers()
{
    // there is not much difference between this and doing a close ??? 

    GItemList* names = createNames();     // added Aug 21, 2018
    setNames(names); 

    NPY<float>* buf = createBuffer();
    setBuffer(buf);

    NPY<unsigned int>* optical_buffer = createOpticalBuffer();
    setOpticalBuffer(optical_buffer);


    LOG(debug) << "GBndLib::createDynamicBuffers" 
              << " buf " << ( buf ? buf->getShapeString() : "NULL" )
              << " optical_buffer  " << ( optical_buffer ? optical_buffer->getShapeString() : "NULL" )
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
    LOG(verbose) << "GBndLib::importIndexBuffer" ; 
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


unsigned int GBndLib::getNumBnd() const
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





GBndLib::GBndLib(Opticks* ok, GMaterialLib* mlib, GSurfaceLib* slib)
   :
    GPropertyLib(ok, "GBndLib"),
    m_dbgbnd(ok->isDbgBnd()),
    m_mlib(mlib),
    m_slib(slib),
    m_index_buffer(NULL),
    m_optical_buffer(NULL)
{
    init();
}

GBndLib::GBndLib(Opticks* ok) 
   :
    GPropertyLib(ok, "GBndLib"),
    m_dbgbnd(ok->isDbgBnd()),
    m_mlib(NULL),
    m_slib(NULL),
    m_index_buffer(NULL),
    m_optical_buffer(NULL)
{
    init();
}


void GBndLib::init()
{
    INSTANCE=this ; 
    if(m_dbgbnd)
    {
        LOG(fatal) << "[--dbgbnd] " << m_dbgbnd ; 
    }
    assert(UNSET == GItemList::UNSET);
}


void GBndLib::closeConstituents()
{
    LOG(LEVEL) ; 
    if(m_mlib) m_mlib->close(); 
    if(m_slib) m_slib->close(); 
}


bool GBndLib::isDbgBnd() const 
{
    return m_dbgbnd ; 
}



bool GBndLib::contains(const char* spec, bool flip) const
{
    guint4 bnd = parse(spec, flip);
    return contains(bnd);
}



guint4 GBndLib::parse( const char* spec, bool flip) const
{
    GBnd b(spec, flip, m_mlib, m_slib, m_dbgbnd);
    return guint4( b.omat, b.osur, b.isur, b.imat ) ; 
}




unsigned GBndLib::addBoundary( const char* spec, bool flip)
{
    // used by GMaker::makeFromCSG
    // hmm: when need to create surf, need the volnames ?

    GBnd b(spec, flip, m_mlib, m_slib, m_dbgbnd);


    guint4 bnd = guint4( b.omat, b.osur, b.isur, b.imat );
    add(bnd);
    unsigned boundary = index(bnd) ;

    if(m_dbgbnd)
    {
       LOG(info) << "[--dbgbnd] "
                 << " spec " << spec 
                 << " flip " << flip 
                 << " bnd " << bnd.description()
                 << " boundary " << boundary
                 ;
    }
    return boundary ; 
}
unsigned GBndLib::addBoundary( const char* omat, const char* osur, const char* isur, const char* imat)
{
   /*
    LOG(error) 
           << " omat " << ( omat ? omat : "-" )  
           << " osur " << ( osur ? osur : "-" ) 
           << " isur " << ( isur ? isur : "-" ) 
           << " imat " << ( imat ? imat : "-" )  
           ;
    */

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
    unsigned omat = m_mlib->getIndex(omat_) ;   // these are 0-based indices or UINT_MAX when no match
    unsigned osur = m_slib->getIndex(osur_) ;
    unsigned isur = m_slib->getIndex(isur_) ;
    unsigned imat = m_mlib->getIndex(imat_) ;
    return add(omat, osur, isur, imat);
}
guint4 GBndLib::add( unsigned omat , unsigned osur, unsigned isur, unsigned imat )
{
    guint4 bnd = guint4(omat, osur, isur, imat);
    add(bnd);
    return bnd ; 
}

void GBndLib::add(const guint4& bnd)  // all the adders invoke this
{
    if(!contains(bnd)) m_bnd.push_back(bnd);
}


bool GBndLib::contains(const guint4& bnd) const 
{
    typedef std::vector<guint4> G ;  
    G::const_iterator b = m_bnd.begin() ;
    G::const_iterator e = m_bnd.end() ;
    G::const_iterator i = std::find(b, e, bnd) ;
    return i != e ;
}

unsigned int GBndLib::index(const guint4& bnd) const 
{
    typedef std::vector<guint4> G ;  
    G::const_iterator b = m_bnd.begin() ;
    G::const_iterator e = m_bnd.end() ;
    G::const_iterator i = std::find(b, e, bnd) ;
    return i == e ? UNSET : std::distance(b, i) ; 
}

std::string GBndLib::description(const guint4& bnd) const
{
    unsigned int idx = index(bnd) ;
    std::string tag = idx == UNSET ? "-" : boost::lexical_cast<std::string>(idx) ; 

    unsigned omat = bnd[OMAT] ;
    unsigned osur = bnd[OSUR] ;
    unsigned isur = bnd[ISUR] ;
    unsigned imat = bnd[IMAT] ;

    std::stringstream ss ; 
    ss 
       << " ("   << std::setw(3) << tag << ")" 
       << " om:" << std::setw(25) << (omat == UNSET ? "OMAT-unset-ERROR" : m_mlib->getName(omat))
       << " os:" << std::setw(31) << (osur == UNSET ? ""                 : m_slib->getName(osur))  
       << " is:" << std::setw(31) << (isur == UNSET ? ""                 : m_slib->getName(isur)) 
       << " im:" << std::setw(25) << (imat == UNSET ? "IMAT-unset-ERROR" : m_mlib->getName(imat)) 
       << " ("   << std::setw(3) << tag << ")" 
       << "     "
       << " (" 
       << std::setw(2) << omat << ","
       << std::setw(2) << ( osur == UNSET ? -1 : (int)osur ) << ","
       << std::setw(2) << ( isur == UNSET ? -1 : (int)isur ) << ","
       << std::setw(2) << imat 
       << ")" 
       ;
    return ss.str();
}

std::string GBndLib::shortname(unsigned boundary) const 
{
    guint4 bnd = getBnd(boundary);
    return shortname(bnd);
}


std::string GBndLib::shortname(const guint4& bnd) const 
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



guint4 GBndLib::getBnd(unsigned boundary) const
{
    unsigned int ni = getNumBnd();
    assert(boundary < ni);
    const guint4& bnd = m_bnd[boundary];
    return bnd ;  
}

unsigned int GBndLib::getInnerMaterial(unsigned boundary) const 
{
    guint4 bnd = getBnd(boundary);
    return bnd[IMAT] ; 
}
unsigned int GBndLib::getOuterMaterial(unsigned boundary) const 
{
    guint4 bnd = getBnd(boundary);
    return bnd[OMAT] ; 
}
unsigned int GBndLib::getInnerSurface(unsigned boundary) const 
{
    guint4 bnd = getBnd(boundary);
    return bnd[ISUR] ; 
}
unsigned int GBndLib::getOuterSurface(unsigned boundary) const 
{
    guint4 bnd = getBnd(boundary);
    return bnd[OSUR] ; 
}

const char* GBndLib::getOuterMaterialName(unsigned boundary) const
{
    unsigned int omat = getOuterMaterial(boundary);
    return m_mlib->getName(omat);
}
const char* GBndLib::getInnerMaterialName(unsigned boundary) const 
{
    unsigned int imat = getInnerMaterial(boundary);
    return m_mlib->getName(imat);
}

const char* GBndLib::getOuterSurfaceName(unsigned boundary) const
{
    unsigned int osur = getOuterSurface(boundary);
    return m_slib->getName(osur);
}
const char* GBndLib::getInnerSurfaceName(unsigned boundary) const
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
    LOG(info) << msg ; 
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::const_iterator it = msu.begin() ; it != msu.end() ; it++)
        LOG(info) << std::setw(5) << it->second 
                   << std::setw(30) << it->first 
                   ;
}

void GBndLib::fillMaterialLineMap( std::map<std::string, unsigned>& msu)
{
    // first occurence of a material within the boundaries
    // has its material line recorded in the MaterialLineMap

    for(unsigned int i=0 ; i < getNumBnd() ; i++)    
    {
        const guint4& bnd = m_bnd[i] ;
        const char* omat = m_mlib->getName(bnd[OMAT]);
        const char* imat = m_mlib->getName(bnd[IMAT]);
        assert(imat && omat);
        if(msu.count(imat) == 0) msu[imat] = getLine(i, IMAT) ;
        if(msu.count(omat) == 0) msu[omat] = getLine(i, OMAT) ; 
    }

    if(m_ok->isMaterialDbg())
    {
        dumpMaterialLineMap(msu, "GBndLib::fillMaterialLineMap (--materialdbg) ");
    }
}

const std::map<std::string, unsigned int>& GBndLib::getMaterialLineMap()
{
    if(m_materialLineMap.size() == 0) fillMaterialLineMap(m_materialLineMap) ;
    return m_materialLineMap ;
}

void GBndLib::fillMaterialLineMap()
{
    if(m_materialLineMap.size() == 0) fillMaterialLineMap(m_materialLineMap) ;
}

const std::map<std::string, unsigned int>& GBndLib::getMaterialLineMapConst() const
{
    return m_materialLineMap ;
}


void GBndLib::dumpMaterialLineMap(const char* msg)
{
    LOG(info) << "GBndLib::dumpMaterialLineMap" ; 

    if(m_materialLineMap.size() == 0) fillMaterialLineMap(m_materialLineMap) ;


    LOG(info) << "GBndLib::dumpMaterialLineMap" 
              << " m_materialLineMap.size()  " << m_materialLineMap.size() 
              ; 

    dumpMaterialLineMap(m_materialLineMap, msg );
}






unsigned GBndLib::getMaterialLine(const char* shortname_)
{
    // used by App::loadGenstep for setting material line in TorchStep
    unsigned ni = getNumBnd();
    unsigned line = 0 ; 
    for(unsigned i=0 ; i < ni ; i++)    
    {
        const guint4& bnd = m_bnd[i] ;
        const char* omat = m_mlib->getName(bnd[OMAT]);
        const char* imat = m_mlib->getName(bnd[IMAT]);

        if(strncmp(imat, shortname_, strlen(shortname_))==0)
        { 
            line = getLine(i, IMAT);  
            break ;
        }
        if(strncmp(omat, shortname_, strlen(shortname_))==0) 
        { 
            line=getLine(i, OMAT); 
            break ;
        } 
    }

    LOG(verbose) << "GBndLib::getMaterialLine"
              << " shortname_ " << shortname_ 
              << " line " << line 
              ; 

    return line ;
}
unsigned GBndLib::getLine(unsigned ibnd, unsigned imatsur)
{
    assert(imatsur < NUM_MATSUR);  // NUM_MATSUR canonically 4
    return ibnd*NUM_MATSUR + imatsur ;   
}
unsigned GBndLib::getLineMin()
{
    unsigned lineMin = getLine(0, 0);
    return lineMin ; 
}
unsigned int GBndLib::getLineMax()
{
    unsigned numBnd = getNumBnd() ; 
    unsigned lineMax = getLine(numBnd - 1, NUM_MATSUR-1);   
    assert(lineMax == numBnd*NUM_MATSUR - 1 );
    return lineMax ; 
}

unsigned GBndLib::getMaterialIndexFromLine(unsigned line) const 
{
    unsigned numBnd = getNumBnd() ; 
    unsigned ibnd = line / NUM_MATSUR ; 
    assert( NUM_MATSUR == 4 ); 
    assert( ibnd < numBnd ) ;  
    unsigned imatsur = line - ibnd*NUM_MATSUR ; 

    LOG(error) 
        << " line " << line 
        << " ibnd " << ibnd 
        << " numBnd " << numBnd 
        << " imatsur " << imatsur
        ;

    assert( imatsur == 0 || imatsur == 3 ); 

    assert( m_optical_buffer );  
    glm::uvec4 optical = m_optical_buffer->getQuadU(ibnd, imatsur, 0 );  
 
    unsigned matIdx1 = optical.x ; 
    assert( matIdx1 >= 1 );   // 1-based index

    return matIdx1 - 1 ;  // 0-based index
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
                 2 : payload-categories corresponding to NUM_FLOAT4
                39 : wavelength samples
                 4 : float4-values

     The only dimension that can easily be extended is the middle payload-categories one, 
     the low side is constrained by layout needed to OptiX tex2d<float4> as this 
     buffer is memcpy into the texture buffer
     high side is constained by not wanting to change texture line indices 

    */

    LOG(verbose) << "GBndLib::createBufferForTex2d" ;

    NPY<float>* mat = m_mlib->getBuffer();
    NPY<float>* sur = m_slib->getBuffer();

    LOG(LEVEL) << "GBndLib::createBufferForTex2d" 
               << " mat " << mat 
               << " sur " << sur
               ; 

    if(mat == NULL ) LOG(fatal) << "NULL mat buffer" ;
    assert(mat);

    if(sur == NULL ) LOG(warning) << "NULL sur buffer" ;
      

    unsigned int ni = getNumBnd();
    unsigned int nj = NUM_MATSUR ;    // om-os-is-im

    // the klm matches the Materials and Surface buffer layouts, so can memcpy in 
    unsigned int nk = NUM_FLOAT4 ;    
    unsigned int nl = getStandardDomainLength() ; 
    unsigned int nm = 4 ; 


    assert( nl == Opticks::DOMAIN_LENGTH || nl == Opticks::FINE_DOMAIN_LENGTH ) ;

    if( mat && sur )
    {
        assert( mat->getShape(1) == sur->getShape(1) );
        assert( mat->getShape(2) == sur->getShape(2) );
    }
    else if(mat)
    {
        assert( mat->getShape(2) == nl );
    } 
    else if(sur)
    { 
        assert( sur->getShape(1) == nk );
    }


    NPY<float>* wav = NPY<float>::make( ni, nj, nk, nl, nm) ;
    wav->fill( GSurfaceLib::SURFACE_UNSET ); 

    LOG(debug) << "GBndLib::createBufferForTex2d"
               << " mat " << ( mat ? mat->getShapeString() : "NULL" )
               << " sur " << ( sur ? sur->getShapeString() : "NULL" )
               << " wav " << wav->getShapeString()
               ; 

    float* mdat = mat ? mat->getValues() : NULL ;
    float* sdat = sur ? sur->getValues() : NULL ;
    float* wdat = wav->getValues(); // destination

    for(unsigned int i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;
        for(unsigned j=0 ; j < nj ; j++)     // over imat/omat/isur/osur species
        {
            unsigned wof = nj*nk*nl*nm*i + nk*nl*nm*j ;

            if(j == IMAT || j == OMAT)    
            {
                unsigned midx = bnd[j] ;
                if(midx != UNSET)
                { 
                    unsigned mof = nk*nl*nm*midx ; 
                    memcpy( wdat+wof, mdat+mof, sizeof(float)*nk*nl*nm );  
                }
                else
                {
                    LOG(fatal) << "GBndLib::createBufferForTex2d"
                                 << " ERROR IMAT/OMAT with UNSET MATERIAL "
                                 << " i " << i  
                                 << " j " << j 
                                 ; 
                    assert(0);
                }
            }
            else if(j == ISUR || j == OSUR) 
            {
                unsigned sidx = bnd[j] ;
                if(sidx != UNSET)
                {
                    assert( sdat && sur );
                    unsigned sof = nk*nl*nm*sidx ;  
                    memcpy( wdat+wof, sdat+sof, sizeof(float)*nk*nl*nm );  
                }
            }

         }     // j
    }          // i
    return wav ; 
}



/**
GBndLib::createOpticalBuffer
-----------------------------

Optical buffer can be derived from the m_bnd array 
of guint4. It contains omat-osur-isur-imat info, 
for materials just the material one based index, for 
surfaces the one based surface index and other optical 
surface parameters. 

As it can be derived it is often not persisted.

**/





NPY<unsigned>* GBndLib::createOpticalBuffer()
{

    bool one_based = true ; // surface and material indices 1-based, so 0 can stand for unset
    // hmm, the bnd itself is zero-based

    unsigned int ni = getNumBnd();
    unsigned int nj = NUM_MATSUR ;    // om-os-is-im
    unsigned int nk = 4 ;           // THIS 4 IS NOT RELATED TO NUM_PROP

    NPY<unsigned>* optical = NPY<unsigned>::make( ni, nj, nk) ;
    optical->zero(); 
    unsigned* odat = optical->getValues();

    for(unsigned i=0 ; i < ni ; i++)      // over bnd
    {
        const guint4& bnd = m_bnd[i] ;

        for(unsigned j=0 ; j < nj ; j++)  // over imat/omat/isur/osur
        {
            unsigned offset = nj*nk*i+nk*j ;
            if(j == IMAT || j == OMAT)    
            {
                unsigned midx = bnd[j] ;
                assert(midx != UNSET);

                odat[offset+0] = one_based ? midx + 1 : midx  ; 
                odat[offset+1] = 0u ; 
                odat[offset+2] = 0u ; 
                odat[offset+3] = 0u ; 

            }
            else if(j == ISUR || j == OSUR)  
            {
                unsigned sidx = bnd[j] ;
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

/**

dbgtex.py optical buffer for 3 boundaries with just omat/imat set::

    Out[1]: 
    array([[[1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0]],

           [[1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0]],

           [[2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 0, 0, 0]]], dtype=uint32)

**/




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



void GBndLib::saveAllOverride(const char* dir)
{
    LOG(LEVEL) << "[ " << dir ;
 
    m_ok->setIdPathOverride(dir);

    save();             // only saves the guint4 bnd index
    saveToCache();      // save float buffer too for comparison with wavelength.npy from GBoundaryLib with GBndLibTest.npy 
    saveOpticalBuffer();

    m_ok->setIdPathOverride(NULL);

    LOG(LEVEL) << "]" ; 

}


