
#include <cuda_runtime.h>
#include <sstream>
#include <map>

#include "SStr.hh"
#include "scuda.h"
#include "squad.h"

#include "QUDA_CHECK.h"
#include "NP.hh"

#include "QTex.hh"
#include "QBnd.hh"

#include "SDigestNP.hh"
#include "PLOG.hh"

const plog::Severity QBnd::LEVEL = PLOG::EnvLevel("QBnd", "INFO"); 

const QBnd* QBnd::INSTANCE = nullptr ; 
const QBnd* QBnd::Get(){ return INSTANCE ; }



/**
QBnd::DescDigest
--------------------

bnd with shape (44, 4, 2, 761, 4, )::

   ni : boundaries
   nj : 0:omat/1:osur/2:isur/3:imat  
   nk : 0 or 1 property group
   nl : wavelengths
   nm : payload   

::

    2022-04-20 14:53:14.544 INFO  [4031964] [test_DescDigest@133] 
    5acc01c3 79cfae67 79cfae67 5acc01c3  Galactic///Galactic
    5acc01c3 79cfae67 79cfae67 8b22bf98  Galactic///Rock
    8b22bf98 79cfae67 79cfae67 5acc01c3  Rock///Galactic
    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air
    8b22bf98 79cfae67 79cfae67 8b22bf98  Rock///Rock
    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air
    c2759ba7 79cfae67 79cfae67 8b22bf98  Air///Steel

**/

std::string QBnd::DescDigest(const NP* bnd, int w ) 
{
    int ni = bnd->shape[0] ; 
    int nj = bnd->shape[1] ;
 
    std::vector<std::string> names ; 
    bnd->get_names(names); 
    assert( int(names.size()) == ni ); 

    std::stringstream ss ; 
    for(int i=0 ; i < ni ; i++)
    {
        ss << std::setw(3) << i << " " ; 
        for(int j=0 ; j < nj ; j++) 
        {
            std::string dig = SDigestNP::Item(bnd, i, j ) ; 
            std::string sdig = dig.substr(0, w); 
            ss << std::setw(w) << sdig << " " ; 
        }
        ss << " " << names[i] << std::endl ; 
    }
    std::string s = ss.str();  
    return s ; 
}

/**
QBnd::Add
-----------

Coordinates addition of boundaries to the optical and bnd buffers using the boundary string 
specification. 

**/

void QBnd::Add( NP** opticalplus, NP** bndplus, const NP* optical, const NP* bnd,  const std::vector<std::string>& specs ) // static 
{
    *opticalplus = AddOptical(optical, bnd->names, specs ); 
    *bndplus = AddBoundary( bnd, specs );     
}

void QBnd::GetSpecsFromString( std::vector<std::string>& specs , const char* specs_, char delim )
{
    std::stringstream ss;
    ss.str(specs_)  ;
    std::string s;
    while (std::getline(ss, s, delim)) if(!SStr::Blank(s.c_str())) specs.push_back(s) ;
    LOG(info) << " specs_ [" << specs_ << "] specs.size " << specs.size()  ;   
}

std::string QBnd::DescOptical(const NP* optical, const NP* bnd )
{
    bool consistent = optical->shape[0] == bnd->shape[0]*4 && bnd->shape[0] == int(bnd->names.size())  ;   

    typedef std::map<unsigned, std::string> MUS ; 
    MUS surf ; 

    std::stringstream ss ; 
    ss << "QBnd::DescOptical"
       << " optical " << optical->sstr() 
       << " bnd " << bnd->sstr() 
       << " bnd.names " << bnd->names.size()
       << " consistent " << ( consistent ? "YES" : "NO:ERROR" )   
       << std::endl 
       ;   

    assert(consistent); 
    assert( optical->shape.size() == 2 );

    unsigned ni = optical->shape[0] ; 
    unsigned nj = optical->shape[1] ; 
    assert( nj == 4 ); 

    const unsigned* oo = optical->cvalues<unsigned>();    

    std::vector<std::string> elem ; 

    for(unsigned i=0 ; i < ni ; i++) 
    {  
        unsigned b = i/4 ; 
        unsigned ii = i % 4 ; 
        if( ii == 0 ) 
        {   
            elem.clear();   
            const std::string& spec = bnd->names[b] ; 
            SStr::Split( spec.c_str(), '/', elem );  
            ss << std::setw(4) << b << " " << spec<< std::endl ;   
        }   

        const std::string& name = elem[ii] ; 

        ss << std::setw(4) << i << " : " << std::setw(4) << ii << " : " ; 

        for(unsigned j=0 ; j < nj ; j++) 
        {
            unsigned idx = i*nj + j ; 
            unsigned val = oo[idx] ; 
            ss << std::setw(4) << val << " " ; 

            if( j == 0 && val > 0 && ( ii == 1 || ii == 2)  ) 
            {
                surf[val] = name ;  
            }
        }
        ss << " " << name << std::endl ; 
    }   

    ss << " surfaces ....... " << std::endl ; 
    for(MUS::const_iterator it=surf.begin() ; it != surf.end() ; it++)
    {
        ss << std::setw(5) << it->first << " : " << it->second << std::endl ; 
    }

    std::string s = ss.str() ; 
    return s ; 
}

/**
QBnd::GetPerfectValues
-------------------------

bnd with shape (44, 4, 2, 761, 4, )::

   ni : boundaries
   nj : 0:omat/1:osur/2:isur/3:imat  
   nk : 0 or 1 property group
   nl : wavelengths
   nm : payload   

**/

void QBnd::GetPerfectValues( std::vector<float>& values, unsigned nk, unsigned nl, unsigned nm, const char* name ) // static 
{
    LOG(info) << name << " nk " << nk << " nl " << nl << " nm " << nm ; 

    assert( nk == 2 ); 
    assert( nl > 0 ); 
    assert( nm == 4 ); 

    float4 payload[2] ; 
    if(     strstr(name, "perfectDetectSurface"))   payload[0] = make_float4(  1.f, 0.f, 0.f, 0.f );
    else if(strstr(name, "perfectAbsorbSurface"))   payload[0] = make_float4(  0.f, 1.f, 0.f, 0.f );
    else if(strstr(name, "perfectSpecularSurface")) payload[0] = make_float4(  0.f, 0.f, 1.f, 0.f );
    else if(strstr(name, "perfectDiffuseSurface"))  payload[0] = make_float4(  0.f, 0.f, 0.f, 1.f );
    else                                            payload[0] = make_float4( -1.f, -1.f, -1.f, -1.f ); 
   
    payload[1] = make_float4( -1.f, -1.f, -1.f, -1.f ); 

    values.resize( nk*nl*nm ); 
    unsigned idx = 0 ; 
    unsigned count = 0 ; 
    for(unsigned k=0 ; k < nk ; k++)          // over payload groups
    {
        const float4& pay = payload[k] ;  
        for(unsigned l=0 ; l < nl ; l++)      // payload repeated over wavelength samples
        {
            for(unsigned m=0 ; m < nm ; m++)  // over payload values 
            {
                 idx = k*nl*nm + l*nm + m ;             
                 assert( idx == count ); 
                 count += 1 ; 
                 switch(m)
                 {
                     case 0: values[idx] = pay.x ; break ; 
                     case 1: values[idx] = pay.y ; break ; 
                     case 2: values[idx] = pay.z ; break ; 
                     case 3: values[idx] = pay.w ; break ; 
                 } 
            }
        }
    }
} 

/**
QBnd::AddOptical
------------------

Used from QBnd::Add in coordination with QBnd::AddBoundary.
Using this alone would break optical:bnd consistency. 

optical buffer has 4 uint for each species and 4 species for each boundary

Water/Steel_surface/Steel_surface/Steel
  19    0    0    0 
  21    0    3   20 
  21    0    3   20 
   4    0    0    0 

The .x is the 1-based material or surface index with 0 signifying none
which shoild only ever happen for surfaces.

**/

NP* QBnd::AddOptical( const NP* optical, const std::vector<std::string>& bnames, const std::vector<std::string>& specs )
{
    unsigned ndim = optical->shape.size() ; 
    unsigned num_bnd = bnames.size() ; 
    unsigned num_add = specs.size()  ; 
    assert( ndim == 2 ); 
    unsigned ni = optical->shape[0] ; 
    unsigned nj = optical->shape[1] ; 
    assert( 4*num_bnd == ni ); 
    assert( nj == 4 ); 
    assert( optical->ebyte == 4 && optical->uifc == 'u' ); 

    unsigned item_bytes = optical->ebyte*optical->itemsize_(0); 
    assert( item_bytes == 16u ); 

    NP* opticalplus = new NP(optical->dtype); 
    std::vector<int> opticalplus_shape(optical->shape); 
    opticalplus_shape[0] += 4*num_add ; 
    opticalplus->set_shape(opticalplus_shape) ; 

    unsigned offset = 0 ; 
    unsigned optical_arr_bytes = optical->arr_bytes() ; 
    memcpy( opticalplus->bytes() + offset, optical->bytes(), optical_arr_bytes );  
    offset += optical_arr_bytes ; 

    uint4 item = make_uint4( 0u, 0u, 0u, 0u ); 

    for(unsigned b=0 ; b < num_add ; b++)
    {
        const char* spec = SStr::Trim(specs[b].c_str());   
        std::vector<std::string> elem ; 
        SStr::Split(spec, '/', elem );  

        bool four_elem = elem.size() == 4 ; 
        if(four_elem == false) LOG(fatal) << " expecting four elem spec [" << spec << "] elem.size " << elem.size() ;  
        assert(four_elem); 

        for(unsigned s=0 ; s < 4 ; s++ )
        {
            const char* qname = elem[s].c_str(); 
            unsigned i, j ; 
            bool found = FindName(i, j, qname, bnames ); 
            unsigned idx = i*4 + j ; 

            const char* ibytes = nullptr ; 
            unsigned num_bytes = 0 ; 

            if(found)
            { 
                optical->itembytes_( &ibytes, num_bytes, idx );         
            }
            else if(strstr(qname, "perfect"))
            {
                assert( s == 1 || s == 2 );  // only expecting "perfect" surfaces not materials
                 //
                // NB: when found==false (i,j) will be stale or undefined SO DO NOT USE THEM HERE 
                //
                // NB: the only way this item is currently used is checking of item.x (aka s.optical.x)  > 0 
                //     to indicate that propagate_at_surface should be used and not propagate_at_boundary 
                //     while the value for .x has traditionally been a 1-based surface index
                //     that index is at qsim.h level just metadata : it is never used to lookup anything
                //
                // TODO: mint a new index to use for added surfaces, rather than here just using 99u 
                // 
                item.x = 99u ; 
                item.y = 99u ; 
                item.z = 99u ; 
                item.w = 99u ; 

                ibytes = (const char*)&item; 
                num_bytes = sizeof(uint4); 
            }
            else
            {
                LOG(fatal) << " FAILED to find qname " << qname ;  
                assert( 0 ); 
            }
            assert( ibytes != nullptr ); 
            assert( num_bytes == item_bytes ); 
            memcpy( opticalplus->bytes() + offset,  ibytes, item_bytes ); 
            offset += item_bytes ; 
        }
    }
    return opticalplus ; 
}


/**
QBnd::AddBoundary
------------------------

Canonically invoked from QBnd::Add in coordination with QBnd::AddOptical to maintain consistency. 

Creates new array containing the src array with extra boundaries constructed 
from materials and surfaces already present in the src array as configured by the 
specs argument. 

**/

NP* QBnd::AddBoundary( const NP* dsrc, const std::vector<std::string>& specs ) // static 
{
    const NP* src = NarrowIfWide(dsrc) ;  

    unsigned ndim = src->shape.size() ; 
    assert( ndim == 5 ); 
    unsigned ni = src->shape[0] ; 
    unsigned nj = src->shape[1] ; 
    unsigned nk = src->shape[2] ; 
    unsigned nl = src->shape[3] ; 
    unsigned nm = src->shape[4] ;

    LOG(info) 
        << " src.ebyte " << src->ebyte  
        << " src.desc " << src->desc() 
        ; 

    assert( src->ebyte == 4 ); 
    assert( ni > 0 );  
    assert( nj == 4 );   // expecting 2nd dimension to be 4: omat/osur/isur/imat 

    unsigned src_bytes = src->arr_bytes() ; 
    unsigned bnd_bytes = src->ebyte*src->itemsize_(0) ; 
    unsigned sub_bytes = src->ebyte*src->itemsize_(0,0) ; 
    assert( bnd_bytes == 4*sub_bytes ); 

    NP* dst = new NP(src->dtype);

    std::vector<int> dst_shape(src->shape); 
    dst_shape[0] += specs.size() ; 
    dst->set_shape(dst_shape) ; 

    std::vector<std::string> names ; 
    src->get_names(names); 

    std::vector<std::string> dst_names(names); 
    LOG(info) 
        << " dst_names.size before " << dst_names.size() 
        << " specs.size " << specs.size()   
        ; 

    unsigned offset = 0 ; 
    memcpy( dst->bytes() + offset, src->bytes(), src_bytes );  
    offset += src_bytes ; 

    for(unsigned b=0 ; b < specs.size() ; b++)
    {
        const char* spec = SStr::Trim(specs[b].c_str());   
        dst_names.push_back(spec); 

        std::vector<std::string> elem ; 
        SStr::Split(spec, '/', elem );  

        bool four_elem = elem.size() == 4 ; 
        if(four_elem == false) LOG(fatal) << " expecting four elem spec [" << spec << "] elem.size " << elem.size() ;  
        assert(four_elem); 

        for(unsigned s=0 ; s < 4 ; s++ )
        {
            const char* qname = elem[s].c_str(); 
            unsigned i, j ; 
            bool found = FindName(i, j, qname, names ); 
            
            const char* ibytes = nullptr ; 
            unsigned num_bytes = 0 ; 

            if(found)
            { 
                src->itembytes_( &ibytes, num_bytes, i, j );         
            }
            else if(strstr(qname, "perfect"))
            {
                std::vector<float> values ; 
                GetPerfectValues( values, nk, nl, nm, qname ); 
                ibytes = (const char*)values.data(); 
                num_bytes = sizeof(float)*values.size();   
            }
            else
            {
                LOG(fatal) << " FAILED to find qname " << qname ;  
                assert( 0 ); 
            }

            assert( ibytes != nullptr ); 
            assert( num_bytes == sub_bytes ); 

            memcpy( dst->bytes() + offset,  ibytes, num_bytes ); 
            offset += sub_bytes ; 
        }
    }

    LOG(info) << " dst_names.size after " << dst_names.size() ; 

    dst->set_names( dst_names ); 
    dst->meta = src->meta ;    // need to pass along the domain metadata 

    std::vector<std::string> dst_names_check ; 
    dst->get_names(dst_names_check); 
    LOG(info) << " dst_names_check.size after " << dst_names_check.size() ; 

    return dst ; 
}

const NP* QBnd::NarrowIfWide(const NP* buf )  // static 
{
    return buf->ebyte == 4 ? buf : NP::MakeNarrow(buf) ; 
}

/**
QBnd::QBnd
------------

Narrows the NP array if wide and creates GPU texture 

**/

QBnd::QBnd(const NP* buf)
    :
    dsrc(buf->ebyte == 8 ? buf : nullptr),
    src(NarrowIfWide(buf)),
    tex(MakeBoundaryTex(src))
{
    buf->get_names(bnames) ;  // vector of string from NP metadata
    assert( bnames.size() > 0 ); 
    INSTANCE = this ; 
} 

std::string QBnd::getItemDigest( int i, int j, int w ) const 
{
    std::string dig = SDigestNP::Item(src, i, j ) ; 
    std::string sdig = dig.substr(0, w); 
    return sdig ; 
}
 

std::string QBnd::descBoundary() const
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) 
       ss << std::setw(2) << i << " " << bnames[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
} 

unsigned QBnd::getNumBoundary() const
{
    return bnames.size(); 
}

const char* QBnd::getBoundarySpec(unsigned idx) const 
{
    assert( idx < bnames.size() ); 
    const std::string& s = bnames[idx]; 
    return s.c_str(); 
}

void QBnd::getBoundarySpec(std::vector<std::string>& names, const unsigned* idx , unsigned num_idx ) const 
{
    for(unsigned i=0 ; i < num_idx ; i++)
    {   
        unsigned index = idx[i] ;  
        const char* spec = getBoundarySpec(index);   // 0-based 
        names.push_back(spec); 
    }   
} 




const unsigned QBnd::MISSING = ~0u ; 

/**
QBnd::getBoundaryIndex
------------------------

returns the index of the first boundary matching *spec*

**/

unsigned QBnd::getBoundaryIndex(const char* spec) const 
{
    unsigned idx = MISSING ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) 
    {
        if(spec && strcmp(bnames[i].c_str(), spec) == 0) 
        {
            idx = i ; 
            break ; 
        }
    }
    return idx ;  
}

void QBnd::getBoundaryIndices( std::vector<unsigned>& bnd_idx, const char* bnd_sequence, char delim ) const 
{
    std::vector<std::string> bnd ; 
    SStr::Split(bnd_sequence,delim, bnd ); 

    bnd_idx.resize( bnd.size() ); 
    for(unsigned i=0 ; i < bnd.size() ; i++)
    {
        const char* spec = bnd[i].c_str(); 
        unsigned bidx = getBoundaryIndex(spec); 
        if( bidx == MISSING ) LOG(fatal) << " invalid spec " << spec ;      
        assert( bidx != MISSING ); 
        bnd_idx[i] = bidx ; 
    }
}

std::string QBnd::descBoundaryIndices( const std::vector<unsigned>& bnd_idx ) const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < bnd_idx.size() ; i++)
    {
        unsigned bidx = bnd_idx[i] ;  
        const char* spec = getBoundarySpec(bidx); 
        ss
            << " i " << std::setw(3) << i 
            << " bidx " << std::setw(3) << bidx
            << " spec " << spec
            << std::endl 
            ;
    }
    std::string s = ss.str(); 
    return s ; 

}


unsigned QBnd::getBoundaryLine(const char* spec, unsigned j) const 
{
    unsigned idx = getBoundaryIndex(spec); 
    bool is_missing = idx == MISSING ; 
    bool is_valid = !is_missing && idx < bnames.size() ;

    if(!is_valid) 
    {
        LOG(error) 
            << " not is_valid " 
            << " spec " << spec
            << " idx " << idx
            << " is_missing " << is_missing 
            << " bnames.size " << bnames.size() 
            ;  
    }

    assert( is_valid ); 
    unsigned line = 4*idx + j ;    
    return line ;  
}


/**
QBnd::getMaterialLine
-----------------------

Searches the bname spec for the *material* name in omat or imat slots, 
returning the first found.  

**/

unsigned QBnd::getMaterialLine( const char* material ) const 
{
    unsigned line = MISSING ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) 
    {
        std::vector<std::string> elem ; 
        SStr::Split(bnames[i].c_str(), '/', elem );  
        const char* omat = elem[0].c_str(); 
        const char* imat = elem[3].c_str(); 

        if(strcmp( material, omat ) == 0 )
        {
            line = i*4 + 0 ; 
            break ; 
        }
        if(strcmp( material, imat ) == 0 )
        {
            line = i*4 + 3 ; 
            break ; 
        }
    }
    return line ; 
}


bool QBnd::findName( unsigned& i, unsigned& j, const char* qname ) const 
{
    return FindName(i, j, qname, bnames); 
}

bool QBnd::FindName( unsigned& i, unsigned& j, const char* qname, const std::vector<std::string>& names ) 
{
    i = MISSING ; 
    j = MISSING ; 
    for(unsigned b=0 ; b < names.size() ; b++) 
    {
        std::vector<std::string> elem ; 
        SStr::Split(names[b].c_str(), '/', elem );  

        for(unsigned s=0 ; s < 4 ; s++)
        {
            const char* name = elem[s].c_str(); 
            if(strcmp(name, qname) == 0 )
            {
                i = b ; 
                j = s ; 
                return true ; 
            }
        }
    }
    return false ;  
}




/**
QBnd::MakeBoundaryTex
------------------------

Creates GPU texture with material and surface properties as a function of wavelenth.
Example of mapping from 5D array of floats into 2D texture of float4::

    .     ni nj nk  nl nm
    blib  36, 4, 2,761, 4

          ni : boundaries
          nj : 0:omat/1:osur/2:isur/3:imat  
          nk : 0 or 1 property group
          nl :  



          ni*nk*nk         -> ny  36*4*2 = 288
                   nl      -> nx           761 (fine domain, 39 when using coarse domain)
                      nm   -> float4 elem    4    

         nx*ny = 11232


TODO: need to get boundary domain range metadata into buffer json sidecar and get it uploaded with the tex

**/

QTex<float4>* QBnd::MakeBoundaryTex(const NP* buf )   // static 
{
    assert( buf->uifc == 'f' && buf->ebyte == 4 );  

    unsigned ni = buf->shape[0];  // (~123) number of boundaries 
    unsigned nj = buf->shape[1];  // (4)    number of species : omat/osur/isur/imat 
    unsigned nk = buf->shape[2];  // (2)    number of float4 property groups per species 
    unsigned nl = buf->shape[3];  // (39 or 761)   number of wavelength samples of the property
    unsigned nm = buf->shape[4];  // (4)    number of prop within the float4

    LOG(LEVEL) << " buf " << ( buf ? buf->desc() : "-" ) ;  
    assert( nm == 4 ); 

    unsigned nx = nl ;           // wavelength samples
    unsigned ny = ni*nj*nk ;     // total number of properties from all (two) float4 property groups of all (4) species in all (~123) boundaries 

    const float* values = buf->cvalues<float>(); 

    char filterMode = 'L' ; 
    //bool normalizedCoords = false ; 
    bool normalizedCoords = true ; 

    QTex<float4>* btex = new QTex<float4>(nx, ny, values, filterMode, normalizedCoords ) ; 

    bool buf_has_meta = buf->has_meta() ;
    if(!buf_has_meta) LOG(fatal) << " buf_has_meta FAIL : domain metadata is required to create texture  buf.desc " << buf->desc() ;  
    assert( buf_has_meta ); 

    quad domainX ; 
    domainX.f.x = buf->get_meta<float>("domain_low",   0.f ); 
    domainX.f.y = buf->get_meta<float>("domain_high",  0.f ); 
    domainX.f.z = buf->get_meta<float>("domain_step",  0.f ); 
    domainX.f.w = buf->get_meta<float>("domain_range", 0.f ); 

    LOG(LEVEL)
        << " domain_low " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.x  
        << " domain_high " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.y  
        << " domain_step " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.z 
        << " domain_range " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.w  
        ;

    assert( domainX.f.y > domainX.f.x ); 
    assert( domainX.f.z > 0.f ); 
    assert( domainX.f.w == domainX.f.y - domainX.f.x ); 

    btex->setMetaDomainX(&domainX); 
    btex->uploadMeta(); 

    return btex ; 
}

std::string QBnd::desc() const
{
    std::stringstream ss ; 
    ss << "QBnd"
       << " src " << ( src ? src->desc() : "-" )
       << " tex " << ( tex ? tex->desc() : "-" )
       << " tex " << tex 
       ; 
    std::string s = ss.str(); 
    return s ; 
}

void QBnd::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 

    LOG(LEVEL) 
        << " width " << std::setw(7) << width 
        << " height " << std::setw(7) << height 
        << " width*height " << std::setw(7) << width*height 
        << " threadsPerBlock"
        << "(" 
        << std::setw(3) << threadsPerBlock.x << " " 
        << std::setw(3) << threadsPerBlock.y << " " 
        << std::setw(3) << threadsPerBlock.z << " "
        << ")" 
        << " numBlocks "
        << "(" 
        << std::setw(3) << numBlocks.x << " " 
        << std::setw(3) << numBlocks.y << " " 
        << std::setw(3) << numBlocks.z << " "
        << ")" 
        ;
}

NP* QBnd::lookup()
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 
    unsigned num_lookup = width*height ; 

    NP* out = NP::Make<float>(height, width, 4 ); 

    quad* out_ = (quad*)out->values<float>(); 
    lookup( out_ , num_lookup, width, height ); 

    return out ; 
}

// from QBnd.cu
extern "C" void QBnd_lookup_0(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, quad* lookup, unsigned num_lookup, unsigned width, unsigned height ); 

void QBnd::lookup( quad* lookup, unsigned num_lookup, unsigned width, unsigned height )
{
    LOG(LEVEL) << "[" ; 

    if( tex->d_meta == nullptr )
    {
        tex->uploadMeta();    // TODO: not a good place to do this, needs to be more standard
    }
    assert( tex->d_meta != nullptr && "must QTex::uploadMeta() before lookups" );

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, width, height ); 

    size_t size = num_lookup*sizeof(quad) ;  

    quad* d_lookup  ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_lookup ), size )); 

    QBnd_lookup_0(numBlocks, threadsPerBlock, tex->texObj, tex->d_meta, d_lookup, num_lookup, width, height );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(lookup), d_lookup, size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_lookup) ); 

    LOG(LEVEL) << "]" ; 
}

void QBnd::dump( quad* lookup, unsigned num_lookup, unsigned edgeitems )
{
    LOG(LEVEL); 
    for(unsigned i=0 ; i < num_lookup ; i++)
    {
        if( i < edgeitems || i > num_lookup - edgeitems)
        {
            quad& props = lookup[i] ;  
            std::cout 
                << std::setw(10) << i 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.x 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.y
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.z 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.w 
                << std::endl 
                ; 
        }
    }
}

