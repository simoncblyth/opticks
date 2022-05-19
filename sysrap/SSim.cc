#include <map>

#include "NPFold.h"
#include "SName.h"
#include "scuda.h"
#include "squad.h"
#include "SDigestNP.hh"

#include "PLOG.hh"
#include "SSim.hh"

const plog::Severity SSim::LEVEL = PLOG::EnvLevel("SSim", "DEBUG"); 
SSim* SSim::INSTANCE = nullptr ; 
const unsigned SSim::MISSING = ~0u ; 

SSim* SSim::Get()
{ 
   if(INSTANCE == nullptr) new SSim ; 
   return INSTANCE ; 
}

SSim* SSim::Load(const char* base)
{
    SSim* sim = new SSim ; 
    sim->load(base);  
    return sim ; 
}

SSim* SSim::Load(const char* base, const char* rel)
{
    SSim* sim = new SSim ; 
    sim->load(base, rel);  
    return sim ; 
}

SSim::SSim()
    :
    fold(new NPFold),
    bd(nullptr)
{
    INSTANCE = this ; 
}

void SSim::add(const char* k, const NP* a )
{ 
    fold->add(k,a);  
    if(strcmp(k, BND)==0) bd = new SName(a->names) ; 
}
const NP* SSim::get(const char* k) const { return fold->get(k);  }

void SSim::load(const char* base){ fold->load(base);  }
void SSim::load(const char* base, const char* rel){ fold->load(base, rel);  }

void SSim::save(const char* base) const { fold->save(base); }
void SSim::save(const char* base, const char* rel) const { fold->save(base, rel); }

std::string SSim::desc() const { return fold->desc() ; }

const char* SSim::getBndName(unsigned bidx) const 
{
    const NP* bnd = fold->get(BND); 
    bool valid = bnd && bidx < bnd->names.size() ; 
    if(!valid) return nullptr ; 
    const std::string& name = bnd->names[bidx] ; 
    return name.c_str()  ;  // no need for strdup as it lives in  NP metadata 
}
int SSim::getBndIndex(const char* bname) const
{
    unsigned count = 0 ;  
    int bidx = bd->getIndex(bname, count ); 
    bool bname_found = count == 1 && bidx > -1  ;
    if(!bname_found) 
       LOG(fatal) 
          << " bname " << bname
          << " bidx " << bidx
          << " count " << count
          << " bname_found " << bname_found
          << " bd.getNumName " << bd->getNumName() 
          << " bd.detail " << std::endl 
          << bd->detail()
          ;    

    assert( bname_found ); 
    return bidx ; 
}



/**
SSim::Add
-----------

Coordinates addition of boundaries to the optical and bnd buffers using the boundary string 
specification. 

**/

void SSim::Add( NP** opticalplus, NP** bndplus, const NP* optical, const NP* bnd,  const std::vector<std::string>& specs ) // static 
{
    *opticalplus = AddOptical(optical, bnd->names, specs ); 
    *bndplus = AddBoundary( bnd, specs );     
}

/**
SSim::AddOptical
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

NP* SSim::AddOptical( const NP* optical, const std::vector<std::string>& bnames, const std::vector<std::string>& specs )
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
SSim::AddBoundary
------------------------

Canonically invoked from SSim::Add in coordination with SSim::AddOptical to maintain consistency. 

Creates new array containing the src array with extra boundaries constructed 
from materials and surfaces already present in the src array as configured by the 
specs argument. 

**/

NP* SSim::AddBoundary( const NP* dsrc, const std::vector<std::string>& specs ) // static 
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

const NP* SSim::NarrowIfWide(const NP* buf )  // static 
{
    return buf->ebyte == 4 ? buf : NP::MakeNarrow(buf) ; 
}


bool SSim::findName( unsigned& i, unsigned& j, const char* qname ) const 
{
    return bd ? FindName(i, j, qname, bd->name) : false ; 
}

bool SSim::FindName( unsigned& i, unsigned& j, const char* qname, const std::vector<std::string>& names ) 
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
SSim::GetPerfectValues
-------------------------

bnd with shape (44, 4, 2, 761, 4, )::

   ni : boundaries
   nj : 0:omat/1:osur/2:isur/3:imat  
   nk : 0 or 1 property group
   nl : wavelengths
   nm : payload   

**/

void SSim::GetPerfectValues( std::vector<float>& values, unsigned nk, unsigned nl, unsigned nm, const char* name ) // static 
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



std::string SSim::DescOptical(const NP* optical, const NP* bnd )
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
SSim::DescDigest
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

std::string SSim::DescDigest(const NP* bnd, int w ) 
{
    int ni = bnd->shape[0] ; 
    int nj = bnd->shape[1] ;
 
    const std::vector<std::string>& names = bnd->names ; 
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

std::string SSim::GetItemDigest( const NP* bnd, int i, int j, int w )
{
    std::string dig = SDigestNP::Item(bnd, i, j ) ; 
    std::string sdig = dig.substr(0, w); 
    return sdig ; 
}




