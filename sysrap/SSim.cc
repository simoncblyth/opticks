#include <map>
#include <csignal>

#include "NPFold.h"
#include "scuda.h"
#include "squad.h"
#include "sdigest.h"
#include "stree.h"

#include "SLOG.hh"
#include "SStr.hh"
#include "ssys.h"
#include "spath.h"

#include "SSim.hh"
#include "SBnd.h"
#include "SPrd.h"
#include "SPMT.h"

const plog::Severity SSim::LEVEL = SLOG::EnvLevel("SSim", "DEBUG"); 
const int SSim::stree_level = ssys::getenvint("SSim__stree_level", 0) ; 

SSim* SSim::INSTANCE = nullptr ; 

SSim* SSim::Get(){ return INSTANCE ; }


SSim* SSim::CreateOrReuse()
{ 
    return INSTANCE ? INSTANCE : Create() ; 
}

void SSim::AddExtraSubfold(const char* k, const char* dir) // static
{
    LOG(LEVEL) << " k " << k << " dir " << dir ;  
    if(NPFold::Exists(dir)) 
    {
        NPFold* fold = NPFold::Load(dir) ;
        LOG(LEVEL) << " fold " << ( fold ? "YES" : "NO " ) ; 
        AddExtraSubfold(k, fold );
    }
    else
    {
        LOG(LEVEL) << " DOESNT EXIST : SKIP " ;  
    } 
}
void SSim::AddExtraSubfold(const char* k, NPFold* f) // static
{
    SSim* ss = CreateOrReuse(); 
    LOG_IF(error, ss == nullptr ) << " SSim::INSTANCE not instanciated yet " ; 
    if(ss == nullptr) return ; 
    ss->add_extra_subfold(k, f); 
}


int SSim::Compare( const SSim* a , const SSim* b )
{
    return ( a && b ) ? NPFold::Compare(a->top, b->top) : -1 ;    
}

std::string SSim::DescCompare( const SSim* a , const SSim* b )
{
    std::stringstream ss ; 
    ss << "SSim::DescCompare" 
       << " a " << ( a ? "Y" : "N" )
       << " b " << ( b ? "Y" : "N" )
       << std::endl 
       << ( ( a && b ) ? NPFold::DescCompare( a->top, b->top ) : "-" )  
       << std::endl 
       ; 
    std::string s = ss.str();
    return s ; 
}


SSim* SSim::Create()
{
    LOG_IF(fatal, INSTANCE) << "replacing SSim::INSTANCE" ; 
    new SSim ; 
    return INSTANCE ;  
}


/**
SSim::Load from persisted geometry  : used for testing 
-------------------------------------------------------
 
**/

const char* SSim::DEFAULT = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry" ; 

SSim* SSim::Load(){ return Load_(DEFAULT) ; }

SSim* SSim::Load_(const char* base_)
{
    LOG(LEVEL) << "[" ; 
    const char* base = spath::Resolve(base_ ? base_ : DEFAULT ); 
    LOG(LEVEL) 
       << " base_ [" << ( base_ ? base_ : "-" ) << "]"
       << " base [" << ( base ? base : "-" ) << "]" 
       ;

    SSim* sim = new SSim ; 
    sim->load(base);    // reldir defaults to "SSim"

    LOG(LEVEL) << "]" ; 
    return sim ; 
}


/**
SSim::Load
------------

Default reldir is "SSim" so the base directory is for example::

    $HOME/.opticks/GEOM/$GEOM/CSGFoundry  
    /tmp/GEOM/$GEOM/CSGFoundry
  

**/

SSim* SSim::Load(const char* base, const char* reldir)
{
    SSim* sim = new SSim ; 
    sim->load(base, reldir);  
    return sim ; 
}





SSim::SSim()
    :
    relp(ssys::getenvvar("SSim__RELP", RELP_DEFAULT )), // alt: "extra/GGeo"
    top(nullptr),
    extra(nullptr),
    tree(new stree)
{
    init(); 
}

/**
SSim::init
------------

**/

void SSim::init()
{
    INSTANCE = this ; 
    tree->set_level(stree_level); 
}


stree* SSim::get_tree() const { return tree ; }


/**
SSim::lookup_mtline
---------------------

Lookup matline for bnd texture or array access 
from an original Geant4 material creation index
as obtained by G4Material::GetIndex  

NB this original mtindex is NOT GENERALLY THE SAME 
as the Opticks material index. 

**/

int SSim::lookup_mtline( int mtindex ) const
{
    return tree->lookup_mtline(mtindex); 
}

std::string SSim::desc_mt() const
{
    return tree->desc_mt() ; 
}






/**
SSim::desc
------------

This and the following accessors require serialization to populate top 

**/

std::string SSim::desc() const { return top ? top->desc() : "-" ; }
std::string SSim::brief() const { return top ? top->brief() : "-" ; }

const NP* SSim::get(const char* k) const 
{ 
    assert( top ); 
    const NPFold* f = top->find_subfold( relp ); 
    if( f == nullptr ) std::cerr
        << "SSim::get"
        << " relp[" << ( relp ? relp : "-" ) << "]" 
        << " k[" << ( k ? k : "-" ) << "]" 
        << " f null "
        << std::endl 
        ;

    return f ? f->get(k) : nullptr ; 
}
void SSim::set(const char* k, const NP* a) 
{
    assert( top ); 
    NPFold* f = top->find_subfold_( relp ); 
    f->set( k, a );   
}


const NP* SSim::get_bnd() const {  return get(snam::BND);  }

/**
SSim::getBndName
-------------------

Return the bnd name for boundary index bidx using the 
metadata names list associated with the bnd.npy array.  

**/

const char* SSim::getBndName(unsigned bidx) const 
{
    const NP* bnd = get_bnd(); 
    bool valid = bnd && bidx < bnd->names.size() ; 
    if(!valid) return nullptr ; 
    const std::string& name = bnd->names[bidx] ; 
    return name.c_str()  ;  // no need for strdup as it lives in NP vector 
}
int SSim::getBndIndex(const char* bname) const
{
    unsigned count = 0 ;  
    const NP* bnd = get_bnd(); 
    int bidx = bnd->get_name_index(bname, count ); 
    bool bname_found = count == 1 && bidx > -1  ;

    LOG_IF(fatal, !bname_found) 
        << " bname " << bname
        << " bidx " << bidx
        << " count " << count
        << " bname_found " << bname_found
        ;    

    assert( bname_found ); 
    return bidx ; 
}

const SBnd* SSim::get_sbnd() const 
{ 
    const NP* bnd = get_bnd(); 
    return bnd ? new SBnd(bnd) : nullptr  ;  
}
const SPrd* SSim::get_sprd() const 
{ 
    const NP* bnd = get_bnd(); 
    return bnd ? new SPrd(bnd) : nullptr  ;  
}

/**
SSim::get_jpmt
---------------

Note that if the top fold does not have the JPMT_RELP "extra/jpmt"  subfold
then this returns nullptr. 

So that means must first call SSim::AddExtraSubfold

**/

const NPFold* SSim::get_jpmt() const 
{
    const NPFold* f = top ? top->find_subfold(JPMT_RELP) : nullptr ; 
    return f ; 
}
const SPMT* SSim::get_spmt() const 
{
    const NPFold* jpmt = get_jpmt(); 
    return jpmt ? new SPMT(jpmt) : nullptr ; 
}
const NPFold* SSim::get_spmt_f() const 
{
    const SPMT* spmt = get_spmt() ;
    const NPFold* spmt_f = spmt ? spmt->serialize() : nullptr ;
    return spmt_f ; 
}



void SSim::add_extra_subfold(const char* k, NPFold* f )
{
    assert(k); 
    LOG_IF(LEVEL, f == nullptr) << "k:" << k  << " f null " ; 
    if(f == nullptr) return ; 

    if( extra == nullptr ) extra = new NPFold ; 
    extra->add_subfold(k,f);  
}

/**
SSim::save
------------

Canonical usage from CSGFoundry::save_ with:: 

    sim->save(dir, SSim::RELDIR)

**/

void SSim::save(const char* base, const char* reldir) 
{
    if(top == nullptr) serialize() ; 
    LOG_IF(fatal, top == nullptr) << " top null : MUST serialize before save, serialize failed ? " ;  
    assert( top != nullptr ) ; 

    const char* dir = spath::Resolve(base, reldir) ;  
    top->save(dir); 
}



/**
SSim::load
------------

**/

void SSim::load(const char* base, const char* reldir)
{ 
    const char* dir = spath::Resolve(base, reldir) ;  
    load_(dir); 
}

void SSim::load_(const char* dir)
{
    LOG(LEVEL) << "[" ; 
    LOG_IF(fatal, top != nullptr)  << " top is NOT nullptr : cannot SSim::load into pre-serialized instance " ;  
    top = new NPFold ; 

    LOG(LEVEL) << "[ top.load [" << dir << "]" ; 

    top->load(dir) ;   

    LOG(LEVEL) << "] top.load [" << dir << "]" ; 

    NPFold* f_tree = top->get_subfold( stree::RELDIR ) ; 
    tree->import( f_tree ); 

    LOG(LEVEL) << "]" ; 
}


/**
SSim::serialize
-----------------

NPFold layout::

   SSim/extra  
   SSim/stree


This is invoked by::

    CSGOptiX::InitSim
    SSim::save 

Q: Is this needed other than for saving ? How so ? 


**/


void SSim::serialize()
{
    bool has_top = hasTop(); 

    LOG(LEVEL) << "[" ; 
    LOG_IF(fatal, has_top )  << " has_top : cannot serialize twice : DONT SERIALIZE AFTER LOADING SSim " ;  
    assert( !has_top );
 
    top = new NPFold ;  
    NPFold* f_tree = tree->serialize() ;
    top->add_subfold( stree::RELDIR, f_tree ); 

    if( extra ) 
    {
        top->add_subfold( EXTRA, extra ); 
    }

    LOG(LEVEL) << "]" ; 
}

bool SSim::hasTop() const
{
   return top != nullptr ; 
}





template<typename ... Args>
void SSim::addFake( Args ... args )
{
    std::vector<std::string> specs = {args...};
    LOG(LEVEL) << "specs.size " << specs.size()  ; 
    addFake_(specs); 

}
template void SSim::addFake( const char* ); 
template void SSim::addFake( const char*, const char* ); 
template void SSim::addFake( const char*, const char*, const char* ); 

/**
SSim::addFake_
----------------

Fabricates boundaries and appends them to the bnd and optical arrays

**/

void SSim::addFake_( const std::vector<std::string>& specs )
{  
    bool has_optical = hasOptical(); 
    LOG_IF(fatal, !has_optical) << " optical+bnd are required " ;
    assert(has_optical);  

    const NP* optical = get(snam::OPTICAL); 
    const NP* bnd     = get(snam::BND); 
 
    NP* opticalplus = nullptr ; 
    NP* bndplus = nullptr ; 

    Add( &opticalplus, &bndplus, optical, bnd, specs ); 

    //NOTE: are leaking the old ones 
    set(snam::OPTICAL, opticalplus); 
    set(snam::BND,     bndplus); 
}



/**
SSim::Add
-----------

Coordinates addition of boundaries to the optical and bnd buffers using the boundary string 
specification. 

**/

void SSim::Add( 
    NP** opticalplus, 
    NP** bndplus, 
    const NP* optical, 
    const NP* bnd, 
    const std::vector<std::string>& specs ) // static 
{
    *opticalplus = AddOptical(optical, bnd->names, specs ); 
    *bndplus = AddBoundary( bnd, specs );     
}

/**
SSim::AddOptical
------------------

Used from SSim::Add in coordination with SSim::AddBoundary.
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

NP* SSim::AddOptical( 
    const NP* optical, 
    const std::vector<std::string>& bnames, 
    const std::vector<std::string>& specs )
{
    unsigned ndim = optical->shape.size() ; 
    bool ndim_expect = ndim == 2 ;
    assert( ndim_expect ); 
    if(!ndim_expect) std::raise(SIGINT); 

    unsigned num_bnd = bnames.size() ; 
    unsigned num_add = specs.size()  ; 
    unsigned ni = optical->shape[0] ; 
    unsigned nj = optical->shape[1] ; 

    bool num_bnd_expect = 4*num_bnd == ni ;
    bool nj_expect = nj == 4  ;
    bool optical_expect = optical->ebyte == 4 && optical->uifc == 'u' ;

    assert( num_bnd_expect ); 
    assert( nj_expect ); 
    assert( optical_expect ) ; 
   
    if(!num_bnd_expect) std::raise(SIGINT); 
    if(!nj_expect) std::raise(SIGINT); 
    if(!optical_expect) std::raise(SIGINT); 

    unsigned item_bytes = optical->ebyte*optical->itemsize_(0); 
    bool item_bytes_expect = item_bytes == 16u ;
    assert( item_bytes_expect ); 
    if(!item_bytes_expect ) std::raise(SIGINT);

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
        LOG_IF(fatal, four_elem == false) << " expecting four elem spec [" << spec << "] elem.size " << elem.size() ;  
        assert(four_elem); 

        for(unsigned s=0 ; s < 4 ; s++ )
        {
            const char* qname = elem[s].c_str(); 
            int i, j ; 
            bool found = SBnd::FindName(i, j, qname, bnames ); 

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
                LOG(error) << "SBin::FindName failed to find qname [" << qname << "] from within the bnames.size " << bnames.size() ; 
                for(unsigned z=0 ; z < bnames.size() ; z++) LOG(error) << " z " << z << " bnames[z] " << bnames[z] ; 
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
    const NP* src = NP::MakeNarrowIfWide(dsrc) ;  

    unsigned ndim = src->shape.size() ; 
    bool ndim_expect = ndim == 5  ;
    assert( ndim_expect );
    if(!ndim_expect) std::raise(SIGINT); 
 
    unsigned ni = src->shape[0] ; 
    unsigned nj = src->shape[1] ; 
    unsigned nk = src->shape[2] ; 
    unsigned nl = src->shape[3] ; 
    unsigned nm = src->shape[4] ;

    LOG(LEVEL) 
        << " src.ebyte " << src->ebyte  
        << " src.desc " << src->desc() 
        ; 

    bool src_expect = src->ebyte == 4 && ni > 0  && nj == 4 ; 
    // expecting 2nd dimension to be 4: omat/osur/isur/imat 
    assert( src_expect ) ; 
    if(!src_expect) std::raise(SIGINT);


    unsigned src_bytes = src->arr_bytes() ; 
    unsigned bnd_bytes = src->ebyte*src->itemsize_(0) ; 
    unsigned sub_bytes = src->ebyte*src->itemsize_(0,0) ; 

    bool bnd_bytes_expect =  bnd_bytes == 4*sub_bytes ;
    assert( bnd_bytes_expect ); 
    if(!bnd_bytes_expect) std::raise(SIGINT); 

    NP* dst = new NP(src->dtype);

    std::vector<int> dst_shape(src->shape); 
    dst_shape[0] += specs.size() ; 
    dst->set_shape(dst_shape) ; 

    std::vector<std::string> names ; 
    src->get_names(names); 

    std::vector<std::string> dst_names(names); 
    LOG(LEVEL) 
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
        LOG_IF(fatal, four_elem == false) << " expecting four elem spec [" << spec << "] elem.size " << elem.size() ;  
        assert(four_elem); 

        for(unsigned s=0 ; s < 4 ; s++ )
        {
            const char* qname = elem[s].c_str(); 
            int i, j ; 
            bool found = SBnd::FindName(i, j, qname, names ); 
            
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

    LOG(LEVEL) << " dst_names.size after " << dst_names.size() ; 

    dst->set_names( dst_names ); 
    dst->meta = src->meta ;    // need to pass along the domain metadata 

    std::vector<std::string> dst_names_check ; 
    dst->get_names(dst_names_check); 

    LOG(LEVEL) << " dst_names_check.size after " << dst_names_check.size() ; 

    return dst ; 
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

void SSim::GetPerfectValues( 
    std::vector<float>& values, 
    unsigned nk, unsigned nl, unsigned nm, const char* name ) // static 
{
    LOG(LEVEL) << name << " nk " << nk << " nl " << nl << " nm " << nm ; 

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

bool SSim::hasOptical() const 
{
    const NP* optical = get(snam::OPTICAL); 
    const NP* bnd = get(snam::BND); 
    bool has_optical = optical != nullptr && bnd != nullptr ; 
    return has_optical ; 
}


std::string SSim::descOptical() const 
{
    const NP* optical = get(snam::OPTICAL); 
    const NP* bnd = get(snam::BND); 

    if(optical == nullptr && bnd == nullptr) return "SSim::descOptical null" ; 
    return DescOptical(optical, bnd); 
}

std::string SSim::DescOptical(const NP* optical, const NP* bnd )
{
    int num_bnd = bnd->shape[0] ; 
    int num_bnd_names = bnd->names.size() ; 
    assert( num_bnd == num_bnd_names ); 

    int num_optical = optical->shape[0] ; 
    bool consistent = num_optical == num_bnd*4  ; 

    typedef std::map<unsigned, std::string> MUS ; 
    MUS surf ; 

    std::stringstream ss ; 
    ss << "SSim::DescOptical"
       << " optical " << optical->sstr() 
       << " bnd " << bnd->sstr() 
       << " num_bnd_names " << num_bnd_names
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

std::string SSim::GetItemDigest( const NP* bnd, int i, int j, int w )
{
    //std::string dig = SDigestNP::Item(bnd, i, j ) ; 
    std::string dig = sdigest::Item(bnd, i, j ) ; 
    std::string sdig = dig.substr(0, w); 
    return sdig ; 
}

bool SSim::findName( int& i, int& j, const char* qname ) const 
{
    const NP* bnd = get_bnd(); 
    return bnd ? SBnd::FindName(i, j, qname, bnd->names) : false ; 
}






