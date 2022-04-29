/**
QSimTest.cc
=============



 +-------------------------------------------+----------------------------------------------------------------+
 |  test                                     |                                                                |    
 +===========================================+================================================================+
 | rng_sequence                              |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | wavelength_s                              |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | wavelength_c                              |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | fill_state                                |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | water                                     |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | rayleigh_scatter                          |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | propagate_to_boundary                     |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | hemisphere_s_polarized                    |                                                                |
 | hemisphere_p_polarized                    |                                                                |
 | hemisphere_x_polarized                    |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | propagate_at_boundary_s_polarized         |                                                                |
 | propagate_at_boundary_p_polarized         |                                                                |
 | propagate_at_boundary_x_polarized         |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+
 | propagate_at_boundary_normal_incidence    |                                                                |
 +-------------------------------------------+----------------------------------------------------------------+

**/

#include <sstream>

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"

#include "SOpticksResource.hh"
#include "SEventConfig.hh"
#include "scuda.h"
#include "squad.h"
#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

#include "QRng.hh"
#include "QBnd.hh"
#include "QPrd.hh"
#include "QProp.hh"
#include "QSim.hh"
#include "QSimLaunch.hh"
#include "QEvent.hh"
#include "QDebug.hh"
#include "QState.hh"

#include "qstate.h"
#include "qdebug.h"

#include "SEvent.hh"


std::string MakeName(const char* prefix, unsigned num, const char* ext)
{
    std::stringstream ss ; 
    ss << prefix ; 
#ifdef FLIP_RANDOM 
    ss << "FLIP_RANDOM_" ; 
#endif
    ss << num << ext ; 
    return ss.str(); 
}


template <typename T>
struct QSimTest
{
    static const char* FOLD ; 
    static const char* Path(const char* subfold, const char* name ); 
    static void  PreInit(unsigned type); 
    static unsigned Num(int argc, char** argv); 

    QSim<T> qs ; 
    unsigned type ;  // QSimLaunch type 
    unsigned num ; 

    QSimTest(unsigned type, unsigned num ); 
    void main(); 

    void rng_sequence(unsigned ni, int ni_tranche_size); 

    void boundary_lookup_all();
    void boundary_lookup_line(const char* material, T x0, T x1, unsigned nx ); 
    void prop_lookup( int iprop, T x0, T x1, unsigned nx ); 
   
    void multifilm_lookup_all();

    void wavelength(char mode, unsigned num_wavelength) ; 

    void scint_photon(unsigned num_photon); 
    void cerenkov_photon(unsigned num_photon, int print_id); 
    void cerenkov_photon_enprop(unsigned num_photon, int print_id); 
    void cerenkov_photon_expt(  unsigned num_photon, int print_id); 

    void generate_photon(); 
    void getStateNames(std::vector<std::string>& names, unsigned num_state) const ; 


    void save(const NP* a, const char* name); 


    void save_array(     const char* subfold, const char* name, const float* data, int ni, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 
    NP*  load_array(     const char* subfold, const char* name); 
     
    void save_photon(    const std::vector<sphoton>& vec,  const char* name ); 
    void save_photon(    const std::vector<quad4>& vec,  const char* name ); 
    NP*  load_photon(    const char* subfold, const char* name); 

    void save_quad(      const char* subfold, const char* name, const std::vector<quad>&  q ); 


    void save_dbg(const char* subfold); 
    void save_dbg_photon(const char* subfold, const char* name); 
    void save_dbg_state( const char* subfold, const char* name); 
    void save_dbg_prd(   const char* subfold, const char* name); 

    void fill_state(unsigned version); 
    void save_state( const char* subfold, const float* data, unsigned num_state  ); 

    void photon_launch_generate(); 
    void photon_launch_mutate(); 

    void mock_propagate(); 

    void quad_launch_generate(); 

}; 

template <typename T>
const char* QSimTest<T>::FOLD = SPath::Resolve("$TMP/QSimTest", DIRPATH) ;  // 2:dirpath create 

template <typename T>
QSimTest<T>::QSimTest(unsigned type_, unsigned num_)
    :
    type(type_),
    num(num_)
{
}


/**
QSimTest::rng_sequence
-------------------------

Default ni and ni_tranche_size_ are 1M and 100k which corresponds to 10 tranche launches
to generate the 256M randoms.  

**/

template <typename T>
void QSimTest<T>::rng_sequence(unsigned ni, int ni_tranche_size_)
{
    unsigned nj = 16 ; 
    unsigned nk = 16 ; 
    unsigned ni_tranche_size = ni_tranche_size_ > 0 ? ni_tranche_size_ : ni ; 

    qs.rng_sequence(FOLD, ni, nj, nk, ni_tranche_size ); 
}


/**
QSimTest::boundary_lookup_all
-------------------------------

Does lookups at every texel of the 2d float4 boundary texture 

**/

template <typename T>
void QSimTest<T>::boundary_lookup_all()
{
    LOG(info); 
    unsigned width = qs.getBoundaryTexWidth(); 
    unsigned height = qs.getBoundaryTexHeight(); 
    const NP* src = qs.getBoundaryTexSrc(); 
    unsigned num_lookup = width*height ; 

    std::vector<quad> lookup(num_lookup); 
    qs.boundary_lookup_all( lookup.data(), width, height ); 

    NP::Write( FOLD, "boundary_lookup_all.npy" ,  (float*)lookup.data(), height, width, 4 ); 
    src->save( FOLD, "boundary_lookup_all_src.npy" ); 
}

/**
QSimTest::boundary_lookup_line
-------------------------------

hmm need templated quad for this to work with T=double 
as its relying on 4*T = quad 

Actually no, if just narrow to float/quad at output  


**/
template <typename T>
void QSimTest<T>::boundary_lookup_line(const char* material, T x0 , T x1, unsigned nx )
{
    LOG(info); 

    unsigned line = qs.bnd->getMaterialLine(material); 
    if( line == ~0u )
    {
        LOG(fatal) << " material not in boundary tex " << material ; 
        assert(0); 
    }

    LOG(info) << " material " << material << " line " << line ; 
    unsigned k = 0 ;    // 0 or 1 picking the property float4 group to collect 


    NP* x = NP::Linspace<T>(x0,x1,nx); 
    T* xx = x->values<T>(); 

    std::vector<quad> lookup(nx) ; 
    qs.boundary_lookup_line( lookup.data(), xx, nx, line, k ); 

    NP::Write( FOLD, "boundary_lookup_line_props.npy" ,    (float*)lookup.data(), nx, 4  ); 
    NP::Write( FOLD, "boundary_lookup_line_wavelength.npy" ,  xx, nx ); 
}

template<typename T>
void QSimTest<T>::prop_lookup( int iprop, T x0, T x1, unsigned nx )
{
    unsigned tot_prop = qs.prop->ni ; 
    const NP* pp = qs.prop->a ; 

    std::vector<unsigned> pids ; 
    if( iprop == -1 )
    { 
        for(unsigned i=0 ; i < tot_prop ; i++ ) pids.push_back(i);
    }
    else
    {
        pids.push_back(iprop); 
    } 

    unsigned num_prop = pids.size() ; 

    LOG(info) 
        << " tot_prop " << tot_prop
        << " iprop " << iprop
        << " pids.size " << pids.size()
        << " num_prop " << num_prop 
        << " pp " << pp->desc()
        ; 


    NP* yy = NP::Make<T>(num_prop, nx) ; 
    NP* x = NP::Linspace<T>(x0,x1,nx); 

    qs.prop_lookup_onebyone( yy->values<T>(), x->cvalues<T>(), nx, pids ) ;

    const char* reldir = sizeof(T) == 8 ? "double" : "float" ; 

    pp->save(FOLD, reldir, "prop_lookup_pp.npy" ); 
    x->save(FOLD, reldir, "prop_lookup_x.npy" ); 
    yy->save(FOLD, reldir, "prop_lookup_yy.npy" ); 
}

template <typename T>
void QSimTest<T>::multifilm_lookup_all(){

    /*
     sample.npy : num * 8 array 
      
      pmtType , bnd , wavelength , aoi , R_s ,R_p, T_s ,T_p ,           
     */
    NP * sample = NP::Load("/tmp/debug_multi_film_table/","sample.npy");
       
    assert(sample);
    float * h_sample = sample->values<float>();
    
    unsigned num_item = sample->shape[1];
    unsigned num_sample = sample->shape[0];
     
    unsigned width = 512 ;
    unsigned height= num_sample/width;
    std::cout<<"  width  = "<< width
             <<"  height = "<< height
             <<"  num_sample ="<< num_sample
             <<std::endl;
   
    assert( height*width == num_sample);
    // convert float to quad2  
    
    quad2 h_quad2_sample[num_sample];
    for(unsigned i = 0 ; i < num_sample ; i++){
        h_quad2_sample[i].q0.u.x = (unsigned) h_sample[num_item*i+0];
        h_quad2_sample[i].q0.u.y = (unsigned) h_sample[num_item*i+1];
        h_quad2_sample[i].q0.f.z =  h_sample[num_item*i+2];
        h_quad2_sample[i].q0.f.w =  h_sample[num_item*i+3];
        h_quad2_sample[i].q1.f.x =  h_sample[num_item*i+4];
        h_quad2_sample[i].q1.f.y =  h_sample[num_item*i+5];
        h_quad2_sample[i].q1.f.z =  h_sample[num_item*i+6];
        h_quad2_sample[i].q1.f.w =  h_sample[num_item*i+7];
   }   


   quad2 h_quad2_result[num_sample];

    qs.multifilm_lookup_all(  h_quad2_sample ,  h_quad2_result ,  width,  height );

    assert(h_quad2_result);
    NP * result = NP::Make<float>(width*height , num_item);
    float* output =  result->values<float>();
    // convert quad2 to float
    
    for(unsigned i = 0 ; i < num_sample; i++){
         output[i*num_item+0] =(float)(h_quad2_result[i].q0.u.x);
         output[i*num_item+1] =(float)(h_quad2_result[i].q0.u.y);
         output[i*num_item+2] = h_quad2_result[i].q0.f.z;
         output[i*num_item+3] = h_quad2_result[i].q0.f.w;
         output[i*num_item+4] = h_quad2_result[i].q1.f.x;
         output[i*num_item+5] = h_quad2_result[i].q1.f.y;
         output[i*num_item+6] = h_quad2_result[i].q1.f.z;
         output[i*num_item+7] = h_quad2_result[i].q1.f.w;
    }   
        
    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(FOLD, create_dirs ); 

    std::cout<<" save multifilm_lut_result.npy in FOLD = "<< fold 
             <<std::endl;
    result->save(fold,"multifilm_lut_result.npy");
}



template <typename T>
void QSimTest<T>::wavelength(char mode, unsigned num_wavelength )
{
    assert( mode == 'S' || mode == 'C' ) ;

    std::vector<T> w(num_wavelength, 0.f) ; 

    std::stringstream ss ; 
    ss << "wavelength" ; ; 
    if( mode == 'S' )
    {
        unsigned hd_factor(~0u) ; 
        qs.scint_wavelength(   w.data(), w.size(), hd_factor );  // hd_factor is an output argument
        assert( hd_factor == 0 || hd_factor == 10 || hd_factor == 20 ); 
        ss << "_scint_hd" << hd_factor ; 
        char scintTexFilterMode = qs.getScintTexFilterMode() ; 
        if(scintTexFilterMode == 'P') ss << "_cudaFilterModePoint" ; 
    }
    else if( mode == 'C' )
    {
        qs.cerenkov_wavelength_rejection_sampled( w.data(), w.size() ); 
        ss << "_cerenkov" ; 
    }

    ss << "_" << num_wavelength << ".npy" ; 
    std::string s = ss.str();
    const char* name = s.c_str(); 

    qs.dump_wavelength( w.data(), w.size() ); 
   
    LOG(info) << " name " << name ; 
    NP::Write( FOLD, name ,  w ); 
}

template <typename T>
void QSimTest<T>::scint_photon(unsigned num_photon)
{
    std::string name = MakeName("photon_", num_photon, ".npy" ); 
    LOG(info) << name ; 

    std::vector<quad4> p(num_photon) ; 
    qs.scint_photon( p.data(), p.size() ); 
    qs.dump_photon(  p.data(), p.size() ); 

    NP::Write( FOLD, name.c_str(),  (float*)p.data(), p.size(), 4, 4  ); 
}

// TODO: elimiate the duplication between these

template <typename T>
void QSimTest<T>::cerenkov_photon(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qs.cerenkov_photon(   p.data(), p.size(), print_id ); // alloc on device, generate and copy back into the vector
    qs.dump_photon(       p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}

template <typename T>
void QSimTest<T>::cerenkov_photon_enprop(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_enprop_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qs.cerenkov_photon_enprop(   p.data(), p.size(), print_id ); 
    qs.dump_photon(              p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}

template <typename T>
void QSimTest<T>::cerenkov_photon_expt(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_expt_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qs.cerenkov_photon_expt(   p.data(), p.size(), print_id ); 
    qs.dump_photon(            p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}






template <typename T>
void QSimTest<T>::generate_photon()
{
    const char* gs_config = SSys::getenvvar("GS_CONFIG", "torch" ); 

    LOG(info) << "[ gs_config " << gs_config ; 
    const NP* gs = SEvent::MakeDemoGensteps(gs_config); 

    QEvent* evt = new QEvent  ; 
    evt->setGensteps(gs);
 
    qs.generate_photon(evt);  

    //std::vector<quad4> photon ; 
    //evt->downloadPhoton(photon); 
    //LOG(info) << " downloadPhoton photon.size " << photon.size() ; 
    //qs.dump_photon( photon.data(), photon.size(), "f0,f1,i2,i3" ); 

    NP* p = evt->getPhotons(); 
    save(p, "p.npy"); 


    LOG(info) << "]" ; 
}



/**
QSimTest::fill_state
-----------------------

Doing this for all boundaries with -0.5 and +0.5 for cosTheta will cover 
all states at a particular wavelength 

**/

template <typename T>
void QSimTest<T>::fill_state(unsigned version)
{
    LOG(info) << "[" ; 

    unsigned num_state = qs.bnd->getNumBoundary() ; 

    if( version == 0 )
    {
        std::vector<quad6> s(num_state) ; 
        qs.fill_state_0( s.data(), s.size() ); 
        save_state("fill_state_0", (float*)s.data(), num_state );
    }
    else if( version == 1 )
    {
        std::vector<qstate> s(num_state) ; 
        qs.fill_state_1( s.data(), s.size() ); 
        save_state("fill_state_1", (float*)s.data(), num_state );
    }
    LOG(info) << "]" ; 
}


template <typename T>
void QSimTest<T>::save_state( const char* subfold, const float* data, unsigned num_state  )
{
    std::vector<std::string> names ; 
    getStateNames(names, num_state); 

    const char* path = SPath::Resolve(FOLD, subfold, "state.npy", FILEPATH ); 

    NP* a = NP::Make<float>( num_state, 6, 4 ); // (6,4) item dimension corresponds to the 6 quads of quad6 
    a->read( data ); 
    a->set_names(names); 
    a->save(path); 
}


template <typename T>
void QSimTest<T>::getStateNames(std::vector<std::string>& names, unsigned num_state) const 
{
    unsigned* idx = new unsigned[num_state] ; 
    for(unsigned i=0 ; i < num_state ; i++) idx[i] = i ; 
    qs.bnd->getBoundarySpec(names, idx, num_state ); 
    delete [] idx ; 
}



template <typename T>
const char* QSimTest<T>::Path(const char* subfold, const char* name )
{
    return SPath::Resolve(FOLD, subfold, name, FILEPATH ); 
}

template <typename T>
void QSimTest<T>::save(const NP* a, const char* name)
{
    const char* subfold = QSimLaunch::Name(type) ; 
    assert( subfold ); 
    const char* path = Path(subfold, name ); 
    LOG(info) << " save " << name << " path " << path << " shape " << a->sstr() ; 
    a->save(path); 
}

 
template <typename T>
void QSimTest<T>::save_array(const char* subfold, const char* name, const float* data, int ni, int nj, int nk, int nl, int nm  )
{
    const char* path = Path(subfold, name); 
    NP::Write( path, data, ni, nj, nk, nl, nm  ); 
}

template <typename T>
NP* QSimTest<T>::load_array(const char* subfold, const char* name )
{
    const char* path = SPath::Resolve(FOLD, subfold, name, FILEPATH ); 
    NP* a = NP::Load(path); 
    return a ; 
}

template <typename T>
void QSimTest<T>::save_photon( const std::vector<quad4>& p, const char* name )
{
    const char* subfold = QSimLaunch::Name(type) ; 
    assert( subfold ); 
    save_array(subfold, name, (float*)p.data(), p.size(), 4, 4  );
}

template <typename T>
void QSimTest<T>::save_photon( const std::vector<sphoton>& p, const char* name )
{
    const char* subfold = QSimLaunch::Name(type) ; 
    assert( subfold ); 
    save_array(subfold, name, (float*)p.data(), p.size(), 4, 4  );
}






template <typename T>
void QSimTest<T>::save_quad(const char* subfold, const char* name, const std::vector<quad>& q )
{
    save_array(subfold, name, (float*)q.data(), q.size(), 4  );
}



template <typename T>
NP* QSimTest<T>::load_photon(const char* subfold, const char* name )
{
    NP* a = load_array(subfold, name);
    assert( a ); 
    assert( a->shape.size() == 3 ); 
    assert( a->shape[1] == 4 && a->shape[2] == 4 ); 
    return a ; 
}

template <typename T>
void QSimTest<T>::save_dbg(const char* subfold)
{
    LOG(info) << " subfold " << subfold ; 
    save_dbg_photon( subfold, "p0.npy"); 
    save_dbg_state(  subfold, "s.npy"); 
    save_dbg_prd(    subfold, "prd.npy"); 
}


template <typename T>
void QSimTest<T>::save_dbg_photon(const char* subfold, const char* name)
{
    LOG(info) << " subfold " << subfold ; 
    const sphoton& p = qs.dbg->p ; 
    save_array(subfold, name, p.cdata(), 1, 4, 4 );      
}

template <typename T>
void QSimTest<T>::save_dbg_state(const char* subfold, const char* name)
{
    LOG(info) << " subfold " << subfold ; 
    const char* path = Path(subfold, name); 
    const qstate& s = qs.dbg->s ; 
    QState::Save( s, path ); 
}


template <typename T>
void QSimTest<T>::save_dbg_prd(const char* subfold, const char* name)
{
    LOG(info) << " subfold " << subfold ; 
    const quad2& qs_prd = qs.dbg->prd ; 
    save_array(subfold, name, qs_prd.cdata(), 1, 2, 4 );      
}


template <typename T>
void QSimTest<T>::photon_launch_generate()
{
    assert( QSimLaunch::IsMutate(type)==false ); 
    const char* subfold = QSimLaunch::Name(type) ; 
    assert( subfold ); 

    std::vector<sphoton> p(num) ; 
    qs.photon_launch_generate( p.data(), p.size(), type ); 

    save_photon( p, "p.npy" ); 

    save_dbg( subfold ); 
}

/**
QSimTest::mock_propagate
----------------------------------------

NB QSimTest::PreInit does MOCK_PROPAGATE specific SEventConfig setup of event maxima 

TODO: 

* adopt sim->evt->record 
* move NP setup into common handling in QEvent probably 

**/


template <typename T>
void QSimTest<T>::mock_propagate()
{
    assert( QSimLaunch::IsMutate(type)==true ); 
    LOG(info) << "[" ; 
    LOG(info) << " SEventConfig::Desc " << SEventConfig::Desc() ;

    NP* p   = qs.duplicate_dbg_ephoton(num); 

    int bounce_max = SEventConfig::MaxBounce(); 
    NP* prd = qs.prd->duplicate_prd(num, bounce_max);  


    qs.mock_propagate( p, prd, type ); 

    const QEvent* event = qs.event ; 

    unsigned num_hit = event->getNumHit(); 
    LOG(info) << " num_hit " << num_hit ;
 
    // TODO: centralize and standardize the below into QEvent::save with base directory argument 

    NP* h = event->getHits(); 
    NP* r = event->getRecords(); 
    NP* c = event->getRec(); 
    NP* d = event->getDomain() ; 

    save(p,   "p.npy"); 
    save(prd, "prd.npy"); 
    save(r,   "r.npy"); 
    save(c,   "c.npy"); 
    save(h,   "h.npy"); 
    save(d,   "d.npy"); 

    LOG(info) << "]" ; 
}



template <typename T>
void QSimTest<T>::quad_launch_generate()
{
    assert( QSimLaunch::IsMutate(type)==false ); 
    const char* subfold = QSimLaunch::Name(type) ; 
    assert( subfold ); 

    std::vector<quad> q(num) ; 
    qs.quad_launch_generate( q.data(), q.size(), type ); 

    save_quad( subfold, "q.npy", q ); 
}

/**
QSimTest::photon_launch_mutate
--------------------------------

How should/could this use QEvent/qevent ?

**/

template <typename T>
void QSimTest<T>::photon_launch_mutate()
{
    assert( QSimLaunch::IsMutate(type)==true ); 

    unsigned src = QSimLaunch::MutateSource(type); 
    const char* src_subfold = QSimLaunch::Name(src); 
    const char* dst_subfold = QSimLaunch::Name(type) ; 
    assert( src_subfold ); 
    assert( dst_subfold ); 

    unsigned num_photon = num ; 
    NP* a = load_photon(src_subfold,  "p.npy" ); 
    LOG(info) << " loaded " << a->sstr() << " from src_subfold " << src_subfold ; 
    unsigned num_photon_ = a->shape[0] ; 
    assert( num_photon_ == num_photon ); 

    sphoton* photons = (sphoton*)a->bytes() ; 
    qs.photon_launch_mutate( photons, num_photon, type ); 

    const char* dst_path = Path(dst_subfold, "p.npy"); 
    a->save(dst_path); 

    save_dbg( dst_subfold ); 
}

/**
QSimTest::PreInit
------------------

SEventConfig settings to configure the QEvent GPU buffers
must be done prior to QEvent::init which happens when QSim is instanciated.

**/

template<typename T>
void QSimTest<T>::PreInit(unsigned type )  // static
{
    LOG(info) << "[ " <<  QSimLaunch::Name(type) ; 
    if( type == MOCK_PROPAGATE )
    {
        QPrd* prd = new QPrd ; 
        LOG(info) << prd->desc() ;  
        int num_bounce = prd->getNumBounce(); 
 
        // NB better not to hide this inside QPrd
        SEventConfig::SetMaxGenstep(0);        // MOCK_PROPAGATE starts from input photons do no gensteps needed
        SEventConfig::SetMaxPhoton(1000000);   // used for QEvent buffer sizing 
        SEventConfig::SetMaxBounce(num_bounce); 
        SEventConfig::SetMaxRecord(num_bounce+1); 
        SEventConfig::SetMaxRec(num_bounce+1); 

        LOG(info) << " SEventConfig::Desc " << SEventConfig::Desc() ;
    }
    LOG(info) << "] " <<  QSimLaunch::Name(type) ; 
}


template<typename T>
unsigned QSimTest<T>::Num(int argc, char** argv)
{
    unsigned M1   = 1000000u ; // 1 million 
    unsigned num_default = SSys::getenvunsigned("NUM", M1 )  ;   
    unsigned num = argc > 1 ? std::atoi(argv[1]) : num_default ; 
    return num ; 
}

template<typename T>
void QSimTest<T>::main()
{
    unsigned K100 =  100000u ; // default 100k usable with any GPU 
    int ni_tranche_size = SSys::getenvint("NI_TRANCHE_SIZE", K100 ); 
    int print_id = SSys::getenvint("PINDEX", -1 ); 
    const char* subfold = QSimLaunch::Name(type) ; 
    assert( subfold ); 

    LOG(info) 
        << " num " << num 
        << " type " << type
        << " subfold " << subfold
        << " ni_tranche_size " << ni_tranche_size
        << " print_id " << print_id
        ; 

    switch(type)
    {
        case RNG_SEQUENCE:                  rng_sequence(num, ni_tranche_size)         ; break ; 
        case WAVELENGTH_S  :                wavelength('S', num)                       ; break ; 
        case WAVELENGTH_C  :                wavelength('C', num)                       ; break ; 
        case SCINT_PHOTON_P:                scint_photon(num);                         ; break ; 
        case CERENKOV_PHOTON_K:             cerenkov_photon(num, print_id);            ; break ; 
        case CERENKOV_PHOTON_ENPROP_E:      cerenkov_photon_enprop(num, print_id);     ; break ; 
        case CERENKOV_PHOTON_EXPT_X :       cerenkov_photon_expt(  num, print_id);     ; break ; 

        case GENERATE_PHOTON_G:             
        case GENTORCH:
                                            generate_photon();                         ; break ; 

        case BOUNDARY_LOOKUP_ALL_A:         boundary_lookup_all()                          ; break ;  
        case BOUNDARY_LOOKUP_LINE_WATER_W:  boundary_lookup_line("Water", 80., 800., 721)  ; break ;  
        case BOUNDARY_LOOKUP_LINE_LS_L:     boundary_lookup_line("LS",    80., 800., 721)  ; break ;  
        case PROP_LOOKUP_Y:                 prop_lookup(-1, -1.f,16.f,1701)                ; break ;  
        case MULTIFILM_LOOKUP:              multifilm_lookup_all()                     ; break ;

        case FILL_STATE_0:                  fill_state(0)                              ; break ;  
        case FILL_STATE_1:                  fill_state(1)                              ; break ;  

        // hmm some conflation here between running from duplicated p0 ephoton and actual photon generation such as scint/cerenkov 

        case PROPAGATE_TO_BOUNDARY:         num=8 ; photon_launch_generate()          ; break ;   
        case PROPAGATE_AT_SURFACE:          num=8 ; photon_launch_generate()          ; break ;  

        case RAYLEIGH_SCATTER_ALIGN: 
        case PROPAGATE_AT_BOUNDARY:   
        case PROPAGATE_AT_BOUNDARY_NORMAL_INCIDENCE:  
        case HEMISPHERE_S_POLARIZED:   
        case HEMISPHERE_P_POLARIZED:  
        case HEMISPHERE_X_POLARIZED:   
        case REFLECT_DIFFUSE:
        case REFLECT_SPECULAR:

        case PROPAGATE_AT_BOUNDARY_S_POLARIZED: 
        case PROPAGATE_AT_BOUNDARY_P_POLARIZED:   
        case PROPAGATE_AT_BOUNDARY_X_POLARIZED:  
                                                 photon_launch_mutate()       ; break ;  
        case RANDOM_DIRECTION_MARSAGLIA:
        case LAMBERTIAN_DIRECTION:
                                                 quad_launch_generate()       ; break ; 
        case MOCK_PROPAGATE:
                                                mock_propagate()              ; break ; 

        default :                           
                                               LOG(fatal) << "unimplemented" << std::endl ; break ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* idpath = SOpticksResource::IDPath(true);  
    const char* cfbase = SOpticksResource::CFBase(); 

    int create_dirs = 0 ; 
    const char* rindexpath = SPath::Resolve(idpath, "GScintillatorLib/LS_ori/RINDEX.npy", create_dirs );  
    // HMM: this is just for one material ... Cerenkov needs this for all 

    // TODO: need better icdf naming now that can do both scint and ck with icdf, see CSG_GGeo_Convert::convertScintillatorLib 
    NP* icdf    = NP::Load(cfbase, "CSGFoundry", "icdf.npy");  
    NP* bnd     = NP::Load(cfbase, "CSGFoundry", "bnd.npy"); 
    NP* optical = NP::Load(cfbase, "CSGFoundry", "optical.npy"); 

    if(icdf == nullptr || bnd == nullptr)
    {
        LOG(fatal) 
            << " MISSING QSim CSGFoundry input arrays "
            << " cfbase " << cfbase 
            << " icdf " << icdf 
            << " bnd " << bnd 
            << " (recreate these with : \"c ; om ; cg ; om ; ./run.sh \" ) "
            ;
        return 1 ; 
    }

    const char* default_testname = "G" ; 
    const char* testname = SSys::getenvvar("TEST", default_testname); 
    int type = QSimLaunch::Type(testname); 
    char FD = SSys::getenvchar("FD", 'F'); // formerly TYPE 
    //char EM = SSys::getenvchar("EM", 'F'); //

    if( type == CERENKOV_PHOTON_EXPT_X ) FD = 'D' ;   // forced double 
    if( type == MULTIFILM_LOOKUP )  { 
         
        NP* multifilm_lut = NP::Load("/tmp/debug_multi_film_table/","all_table.npy");
        assert(multifilm_lut);
        QSim<float>::UploadMultiFilmLUT(multifilm_lut);

      } //enable multifilm : True

    if( FD == 'F')
    { 
        QSim<float>::UploadComponents(icdf, bnd, optical, rindexpath ); 
        QSimTest<float>::PreInit(type)  ; 
        unsigned num = QSimTest<float>::Num(argc, argv); 
        QSimTest<float> qst(type, num)  ; 
        qst.main(); 
    }
    else if( FD == 'D' )
    {
        QSim<double>::UploadComponents(icdf, bnd, optical, rindexpath ); 
        QSimTest<double>::PreInit(type)  ; 
        unsigned num = QSimTest<double>::Num(argc, argv); 
        QSimTest<double> qst(type, num) ; 
        qst.main(); 
    }
    cudaDeviceSynchronize();
    return 0 ; 
}
