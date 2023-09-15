/**
QSimTest.cc
=============

**/

#include <sstream>

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"

#include "SOpticksResource.hh"
#include "SEventConfig.hh"
#include "scuda.h"
#include "squad.h"
#include "ssys.h"

#include "SPath.hh"   // TODO: migrate to spath.h 
#include "spath.h"   

#include "SSim.hh"
#include "SBnd.h"
#include "SPrd.h"

#include "SEvt.hh"
#include "NP.hh"
#include "sstate.h"

#include "QRng.hh"
#include "QBnd.hh"
#include "QProp.hh"
#include "QSim.hh"
#include "QSimLaunch.hh"
#include "QEvent.hh"

#include "QDebug.hh"
#include "qdebug.h"

#include "SEvent.hh"



struct QSimTest
{
    static const char* FOLD ; 
    static const plog::Severity LEVEL ; 
    static void  EventConfig(unsigned type, const SPrd* prd);  // must be run after SEvt is instanciated
    static unsigned Num(int argc, char** argv); 

    const SPrd* sprd ; 
    QSim* qs ; 
    unsigned type ;  // QSimLaunch type 
    unsigned num ; 
    const char* subfold ; 
    int rc ; 

    QSimTest(unsigned type, unsigned num, const SPrd* sprd ); 
    void main(); 

    static const bool rng_sequence_PRECOOKED ; 
    void rng_sequence(unsigned ni, int ni_tranche_size); 

    void boundary_lookup_all();
    void boundary_lookup_line(const char* material, float x0, float x1, unsigned nx ); 

    template<typename T>
    void prop_lookup( int iprop, T x0, T x1, unsigned nx ); 
   
    void multifilm_lookup_all();

    void wavelength() ; 
    void RandGaussQ_shoot(); 

    void dbg_gs_generate(); 


    void generate_photon(); 
    void getStateNames(std::vector<std::string>& names, int num_state) const ; 

    void fill_state(unsigned version); 
    void save_state( const char* subfold, const float* data, int num_state  ); 

    void photon_launch_generate(); 
    void photon_launch_mutate(); 
    void fake_propagate(); 
    void quad_launch_generate(); 


}; 

/**
QSimTest::LEVEL
----------------

Inhibiting logging  within executables does not work ... 

**/

const plog::Severity QSimTest::LEVEL = SLOG::EnvLevel("QSimTest", "INFO"); 



QSimTest::QSimTest(unsigned type_, unsigned num_, const SPrd* sprd_)
    :
    sprd(sprd_),
    qs(QSim::Create()),
    type(type_),
    num(num_),
    subfold(QSimLaunch::Name(type)),
    rc(0)
{
}


/**
QSimTest::rng_sequence
-------------------------

Default ni and ni_tranche_size_ are 1M and 100k which corresponds to 10 tranche launches
to generate the 256M randoms.  

Default dir is $TMP/QSimTest/rng_sequence leading to npy paths like::

    /tmp/blyth/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy

**/


const bool QSimTest::rng_sequence_PRECOOKED = ssys::getenvbool("QSimTest__rng_sequence_PRECOOKED") ; 

void QSimTest::rng_sequence(unsigned ni, int ni_tranche_size_)
{
    unsigned nj = 16 ; 
    unsigned nk = 16 ; 
    unsigned ni_tranche_size = ni_tranche_size_ > 0 ? ni_tranche_size_ : ni ; 

    const char* udir = rng_sequence_PRECOOKED ? "$HOME/.opticks/precooked/QSimTest/rng_sequence" : "$FOLD" ; 
    LOG_IF(error, rng_sequence_PRECOOKED) 
        << " QSimTest__rng_sequence_PRECOOKED envvar triggers directory override " << std::endl 
        << " default  [" << "$FOLD" << "] " << std::endl 
        << " override [" << udir << "]"  
        ; 

    qs->rng_sequence<float>(udir, ni, nj, nk, ni_tranche_size ); 
}


/**
QSimTest::boundary_lookup_all
-------------------------------

Does lookups at every texel of the 2d float4 boundary texture 

**/

void QSimTest::boundary_lookup_all()
{
    unsigned width = qs->getBoundaryTexWidth(); 
    unsigned height = qs->getBoundaryTexHeight(); 
    const NP* src = qs->getBoundaryTexSrc(); 

    assert( height % 8 == 0 ); 
    unsigned num_bnd = height/8 ;  

    NP* l = qs->boundary_lookup_all( width, height ); 
    assert( l->has_shape( num_bnd, 4, 2, width, 4 ) ); 

    l->save("$FOLD/lookup_all.npy" ); 
    src->save("$FOLD/lookup_all_src.npy" ); 
}

/**
QSimTest::boundary_lookup_line
-------------------------------

Single material property lookups across domain of wavelength values

**/
void QSimTest::boundary_lookup_line(const char* material, float x0 , float x1, unsigned nx )
{
    LOG(info); 

    unsigned line = qs->bnd->sbn->getMaterialLine(material); 
    if( line == ~0u )
    {
        LOG(fatal) << " material not in boundary tex " << material ; 
        assert(0); 
    }

    LOG(info) << " material " << material << " line " << line ; 
    unsigned k = 0 ;    // 0 or 1 picking the property float4 group to collect 

    NP* x = NP::Linspace<float>(x0,x1,nx); 
    float* xx = x->values<float>(); 

    NP* l = qs->boundary_lookup_line( xx, nx, line, k ); 

    l->save("$FOLD/lookup_line.npy" ); 
    x->save("$FOLD/lookup_line_wavelength.npy"); 
}

/**
QSimTest::prop_lookup
----------------------

Testing QProp/qprop::interpolate machinery for on device interpolated property access
very simular to traditional Geant4 interpolation, without the need for textures. 

Multiple launches are done by QSim::prop_lookup_onebyone 

**/

template<typename T>
void QSimTest::prop_lookup( int iprop, T x0, T x1, unsigned nx )
{
    unsigned tot_prop = qs->prop->ni ; 
    const NP* pp = qs->prop->a ; 

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

    qs->prop_lookup_onebyone( yy->values<T>(), x->cvalues<T>(), nx, pids ) ;

    const char* reldir = sizeof(T) == 8 ? "double" : "float" ; 

    pp->save("$FOLD", reldir, "prop_lookup_pp.npy" ); 
    x->save("$FOLD", reldir, "prop_lookup_x.npy" ); 
    yy->save("$FOLD", reldir, "prop_lookup_yy.npy" ); 
}





void QSimTest::multifilm_lookup_all(){

    /*
     sample.npy : num * 8 array 
      
      pmtType , bnd , wavelength , aoi , R_s, T_s , R_p, T_p ,           
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
                      // SCB :    ^^^^ this is casting when should be reinterpreting ?                                                

        h_quad2_sample[i].q0.f.z =  h_sample[num_item*i+2];
        h_quad2_sample[i].q0.f.w =  h_sample[num_item*i+3];
        h_quad2_sample[i].q1.f.x =  h_sample[num_item*i+4];
        h_quad2_sample[i].q1.f.y =  h_sample[num_item*i+5];
        h_quad2_sample[i].q1.f.z =  h_sample[num_item*i+6];
        h_quad2_sample[i].q1.f.w =  h_sample[num_item*i+7];
   }   


   quad2 h_quad2_result[num_sample];

    qs->multifilm_lookup_all(  h_quad2_sample ,  h_quad2_result ,  width,  height );

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
        

    result->save("$FOLD/multifilm_lut_result.npy");
}



void QSimTest::wavelength()
{
    NP* w = nullptr ; 

    std::stringstream ss ; 
    ss << "wavelength" ; ; 
    if( type == WAVELENGTH_SCINTILLATION )
    {
        unsigned hd_factor(~0u) ; 
        w = qs->scint_wavelength( num, hd_factor );  // hd_factor is an output argument
        assert( hd_factor == 0 || hd_factor == 10 || hd_factor == 20 ); 
        ss << "_scint_hd" << hd_factor ; 

        char scintTexFilterMode = qs->getScintTexFilterMode() ; 
        if(scintTexFilterMode == 'P') ss << "_cudaFilterModePoint" ; 
    }
    else if( type == WAVELENGTH_CERENKOV )
    {
      //  w = qs->cerenkov_wavelength_rejection_sampled(num); 
        assert(0); 
        ss << "_cerenkov" ; 
    }

    ss << "_" << num << ".npy" ; 
    std::string s = ss.str();
    const char* name = s.c_str(); 

    float* ww = w->values<float>(); 
    qs->dump_wavelength( ww, num ); 
   
    LOG(info) << " name " << name ; 
    w->save("$FOLD", name ); 
}



void QSimTest::RandGaussQ_shoot()
{
    NP* v = qs->RandGaussQ_shoot(num)  ; 
    v->save("$FOLD/RandGaussQ_shoot.npy" ); 
}



void QSimTest::dbg_gs_generate()
{
    NP* p = qs->dbg_gs_generate(num, type); 

    p->save("$FOLD/p.npy"); 

    if( type == SCINT_GENERATE )
    {
        qs->dbg->save_scint_gs("$FOLD");  
    }
    else if( type == CERENKOV_GENERATE )
    {
        qs->dbg->save_cerenkov_gs("$FOLD");  
    }
    else
    {
        LOG(fatal) << "unexpected type " << type << " subfold " << subfold ; 
    }
}



void QSimTest::generate_photon()
{
    const char* gs_config = ssys::getenvvar("GS_CONFIG", "torch" ); 

    LOG(info) << "[ gs_config " << gs_config ; 
    const NP* gs = SEvent::MakeDemoGensteps(gs_config); 

    SEvt* evt = SEvt::Create(SEvt::EGPU) ; 
    assert(evt); 

    evt->addGenstep(gs); 

    qs->generate_photon();  

    NP* p = qs->event->gatherPhoton(); 
    p->save("$FOLD/p.npy"); 

    LOG(info) << "]" ; 
}



/**
QSimTest::fill_state
-----------------------

Doing this for all boundaries with -0.5 and +0.5 for cosTheta will cover 
all states at a particular wavelength 

**/

void QSimTest::fill_state(unsigned version)
{
    LOG(info) << "[" ; 

    unsigned num_state = qs->bnd->sbn->getNumBoundary() ; 

    if( version == 0 )
    {
        std::vector<quad6> s(num_state) ; 
        qs->fill_state_0( s.data(), s.size() ); 
        save_state("fill_state_0", (float*)s.data(), num_state );
    }
    else if( version == 1 )
    {
        std::vector<sstate> s(num_state) ; 
        qs->fill_state_1( s.data(), s.size() ); 
        save_state("fill_state_1", (float*)s.data(), num_state );
    }
    LOG(info) << "]" ; 
}


void QSimTest::save_state( const char* subfold, const float* data, int num_state  )
{
    std::vector<std::string> names ; 
    getStateNames(names, num_state); 

    NP* a = NP::Make<float>( num_state, 6, 4 ); // (6,4) item dimension corresponds to the 6 quads of quad6 
    a->read( data ); 
    a->set_names(names); 
    a->save("$FOLD/state.npy"); 
}


void QSimTest::getStateNames(std::vector<std::string>& names, int num_state) const 
{
    int* idx = new int[num_state] ; 
    for(int i=0 ; i < num_state ; i++) idx[i] = i ; 
    qs->bnd->sbn->getBoundarySpec(names, idx, num_state ); 
    delete [] idx ; 
}

void QSimTest::photon_launch_generate()
{
    assert( QSimLaunch::IsMutate(type)==false ); 
    NP* p = qs->photon_launch_generate(num, type ); 
    p->save("$FOLD/p.npy"); 
    qs->dbg->save("$FOLD"); 
}

/**
QSimTest::fake_propagate
----------------------------------------

NB QSimTest::EventConfig does FAKE_PROPAGATE specific SEventConfig setup of event maxima 

**/


void QSimTest::fake_propagate()
{
    assert( QSimLaunch::IsMutate(type)==true ); 
    LOG(info) << "[" ; 
    LOG(info) << " SEventConfig::Desc " << SEventConfig::Desc() ;

    NP* p = sphoton::make_ephoton_array(num); 

    SEvt* sev = SEvt::Get_EGPU(); 
    assert( sev ); 
    sev->setInputPhoton(p);  
    sev->setFramePlaceholder() ; 

    int bounce_max = SEventConfig::MaxBounce(); 
    NP* prd = sprd->fake_prd(num, bounce_max);  

    LOG(info) 
        << " num " << num 
        << " p " << ( p ? p->sstr() : "-" ) 
        << " bounce_max " << bounce_max 
        << " prd " << ( prd ? prd->sstr() : "-" ) 
        ; 

    // sev->add_array("prd0", prd ); 
    // its too soon to add array here, must be after the QEvent::setGenstep 
    // which calls SEvt/clear (done in QSim::fake_propagate)

    int eventID = 0 ; 

    sev->beginOfEvent(eventID) ;  // this tees up input photon gensteps

    qs->fake_propagate( prd, type ); 

    sev->endOfEvent(eventID) ;    // saves and clears 

    LOG(info) << "]" ; 
}







void QSimTest::quad_launch_generate()
{
    assert( QSimLaunch::IsMutate(type)==false ); 
    NP* q = qs->quad_launch_generate(num, type ); 
    q->set_meta<std::string>("source", "QSimTest.sh"); 
    q->save("$FOLD/q.npy"); 
}



/**
QSimTest::photon_launch_mutate
--------------------------------

How should/could this use QEvent/qevent ?

**/

void QSimTest::photon_launch_mutate()
{
    assert( QSimLaunch::IsMutate(type)==true ); 

    unsigned src = QSimLaunch::MutateSource(type); 
    const char* src_subfold = QSimLaunch::Name(src); 
    assert( src_subfold ); 

    unsigned num_photon = num ; 
    NP* a = NP::Load("$FOLD", src_subfold,  "p.npy" ); 

    if( a == nullptr )
    {
        LOG(fatal) 
             << "failed to NP::Load photons from "
             <<  " src_subfold " << src_subfold   
             << std::endl 
             << " YOU PROBABLY NEED TO RUN ANOTHER TEST FIRST TO GENERATE THE PHOTONS "
             ;
        rc = 101 ; 
        return ; 
    } 

    LOG(info) << " loaded " << a->sstr() << " from src_subfold " << src_subfold ; 
    unsigned num_photon_ = a->shape[0] ; 
    assert( num_photon_ == num_photon ); 

    sphoton* photons = (sphoton*)a->bytes() ; 
    qs->photon_launch_mutate( photons, num_photon, type ); 

    a->save("$FOLD/p.npy"); 

    qs->dbg->save("$FOLD"); 
}

/**
QSimTest::EventConfig
-----------------------

SEventConfig settings to configure the QEvent GPU buffers
must be done prior to QEvent::init which happens when QSim is instanciated.

TODO : looks like for FAKE_PROPAGATE are changing config after 
SEvt instanciation, hence the changes dont do anything ? 

SO THIS NEEDS REWORKING TO CHANGE CONFIG EARLIER 

**/

void QSimTest::EventConfig(unsigned type, const SPrd* prd )  // static
{
    SEvt* sev = SEvt::Get_EGPU();
    LOG_IF(fatal, sev != nullptr ) << "QSimTest::EventConfig must be done prior to instanciating SEvt, eg for fake_propagate bounce consistency " ;  
    assert(sev == nullptr); 

    LOG(LEVEL) << "[ " <<  QSimLaunch::Name(type) ; 
    if( type == FAKE_PROPAGATE )
    {
        LOG(LEVEL) << prd->desc() ;  
        int maxbounce = prd->getNumBounce(); 

        SEventConfig::SetMaxBounce(maxbounce); 
        SEventConfig::SetEventMode("StandardFullDebug"); 
        SEventConfig::Initialize();   

        SEventConfig::SetMaxGenstep(1);        // FAKE_PROPAGATE starts from input photons but uses a single placeholder genstep 
        SEventConfig::SetMaxPhoton(1000000);   // used for QEvent buffer sizing 

        LOG(LEVEL) << " SEventConfig::Desc " << SEventConfig::Desc() ;
    }
    LOG(LEVEL) << "] " <<  QSimLaunch::Name(type) ; 
}


unsigned QSimTest::Num(int argc, char** argv)
{
    unsigned M1   = 1000000u ; // 1 million 
    unsigned num_default = ssys::getenvunsigned("NUM", M1 )  ;   
    unsigned num = argc > 1 ? std::atoi(argv[1]) : num_default ; 
    return num ; 
}

void QSimTest::main()
{
    unsigned K100 =  100000u ; // default 100k usable with any GPU 
    int ni_tranche_size = ssys::getenvint("NI_TRANCHE_SIZE", K100 ); 
    int print_id = ssys::getenvint("PINDEX", -1 ); 
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

        case WAVELENGTH_SCINTILLATION:      
        case WAVELENGTH_CERENKOV:           
                                            wavelength()                               ; break ; 

        case RANDGAUSSQ_SHOOT:
                                            RandGaussQ_shoot()                         ; break ;  

        case SCINT_GENERATE:           
        case CERENKOV_GENERATE:             
                                             dbg_gs_generate()               ; break ; 

        case CERENKOV_GENERATE_ENPROP_FLOAT:
        case CERENKOV_GENERATE_ENPROP_DOUBLE:
        case CERENKOV_GENERATE_EXPT :      
                                              assert(0)                       ;   break ; 

        case GENERATE_PHOTON_G:             
        case GENTORCH:
                                            generate_photon();                         ; break ; 

        case BOUNDARY_LOOKUP_ALL:           boundary_lookup_all()                          ; break ;  
        case BOUNDARY_LOOKUP_WATER:         boundary_lookup_line("Water", 80., 800., 721)  ; break ;  
        case BOUNDARY_LOOKUP_LS:            boundary_lookup_line("LS",    80., 800., 721)  ; break ;  

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
                                                 assert(0) ; break ;  // TODO: review these tests and revive them 
        case REFLECT_DIFFUSE:
        case REFLECT_SPECULAR:
                                                 photon_launch_generate()  ; break ; 
        case HEMISPHERE_S_POLARIZED:   
        case HEMISPHERE_P_POLARIZED:  
        case HEMISPHERE_X_POLARIZED:   
                                                 photon_launch_generate()       ; break ;  
        case PROPAGATE_AT_BOUNDARY_S_POLARIZED: 
        case PROPAGATE_AT_BOUNDARY_P_POLARIZED:   
        case PROPAGATE_AT_BOUNDARY_X_POLARIZED:  
		case PROPAGATE_AT_MULTIFILM_S_POLARIZED: 
		case PROPAGATE_AT_MULTIFILM_P_POLARIZED: 
		case PROPAGATE_AT_MULTIFILM_X_POLARIZED:
                                                 photon_launch_mutate()         ; break ;  
        case QGEN_RANDOM_DIRECTION_MARSAGLIA:
        case QGEN_LAMBERTIAN_DIRECTION:
        case QGEN_SMEAR_NORMAL_SIGMA_ALPHA:
        case QGEN_SMEAR_NORMAL_POLISH:
                                                 quad_launch_generate()       ; break ; 
        case FAKE_PROPAGATE:
                                                fake_propagate()              ; break ; 

        default :                           
                                               LOG(fatal) << "unimplemented" << std::endl ; break ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* testname = ssys::getenvvar("TEST", "hemisphere_s_polarized"); 
    int type = QSimLaunch::Type(testname); 
    unsigned num = QSimTest::Num(argc, argv); 

    SSim* sim = SSim::Load(); 
    QSim::UploadComponents(sim);   // instanciates things like QBnd
    const SPrd* prd = sim->get_sprd() ; 

    QSimTest::EventConfig(type, prd );  // must be after QBnd instanciation and before SEvt instanciation

    SEvt* evt = SEvt::Create(SEvt::EGPU) ; 
    assert(evt);  

    QSimTest qst(type, num, prd)  ; 
    qst.main(); 

    cudaDeviceSynchronize();

    LOG(info) << " qst.rc " << qst.rc ; 
    return qst.rc  ; 
}
