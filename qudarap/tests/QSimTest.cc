#include <sstream>

#include "OPTICKS_LOG.hh"
#include "scuda.h"
#include "squad.h"
#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

#include "QRng.hh"
#include "QBnd.hh"
#include "QProp.hh"
#include "QSim.hh"
#include "QEvent.hh"


template <typename T>
struct QSimTest
{
    static const char* FOLD ; 
    static std::string MakeName(const char* prefix, unsigned num, const char* ext);

    QSim<T>& qs ; 
    QSimTest(QSim<T>& qs); 
    void main(int argc, char** argv, char test); 

    void rng_sequence(unsigned ni, int ni_tranche_size); 
    void wavelength(char mode, unsigned num_wavelength) ; 

    void scint_photon(unsigned num_photon); 
    void cerenkov_photon(unsigned num_photon, int print_id); 
    void cerenkov_photon_enprop(unsigned num_photon, int print_id); 
    void cerenkov_photon_expt(  unsigned num_photon, int print_id); 

    void generate_photon(); 


    void boundary_lookup_all();
    void boundary_lookup_line(const char* material, T x0, T x1, unsigned nx ); 


    void prop_lookup( int iprop, T x0, T x1, unsigned nx ); 

}; 

template <typename T>
const char* QSimTest<T>::FOLD = "$TMP/QSimTest" ; 

template <typename T>
QSimTest<T>::QSimTest(QSim<T>& qs_)
    :
    qs(qs_) 
{
}



template <typename T>
void QSimTest<T>::rng_sequence(unsigned ni, int ni_tranche_size_)
{
    unsigned nj = 16 ; 
    unsigned nk = 16 ; 
    unsigned ni_tranche_size = ni_tranche_size_ > 0 ? ni_tranche_size_ : ni ; 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(FOLD, create_dirs ); 
    qs.rng_sequence(fold, ni, nj, nk, ni_tranche_size ); 
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


template <typename T>
std::string QSimTest<T>::MakeName(const char* prefix, unsigned num, const char* ext)
{
    std::stringstream ss ; 
    ss << prefix ; 
#ifdef FLIP_RANDOM 
    ss << "FLIP_RANDOM_" ; 
#endif
    ss << num << ext ; 
    return ss.str(); 
}



// TODO: elimiate the duplication between these

template <typename T>
void QSimTest<T>::cerenkov_photon(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qs.cerenkov_photon(   p.data(), p.size(), print_id ); 
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
    LOG(info) << "[" ; 

    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    unsigned x_total = 0 ; 
    for(unsigned i=0 ; i < photon_counts_per_genstep.size() ; i++) x_total += photon_counts_per_genstep[i] ; 

    const NP* gs = QEvent::MakeCountGensteps(photon_counts_per_genstep) ; 


    QEvent* evt = new QEvent  ; 
    evt->setGensteps(gs); 
    qs.generate_photon(evt);  

    std::vector<quad4> photon ; 
    evt->downloadPhoton(photon); 
    LOG(info) << " downloadPhoton photon.size " << photon.size() ; 

    qs.dump_photon( photon.data(), photon.size(), "f0,f1,i2,i3" ); 

    LOG(info) << "]" ; 
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


template<typename T>
void QSimTest<T>::main(int argc, char** argv, char test )
{
    unsigned num_default = SSys::getenvunsigned("NUM", 1000000u )  ;   
    unsigned num = argc > 1 ? std::atoi(argv[1]) : num_default ; 
    int ni_tranche_size = SSys::getenvint("NI_TRANCHE_SIZE", 100000 ); // default 100k usable with any GPU 
    int print_id = SSys::getenvint("PINDEX", -1 ); 

    LOG(info) 
        << " num_default " << num_default 
        << " num " << num 
        << " test " << test
        << " ni_tranche_size " << ni_tranche_size
        << " print_id " << print_id
        ; 


    T x0 = 80. ; 
    T x1 = 800. ; 
    unsigned nx = 721u ; 

    switch(test)
    {
        case 'F': rng_sequence(num, ni_tranche_size)         ; break ; 
        case 'S': wavelength('S', num)                       ; break ; 
        case 'C': wavelength('C', num)                       ; break ; 
        case 'P': scint_photon(num);                         ; break ; 
        case 'K': cerenkov_photon(num, print_id);            ; break ; 
        case 'E': cerenkov_photon_enprop(num, print_id);     ; break ; 
        case 'X': cerenkov_photon_expt(  num, print_id);     ; break ; 
        case 'G': generate_photon();                         ; break ; 
        case 'A': boundary_lookup_all()                      ; break ;  
        case 'W': boundary_lookup_line("Water", x0, x1, nx)  ; break ;  
        case 'L': boundary_lookup_line("LS",    x0, x1, nx)  ; break ;  
        case 'Y': prop_lookup(-1, -1.f,16.f,1701)            ; break ;  
        default : std::cout << "test unimplemented" << std::endl ; break ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    // TODO: this needs formalization, ie not in the test, and inclusion of QCerenkovIntegral prep 

    int create_dirs = 0 ; // 0:nop
    const char* cfbase =  SPath::Resolve(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), create_dirs ) ; 
    NP* icdf = NP::Load(cfbase, "CSGFoundry", "icdf.npy");  // need better naming, see CSG_GGeo_Convert::convertScintillatorLib 
    NP* bnd = NP::Load(cfbase, "CSGFoundry", "bnd.npy"); 

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

    char dtest = 'G' ; 
    char test = SSys::getenvchar("TEST", dtest); 
    char type = SSys::getenvchar("TYPE", 'F'); 
    if( test == 'X') type = 'D' ;   // forced double 

    if( type == 'F')
    { 
        QSim<float>::UploadComponents(icdf, bnd); 
        QSim<float> qs ; 
        QSimTest<float> qst(qs) ; 
        qst.main( argc, argv, test ); 
    }
    else if( type == 'D' )
    {
        QSim<double>::UploadComponents(icdf, bnd); 
        QSim<double> qs ; 
        QSimTest<double> qst(qs) ; 
        qst.main( argc, argv, test ); 
    }

    return 0 ; 
}
