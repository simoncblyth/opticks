#pragma once
/**
SPMT.h : summarize PMTSimParamData NPFold into the few arrays needed on GPU
============================================================================

Usage sketch
-------------

::

    SSim::get_spmt
    QSim::UploadComponents
    QPMT
    qpmt


WITH_CUSTOM4
--------------

Custom4 is used only for access to the header-only
TMM complex calculation.
Lack of the WITH_CUSTOM4 flag in sysrap IS A problem as
SPMT.h despite being header only is not directly included
into QPMT it gets compiled into SSim

Aims
----

1. replace and extend from PMTSim/JPMT.h based off the complete
   serialized PMT info rather than the partial NP_PROP_BASE rindex,
   thickness info loaded by JPMT.h.

   * DONE : (rindex,thickness) matches JPMT after ordering and scale fixes
   * TODO : include some pmtcat names in source metadata (in _PMTSimParamData?)
            to makes the order more explicit
   * DONE : compared get_stackspec scan from JPMT and SPMT

2. DONE :  (17612,2) PMT info arrays with [pmtcat 0/1/2, qescale]
3. DONE : update QPMT.hh/qpmt.h to upload the SPMT.h arrays and test them on device



Related developments
---------------------

* jcv PMTSimParamSvc     # junosw code that populates the below data core
* jcv PMTSimParamData    # PMT data core knocked out from PMTSimParamSvc for low dependency PMT data access
* jcv _PMTSimParamData   # code that serializes the PMT data core into NPFold
* SPMT.h                 # summarize PMT data into arrays for GPU upload
* QPMT.hh                # CPU QProp setup, uploads
* qpmt.h                 # on GPU use of PMT data
* j/Layr/JPMT.h          # earlier incarnation using "dirty" NP_PROP_BASE approach
* j/Layr/JPMTTest.sh

* Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/tests/PMTSimParamData.sh

  * python load the persisted PMTSimParamData

* Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/tests/PMTSimParamData_test.sh

  * _PMTSimParamData::Load from "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/extra/jpmt/PMTSimParamData"
  * test a few simple queries against the loaded PMTSimParamData
  * does _PMTSimParamData::Scan_pmtid_qe

* Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/tests/PMTAccessor_test.sh

  * PMTAccessor::Load from "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/extra/jpmt"
  * standalone CPU use of PMTAccessor to do the stack calc

* qudarap/tests/QPMTTest.sh

  * formerly used JPMT NP_PROP_BASE loading rindex and thickness, not qeshape and lcqs
  * DONE: add in SPMT.h rather than JPMT and extend to include qeshape and lcqs
  * DONE: on GPU interpolation check using QPMT


**/

#include <cstdio>
#include <csignal>

#include "NPFold.h"
#include "NPX.h"
#include "scuda.h"
#include "squad.h"
#include "ssys.h"
#include "sproc.h"
#include "spath.h"
#include "s_pmt.h"


#ifdef WITH_CUSTOM4
#include "C4MultiLayrStack.h"
#endif


struct SPMT_Total
{
    static constexpr const int FIELDS = 7 ;

    int CD_LPMT ;
    int SPMT ;
    int WP ;
    int WP_ATM_LPMT ;
    int WP_ATM_MPMT ;
    int WP_WAL_PMT ;
    int ALL ;

    int SUM() const ;
    std::string desc() const ;
};


inline int SPMT_Total::SUM() const
{
    return CD_LPMT + SPMT + WP + WP_ATM_LPMT + WP_ATM_MPMT + WP_WAL_PMT ;
}
inline std::string SPMT_Total::desc() const
{
    std::stringstream ss ;
    ss
       << "[SPMT_Total.desc\n"
#ifdef WITH_MPMT
       << " WITH_MPMT : YES\n"
#else
       << " WITH_MPMT : NO\n"
#endif
       << std::setw(16) << "CD_LPMT"     << std::setw(7) << CD_LPMT << "\n"
       << std::setw(16) << "SPMT"        << std::setw(7) << SPMT << "\n"
       << std::setw(16) << "WP"          << std::setw(7) << WP << "\n"
       << std::setw(16) << "WP_ATM_LPMT" << std::setw(7) << WP_ATM_LPMT << "\n"
       << std::setw(16) << "WP_ATM_MPMT" << std::setw(7) << WP_ATM_MPMT << "\n"
       << std::setw(16) << "WP_WAL_PMT"  << std::setw(7) << WP_WAL_PMT << "\n"
       << std::setw(16) << "ALL"         << std::setw(7) << ALL << "\n"
       << std::setw(16) << "SUM:"        << std::setw(7) << SUM() << "\n"
       << std::setw(16) << "SUM()==ALL"  << std::setw(7) << ( SUM() == ALL ? "YES" : "NO " ) << "\n"
       << std::setw(16) << "SUM()-ALL"   << std::setw(7) << ( SUM() - ALL ) << "\n"
       << "]SPMT_Total.desc\n"
       ;
    std::string str = ss.str() ;
    return str ;
}


struct SPMT_Num
{
    static constexpr const int FIELDS = 7 ;

    int m_nums_cd_lpmt ;
    int m_nums_cd_spmt ;
    int m_nums_wp_lpmt ;
    int m_nums_wp_atm_lpmt ;
    int m_nums_wp_wal_pmt ;
    int m_nums_wp_atm_mpmt ;
    int ALL ;

    int SUM() const ;
    std::string desc() const ;
};

inline int SPMT_Num::SUM() const
{
    return m_nums_cd_lpmt + m_nums_cd_spmt + m_nums_wp_lpmt + m_nums_wp_atm_lpmt + m_nums_wp_wal_pmt + m_nums_wp_atm_mpmt ;
}

inline std::string SPMT_Num::desc() const
{
    std::stringstream ss ;
    ss
       << "[SPMT_Num.desc\n"
#ifdef WITH_MPMT
       << " WITH_MPMT : YES\n"
#else
       << " WITH_MPMT : NO\n"
#endif
       << std::setw(20) << "m_nums_cd_lpmt     :" << std::setw(7) << m_nums_cd_lpmt << "\n"
       << std::setw(20) << "m_nums_cd_spmt     :" << std::setw(7) << m_nums_cd_spmt << "\n"
       << std::setw(20) << "m_nums_wp_lpmt     :" << std::setw(7) << m_nums_wp_lpmt << "\n"
       << std::setw(20) << "m_nums_wp_atm_lpmt :" << std::setw(7) << m_nums_wp_atm_lpmt << "\n"
       << std::setw(20) << "m_nums_wp_wal_pmt  :" << std::setw(7) << m_nums_wp_wal_pmt << "\n"
       << std::setw(20) << "m_nums_wp_atm_mpmt :" << std::setw(7) << m_nums_wp_atm_mpmt << "\n"
       << std::setw(20) << "ALL                :" << std::setw(7) << ALL << "\n"
       << std::setw(20) << "SUM                :" << std::setw(7) << SUM() << "\n"
       << std::setw(20) << "SUM==ALL           :" << std::setw(7) << ( SUM() == ALL ? "YES" : "NO " ) << "\n"
       << "]SPMT_Num.desc\n"
       ;
    std::string str = ss.str() ;
    return str ;
}





struct SPMT
{
    static constexpr const char* _level = "SPMT__level" ;
    static const int level ;
    static constexpr const float hc_eVnm = 1239.84198433200208455673  ;

    enum { L0, L1, L2, L3 } ;
    enum { RINDEX, KINDEX } ;

    struct LCQS { int lc ; float qs ; } ;


    static constexpr const float EN0 = 1.55f ;
    static constexpr const float EN1 = 4.20f ;  // 15.5
    static constexpr const int   N_EN = 420 - 155 + 1 ;

    static constexpr const float WL0 = 440.f ;
    static constexpr const float WL1 = 440.f ;
    static constexpr const int   N_WL = 1 ;


    static constexpr const char* PATH = "$CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt" ;

    // TODO: get these from s_pmt.h also
    static constexpr int NUM_PMTCAT = 3 ; // (NNVT, HAMA, NNVT_HiQE)
    static constexpr int NUM_LAYER = 4 ;  // (Pyrex, ARC, PHC, Vacuum)
    static constexpr int NUM_PROP = 2 ;   // (RINDEX,KINDEX) real and imaginary parts of the index

    // below three can be changed via envvars
    static const int N_LPMT ;   // N_LPMT must be less than or equal to NUM_CD_LPMT
    static const int N_MCT ;
    static const int N_SPOL ;

    static constexpr const char* QE_shape_PMTCAT_NAMES = "QEshape_NNVT.npy,QEshape_R12860.npy,QEshape_NNVT_HiQE.npy" ;
    // follows PMTCategory kPMT enum order but using QEshape array naming convention
    // because there is no consistently used naming convention have to do these dirty things

    static constexpr const char* QE_shape_S_PMTCAT_NAMES = "QEshape_HZC.npy" ;


    static constexpr const char* CE_theta_PMTCAT_NAMES = "CE_NNVTMCP.npy,CE_R12860.npy,CE_NNVTMCP_HiQE.npy" ;
    static constexpr const char* CE_costh_PMTCAT_NAMES = "CECOS_NNVTMCP.npy,CECOS_R12860.npy,CECOS_NNVTMCP_HiQE.npy" ;


    static const NPFold* CreateFromJPMTAndSerialize(const char* path=nullptr);
    static SPMT* CreateFromJPMT(const char* path=nullptr);
    SPMT(const NPFold* jpmt);
    double get_pmtid_qescale( int pmtid ) const ;

    void init();

    void init_total();

    static NP* FAKE_WHEN_MISSING_PMTParamData_serialize_pmtNum();
    static constexpr const char* SPMT__init_pmtNum_FAKE_WHEN_MISSING = "SPMT__init_pmtNum_FAKE_WHEN_MISSING" ;
    static int init_pmtNum_FAKE_WHEN_MISSING ;
    void init_pmtNum();

    void init_pmtCat();

    void init_lpmtCat();
    void init_qeScale();

    void init_rindex_thickness();
    static NP* MakeCatPropArrayFromFold( const NPFold* fold, const char* _names, std::vector<const NP*>& v_prop, double domain_scale );
    void init_qeshape();
    void init_s_qeshape();
    void init_cetheta();
    void init_cecosth();



    void init_lcqs();
    void init_s_qescale();

    static int TranslateCat(int lpmtcat);

    std::string descDetail() const ;
    std::string desc() const ;

    bool is_complete() const ;
    NPFold* serialize() const ;
    NPFold* serialize_() const ;

    template<typename T>
    static T GetFrac(int i, int ni );

    template<typename T>
    static T GetValueInRange(int j, int nj, T X0, T X1 );


    float get_energy(int j, int nj) const ;
    float get_wavelength(int j, int nj) const ;

    float get_minus_cos_theta_linear_angle(int k, int nk ) const ;
    float get_minus_cos_theta_linear_cosine(int k, int nk ) const ;

    float get_rindex(int cat, int layr, int prop, float energy_eV) const ;
    NP* get_rindex() const ;

    float get_qeshape(int cat, float energy_eV) const ;
    float get_s_qeshape(int cat, float energy_eV) const ;
    NP* get_qeshape() const ;
    NP* get_s_qeshape() const ;

    float get_thickness_nm(int cat, int layr) const ;
    NP* get_thickness_nm() const ;


    void get_lpmtid_stackspec( quad4& spec, int lpmtid, float energy_eV) const ;  // EXPT

    static constexpr const char* LPMTID_LIST = "0,10,55,98,100,137,1000,10000,17611,50000,51000,52000,52399" ;
    static const NP* Make_LPMTID_LIST();


#ifdef WITH_CUSTOM4

    /**
    SPMTData : Full calculation details for debugging
    ----------------------------------------------------
    **/
    struct SPMTData
    {
        float4             args ;         //  (1,4)
        float4             ARTE ;         //  (1,4)
        float4             extra ;        //  (1,4)
        quad4              spec ;         //  (4,4)
        Stack<float,4>     stack ;        // (44,4)
                                          // ------
                                          // (51,4)

        const float* cdata() const { return &args.x ; }
    };

    void annotate( NP* art ) const ;

    void get_ARTE(SPMTData& pd, int lpmtid, float wavelength_nm, float minus_cos_theta, float dot_pol_cross_mom_nrm ) const ;


    NPFold* make_c4scan() const ;
#endif

    void get_stackspec( quad4& spec, int cat, float energy_eV) const ;
    NP*  get_stackspec() const ;

    // lpmtidx : contiguous for CD_LPMT + WP
    // lpmtid : non-contiguous CD_LPMT, SPMT, WP

    int  get_lpmtcat_from_lpmtidx( int lpmtidx ) const ;
    NP*  get_lpmtcat_from_lpmtidx() const ;

    int  get_lpmtcat_from_lpmtid(  int lpmtid  ) const ;
    int  get_lpmtcat_from_lpmtid(  int* lpmtcat, const int* lpmtid , int num ) const ;
    NP*  get_lpmtcat_from_lpmtid(const NP* lpmtid) const ;




    float get_qescale_from_lpmtid(int lpmtid) const ;
    int   get_copyno_from_lpmtid(int lpmtid) const  ;

    float get_qescale_from_lpmtidx(int lpmtidx) const ;
    int   get_copyno_from_lpmtidx(int lpmtidx) const ;

    NP*   get_qescale_from_lpmtidx() const ;
    NP*   get_copyno_from_lpmtidx() const ;

    void  get_lcqs_from_lpmtid(  int& lc, float& qs, int lpmtid ) const ;
    void  get_lcqs_from_lpmtidx( int& lc, float& qs, int lpmtidx ) const ;
    NP*   get_lcqs_from_lpmtidx() const ;

    float get_s_qescale_from_spmtid( int pmtid ) const ;
    NP*   get_s_qescale_from_spmtid() const ;


    float get_pmtcat_qe(int cat, float energy_eV) const ;
    NP*   get_pmtcat_qe() const ;

    float get_lpmtidx_qe(int lpmtidx, float energy_eV) const ;
    NP*   get_lpmtidx_qe() const ;

    NPFold* make_testfold() const ;



    const char* ExecutableName ;

    const NPFold* jpmt ;
    const NPFold* PMT_RINDEX ;       // PyrexRINDEX and VacuumRINDEX
    const NPFold* PMTSimParamData ;  // lpmtCat for 17612 and many other arrays
    const NPFold* PMTParamData ;     // only pmtCat for 45612

    SPMT_Num num = {} ;
    SPMT_Total total = {} ;

    const NPFold* MPT ;              // ARC_RINDEX, ARC_KINDEX, PHC_RINDEX, PHC_KINDEX
    const NPFold* CONST ;            // ARC_THICKNESS PHC_THICKNESS
    const NPFold* QE_shape ;
    const NPFold* CE_theta ;
    const NPFold* CE_costh ;

    std::vector<const NP*> v_rindex ;
    std::vector<const NP*> v_qeshape ;
    std::vector<const NP*> v_cetheta ;
    std::vector<const NP*> v_cecosth ;

    std::vector<const NP*> v_s_qeshape ;


    std::vector<LCQS>      v_lcqs ;    // NUM_CD_LPMT + NUM_WP

    NP* rindex ;    // (NUM_PMTCAT, NUM_LAYER, NUM_PROP, N_EN, 2:[energy,value] )
    NP* qeshape ;   // (NUM_PMTCAT, NUM_SAMPLES~44, 2:[energy,value] )
    NP* cetheta ;   // (NUM_PMTCAT, NUM_SAMPLES~9, 2:[angle_radians,value] )
    NP* cecosth ;   // (NUM_PMTCAT, NUM_SAMPLES~9, 2:[cosine_angle,value] )
    NP* lcqs ;      // (NUM_CD_LPMT + NUM_WP, 2)

    NP* thickness ; // (NUM_PMTCAT, NUM_LAYER, 1:value )
    float* tt ;

    const NP* pmtCat ;
    int pmtCat_ni ;
    const int* pmtCat_v ;
    const NP* pmtNum ;

    const NP* lpmtCat ;
    const int* lpmtCat_v ;

    const NP* qeScale ;
    const double* qeScale_v ;

    NP* s_qeshape ;  // (NUM_PMTCAT:1, NUM_SAMPLES~60, 2:[energy,value] )
    NP* s_qescale ;  // (NUM_SPMT, 1)
    float* s_qescale_v ;

};


const int SPMT::level  = ssys::getenvint(_level, 0);
const int SPMT::N_LPMT = ssys::getenvint("N_LPMT", 1 ); // 10 LPMT default for fast scanning
const int SPMT::N_MCT  = ssys::getenvint("N_MCT",  180 );  // "AOI" (actually mct) scan points from -1. to 1.
const int SPMT::N_SPOL = ssys::getenvint("N_SPOL", 1 ); // polarization scan points from S-pol to P-pol

inline const NPFold* SPMT::CreateFromJPMTAndSerialize(const char* path) // static
{
    SPMT* spmt = SPMT::CreateFromJPMT(path);
    const NPFold* fold = spmt ? spmt->serialize() : nullptr ;
    return fold ;
}

/**
SPMT::CreateFromJPMT  (formerly SPMT::Load)
---------------------------------------------

Default path is::

    $CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt

Which is a directory expected to contain sub-dirs::

     PMT_RINDEX
     PMTSimParamData

     PMTParamData   # this also there, but not used ?


**/


inline SPMT* SPMT::CreateFromJPMT(const char* path_)
{
    if(level > 0) printf("[SPMT::CreateFromJPMT [%s]\n", ( path_ == nullptr ? "path_-null" : path_ ));

    const char* path = spath::Resolve( path_ != nullptr ? path_ : PATH ) ;
    bool unresolved = sstr::StartsWith(path,"CFBaseFromGEOM");
    if(unresolved) printf("-SPMT::CreateFromJPMT unresolved path[%s]\n", path) ;
    NPFold* fold = NPFold::LoadIfExists(path) ;
    if(level > 0) printf("-SPMT::CreateFromJPMT path %s \n", ( path == nullptr ? "path-null" : path ) );
    if(level > 0) printf("-SPMT::CreateFromJPMT fold %s \n", ( fold == nullptr ? "fold-null" : "fold-ok" ) );
    SPMT* sp = fold ? new SPMT(fold) : nullptr ;

    if(level > 0) printf("]SPMT::CreateFromJPMT sp[%s]\n", ( sp ? "YES" : "NO " ) );
    return sp ;
}

inline SPMT::SPMT(const NPFold* jpmt_)
    :
    ExecutableName(sproc::ExecutableName()),
    jpmt(jpmt_),
    PMT_RINDEX(     jpmt ? jpmt->get_subfold("PMT_RINDEX")      : nullptr ),
    PMTSimParamData(jpmt ? jpmt->get_subfold("PMTSimParamData") : nullptr ),
    PMTParamData(   jpmt ? jpmt->get_subfold("PMTParamData")    : nullptr ),
    MPT(            PMTSimParamData ? PMTSimParamData->get_subfold("MPT")   : nullptr ),
    CONST(          PMTSimParamData ? PMTSimParamData->get_subfold("CONST") : nullptr ),
    QE_shape(       PMTSimParamData ? PMTSimParamData->get_subfold("QEshape") : nullptr ),
    CE_theta(       PMTSimParamData ? PMTSimParamData->get_subfold("CEtheta") : nullptr ),
    CE_costh(       PMTSimParamData ? PMTSimParamData->get_subfold("CECOStheta") : nullptr ),
    v_lcqs(0),
    rindex(nullptr),
    qeshape(nullptr),
    cetheta(nullptr),
    cecosth(nullptr),
    lcqs(nullptr),
    thickness(NP::Make<float>(NUM_PMTCAT, NUM_LAYER, 1)),
    tt(thickness->values<float>()),
    pmtCat( PMTParamData ? PMTParamData->get("pmtCat") : nullptr ),
    pmtCat_ni( pmtCat ? pmtCat->shape[0] : 0 ),
    pmtCat_v( pmtCat ? pmtCat->cvalues<int>() : nullptr ),
    pmtNum( PMTParamData ? PMTParamData->get("pmtNum") : nullptr ),
    lpmtCat( PMTSimParamData ? PMTSimParamData->get("lpmtCat")  : nullptr ),
    lpmtCat_v( lpmtCat ? lpmtCat->cvalues<int>() : nullptr ),
    qeScale( PMTSimParamData ? PMTSimParamData->get("qeScale")  : nullptr ),
    qeScale_v( qeScale ? qeScale->cvalues<double>() : nullptr ),
    s_qeshape(nullptr),
    s_qescale(NP::Make<float>(s_pmt::NUM_SPMT,1)),
    s_qescale_v( s_qescale ? s_qescale->values<float>() : nullptr )
{
    init();
}


/**
SPMT::get_pmtid_qescale
------------------------

qeScale from  _PMTSimParamData::serialize uses s_pmt.h contiguousidx order : CD_LPMT, WP, SPMT

**/

inline double SPMT::get_pmtid_qescale( int pmtid ) const
{
    int contiguousidx = s_pmt::contiguousidx_from_pmtid( pmtid );
    float qesc = qeScale_v[contiguousidx] ;
    return qesc ;
}




/**
SPMT::init
-----------

Converts PMTSimParamData NPFold into arrays ready
for upload to GPU (by QPMT) with various changes:

1. only 3 LPMT categories selected
2. energy domain scaled to use eV
3. layer thickness changed to use nm
4. arrays are narrowed from double to float

**/

inline void SPMT::init()
{
    init_total();

    init_pmtNum();
    init_pmtCat();

    init_lpmtCat();
    init_qeScale();

    init_rindex_thickness();
    init_qeshape();
    init_s_qeshape();
    init_cetheta();
    init_cecosth();

    init_lcqs();
    init_s_qescale();
}



/**
SPMT::init_total
-----------------

OLD::

    SPMT::init_total SPMT_Total CD_LPMT: 17612 SPMT: 25600 WP:  2400 ALL: 45612

SPMT_test.sh::

    In [21]: s.jpmt.PMTSimParamData.pmtTotal
    Out[21]: array([17612, 25600,  2400,   348,     0,     5, 45965], dtype=int32)

    In [22]: s.jpmt.PMTSimParamData.pmtTotal_names
    Out[22]: array(['PmtTotal', 'PmtTotal_SPMT', 'PmtTotal_WP', 'PmtTotal_WP_ATM_LPMT', 'PmtTotal_WP_ATM_MPMT', 'PmtTotal_WP_WAL_PMT', 'PmtTotal_ALL'], dtype='<U20')


    In [8]: 17612 + 25600 + 2400 + 348 + 5
    Out[8]: 45965


The pmtTotal do not yet have the 600 MPMT

Provenance of the pmtTotal::

    jcv PMTSimParamData
    jcv _PMTSimParamData
    jcv PMTSimParamSvc


**/

inline void SPMT::init_total()
{
    if( PMTSimParamData == nullptr )
    {
        std::cout << "SPMT::init_total exit as PMTSimParamData is NULL\n" ;
        return ;
    }
    assert( PMTSimParamData );
    const NP* pmtTotal = PMTSimParamData->get("pmtTotal") ; // (7,)   formerly (4,)

    if(level > 0) std::cout
       << "SPMT::init_total"
       << " pmtTotal.sstr " << ( pmtTotal ? pmtTotal->sstr() : "-" )
       << "\n"
       ;

    assert( pmtTotal && pmtTotal->uifc == 'i' && pmtTotal->ebyte == 4 );
    assert( pmtTotal->shape.size() == 1 && pmtTotal->shape[0] == SPMT_Total::FIELDS );
    assert( pmtTotal->names.size() == SPMT_Total::FIELDS );
    [[maybe_unused]] const int* pmtTotal_v = pmtTotal->cvalues<int>();

    std::vector<std::string> xnames = {
         "PmtTotal",
         "PmtTotal_SPMT",
         "PmtTotal_WP",
         "PmtTotal_WP_ATM_LPMT",
         "PmtTotal_WP_ATM_MPMT",
         "PmtTotal_WP_WAL_PMT",
         "PmtTotal_ALL"
       };
    assert( pmtTotal->names == xnames );


    total.CD_LPMT     = pmtTotal->get_named_value<int>("PmtTotal",      -1) ;
    total.SPMT        = pmtTotal->get_named_value<int>("PmtTotal_SPMT", -1) ;
    total.WP          = pmtTotal->get_named_value<int>("PmtTotal_WP",   -1) ;
    total.WP_ATM_LPMT = pmtTotal->get_named_value<int>("PmtTotal_WP_ATM_LPMT",   -1) ;
    total.WP_ATM_MPMT = pmtTotal->get_named_value<int>("PmtTotal_WP_ATM_MPMT",   -1) ;
    total.WP_WAL_PMT  = pmtTotal->get_named_value<int>("PmtTotal_WP_WAL_PMT",   -1) ;
    total.ALL         = pmtTotal->get_named_value<int>("PmtTotal_ALL",  -1) ;

    if(level > 0) std::cout << "SPMT::init_total " << total.desc() << "\n" ;

    bool x_total_ALL = total.ALL == total.SUM() ;
    if(!x_total_ALL) std::cerr
        << "SPMT::init_total"
        << " x_total_ALL " << ( x_total_ALL ? "YES" : "NO " )
        << total.desc()
        << "\n"
        ;

    assert( x_total_ALL );

    assert( pmtTotal_v[0]     == total.CD_LPMT );
    assert( total.CD_LPMT     == s_pmt::NUM_CD_LPMT );
    assert( total.SPMT        == s_pmt::NUM_SPMT ) ;
    assert( total.WP          == s_pmt::NUM_WP  ) ;
    assert( total.WP_ATM_LPMT == s_pmt::NUM_WP_ATM_LPMT ) ;
    assert( total.WP_WAL_PMT  == s_pmt::NUM_WP_WAL_PMT ) ;

#ifdef WITH_MPMT
    bool x_MPMT = total.WP_ATM_MPMT == s_pmt::NUM_WP_ATM_MPMT ;
#else
    bool x_MPMT = total.WP_ATM_MPMT == 0 ;
#endif
    if(!x_MPMT) std::cerr
        << "SPMT::init_total"
#ifdef WITH_MPMT
        << " WITH_MPMT "
#else
        << " NOT:WITH_MPMT "
#endif
        << " x_MPMT " << ( x_MPMT ? "YES" : "NO " )
        << " total.WP_ATM_MPMT " << total.WP_ATM_MPMT
        << " s_pmt::NUM_WP_ATM_MPMT " << s_pmt::NUM_WP_ATM_MPMT
        << total.desc()
        ;
    assert( x_MPMT ) ;

}



/**
SPMT::FAKE_WHEN_MISSING_PMTParamData_serialize_pmtNum
-------------------------------------------------------

Attempt to give backward compatibility for older junosw branch
that lacks the pmtNum by faking it.

**/

inline NP* SPMT::FAKE_WHEN_MISSING_PMTParamData_serialize_pmtNum() // static
{
    NPX::KV<int> kv ;
    kv.add("m_nums_cd_lpmt", s_pmt::NUM_CD_LPMT );
    kv.add("m_nums_cd_spmt", s_pmt::NUM_SPMT );
    kv.add("m_nums_wp_lpmt", s_pmt::NUM_WP );
    kv.add("m_nums_wp_atm_lpmt", s_pmt::NUM_WP_ATM_LPMT );
    kv.add("m_nums_wp_wal_pmt", s_pmt::NUM_WP_WAL_PMT );
    kv.add("m_nums_wp_atm_mpmt",  s_pmt::NUM_WP_ATM_MPMT ); // HMM _ALREADY ?
    return kv.values() ;
}


/**
SPMT::init_pmtNum
------------------

pmtNum and pmtCat info both come from the serialization of PMTParamData
so they are required to correspond to each other

**/

inline int SPMT::init_pmtNum_FAKE_WHEN_MISSING = ssys::getenvint(SPMT__init_pmtNum_FAKE_WHEN_MISSING, 1 );  // default to doing this

inline void SPMT::init_pmtNum()
{
    std::cout
       << "SPMT::init_pmtNum"
       << " pmtNum.sstr " <<  (pmtNum ? pmtNum->sstr() : "-"  )
       << " init_pmtNum_FAKE_WHEN_MISSING " << init_pmtNum_FAKE_WHEN_MISSING
       << "\n"
       ;
    if(!pmtNum)
    {
        if( init_pmtNum_FAKE_WHEN_MISSING == 0 )
        {
            return ;
        }
        else
        {
            pmtNum = FAKE_WHEN_MISSING_PMTParamData_serialize_pmtNum();
        }
    }

    std::vector<std::string> xnames = {
         "m_nums_cd_lpmt",
         "m_nums_cd_spmt",
         "m_nums_wp_lpmt",
         "m_nums_wp_atm_lpmt",
         "m_nums_wp_wal_pmt",
         "m_nums_wp_atm_mpmt"
       };
    assert( pmtNum->names == xnames );

    num.m_nums_cd_lpmt      = pmtNum->get_named_value<int>("m_nums_cd_lpmt", -1) ;
    num.m_nums_cd_spmt      = pmtNum->get_named_value<int>("m_nums_cd_spmt", -1) ;
    num.m_nums_wp_lpmt      = pmtNum->get_named_value<int>("m_nums_wp_lpmt", -1) ;
    num.m_nums_wp_atm_lpmt  = pmtNum->get_named_value<int>("m_nums_wp_atm_lpmt", -1) ;
    num.m_nums_wp_wal_pmt   = pmtNum->get_named_value<int>("m_nums_wp_wal_pmt", -1) ;
    num.m_nums_wp_atm_mpmt  = pmtNum->get_named_value<int>("m_nums_wp_atm_mpmt", -1) ;
    num.ALL = num.SUM();

    std::cout
        << "[SPMT::init_pmtNum\n"
        << num.desc()
        << "]SPMT::init_pmtNum\n"
        ;

    assert( num.m_nums_cd_lpmt     == s_pmt::NUM_CD_LPMT );
    assert( num.m_nums_cd_spmt     == s_pmt::NUM_SPMT ) ;
    assert( num.m_nums_wp_lpmt     == s_pmt::NUM_WP ) ;
    assert( num.m_nums_wp_atm_lpmt == s_pmt::NUM_WP_ATM_LPMT ) ;
    assert( num.m_nums_wp_wal_pmt  == s_pmt::NUM_WP_WAL_PMT ) ;
    //assert( num.m_nums_wp_atm_mpmt == s_pmt::NUM_WP_ATM_MPMT_ALREADY ) ;

}


/**
SPMT::init_pmtCat
--------------------

Comes from _PMTParamData::serialize::

    f->add("pmtCat", NPX::ArrayFromDiscoMapUnordered<int>(data.m_pmt_categories));

::

    In [2]: f.pmtCat[:17612]    ## CD_LPMT
    Out[2]:
    array([[    0,     1],
           [    1,     3],
           [    2,     1],
           [    3,     3],
           [    4,     1],
           ...,
           [17607,     1],
           [17608,     3],
           [17609,     1],
           [17610,     1],
           [17611,     1]], shape=(17612, 2), dtype=int32)

    In [7]: f.pmtCat[17612:17612+25600]    ## S_PMT
    Out[7]:
    array([[20000,     2],
           [20001,     2],
           [20002,     2],
           [20003,     2],
           [20004,     2],
           ...,
           [45595,     2],
           [45596,     2],
           [45597,     2],
           [45598,     2],
           [45599,     2]], shape=(25600, 2), dtype=int32)

    In [8]: f.pmtCat[17612+25600:17612+25600+2400]   ## WP_PMT
    Out[8]:
    array([[50000,     3],
           [50001,     0],
           [50002,     0],
           [50003,     3],
           [50004,     0],
           ...,
           [52395,     0],
           [52396,     3],
           [52397,     3],
           [52398,     3],
           [52399,     0]], shape=(2400, 2), dtype=int32)



2025/06 WP_ATM and WP_WAL have been added to pmtCat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PMTParamSvc::init_IDService_WP_ATM_LPMT
PMTParamSvc::init_IDService_WP_WAL_PMT


From SPMT_test.sh::

    In [16]: s.jpmt.PMTParamData.pmtCat[17612+25600+2400:17612+25600+2400+348]   ## WP_ATM_LPMT
    Out[16]:
    array([[52400,     3],
           [52401,     0],
           [52402,     0],
           [52403,     3],
           [52404,     3],
           ...
           [52744,     0],
           [52745,     3],
           [52746,     0],
           [52747,     0]], dtype=int32)


    In [18]: s.jpmt.PMTParamData.pmtCat[17612+25600+2400+348:17612+25600+2400+348+5]   ## WP_WAL_PMT
    Out[18]:
    array([[54000,     3],
           [54001,     0],
           [54002,     0],
           [54003,     0],
           [54004,     0]], dtype=int32)



2026/01 yupd_bottompipe_adjust
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In older branch without MPMT fully impl they are already in pmtCat::

    In [27]: np.all( f.pmtCat[17612+25600+2400+348:17612+25600+2400+348+600][:,1] == 4 )
    Out[27]: np.True_

    In [28]: f.pmtCat[17612+25600+2400+348+600:17612+25600+2400+348+600+5]
    Out[28]:
    array([[54000,     3],
           [54001,     0],
           [54002,     0],
           [54003,     0],
           [54004,     0]], dtype=int32)

So the implicit contiguous index of pmtCat follows the order, and included MPMT::

    CD-LPMT:17612
    CD-SPMT:25600
    WP-LPMT:2400
    WP-Atmosphere-LPMT:348
    WP-Atmosphere-MPMT:600
    WP-Water-attenuation-length:5


**/


void SPMT::init_pmtCat()
{
    bool expected_type  = pmtCat && pmtCat->uifc == 'i' && pmtCat->ebyte == 4 ;
    bool expected_shape = pmtCat && pmtCat->shape.size() == 2 && pmtCat->shape[0] == num.ALL && pmtCat->shape[1] == 2 ;

    if(!expected_shape || !expected_type) std::cerr
       << "SPMT::init_pmtCat"
       << " expected_type " << ( expected_type ? "YES" : "NO " )
       << " expected_shape[first-dim-is-num.ALL] " << ( expected_shape ? "YES" : "NO " )
       << " pmtCat.shape[0] " << pmtCat->shape[0]
       << " num.ALL " << num.ALL
       << " pmtCat " << ( pmtCat ? pmtCat->sstr() : "-" )
       << " num " << num.desc()
       << "\n"
       ;

    if(!pmtCat) return ;
    assert( expected_type );
    assert( expected_shape );
    assert( pmtCat_v );
}






void SPMT::init_lpmtCat()
{
    if(!lpmtCat) return ;
    assert( lpmtCat && lpmtCat->uifc == 'i' && lpmtCat->ebyte == 4 );
    assert( lpmtCat->shape[0] == s_pmt::NUM_CD_LPMT );
    assert( lpmtCat_v );
}

void SPMT::init_qeScale()
{
    if(!qeScale) return ;
    assert( qeScale && qeScale->uifc == 'f' && qeScale->ebyte == 8 );
    assert( qeScale_v );

#ifdef WITH_MPMT
    int expect_qeScale_items = s_pmt::NUM_ALL  ;
#else
    assert( s_pmt::NUM_CD_LPMT + s_pmt::NUM_SPMT + s_pmt::NUM_WP + s_pmt::NUM_WP_ATM_LPMT + s_pmt::NUM_WP_WAL_PMT  == s_pmt::NUM_ALL_EXCEPT_MPMT );
    int expect_qeScale_items = s_pmt::NUM_ALL_EXCEPT_MPMT ;
#endif
    bool qeScale_shape_expect = qeScale->shape[0] == expect_qeScale_items ;

    if(!qeScale_shape_expect)
    std::cerr
        << "SPMT::init_qeScale"
        << " qeScale.sstr " << ( qeScale ? qeScale->sstr() : "-" )
        << " qeScale_shape_expect " << ( qeScale_shape_expect ? "YES" : "NO " )
        << " expect_qeScale_items " << expect_qeScale_items
        << "\n"
        << s_pmt::desc()
        ;

   assert( qeScale_shape_expect  );
}



/**
SPMT::init_rindex_thickness
------------------------------

1. this is similar to JPMT::init_rindex_thickness
2. energy domain is scaled to use eV units prior to narrowing from double to float
3. thickness values are scaled to use nm, not m
4. originally there was dumb vacuum rindex with (4,2) for a constant.
   Fixed this to (2,2) with NP::MakePCopyNotDumb. Before the fix::

    In [19]: t.rindex[0,3]
    Out[19]:
    array([[[ 1.55,  1.  ],
            [ 6.2 ,  1.  ],
            [10.33,  1.  ],
            [15.5 ,  1.  ],
            [ 0.  ,  0.  ],
            [ 0.  ,  0.  ],

**/

inline void SPMT::init_rindex_thickness()
{
    if(MPT == nullptr || CONST == nullptr) return ;
    int MPT_sub = MPT->get_num_subfold() ;
    int CONST_items = CONST->num_items() ;

    bool MPT_sub_expect = MPT_sub == NUM_PMTCAT  ;
    if(!MPT_sub_expect) std::raise(SIGINT);
    assert( MPT_sub_expect );    // NUM_PMTCAT:3

    bool CONST_items_expect = CONST_items == NUM_PMTCAT  ;

    if(!CONST_items_expect) std::cerr
        << "SPMT::init_rindex_thickness"
        << " CONST_items_expect " << ( CONST_items_expect ? "YES" : "NO " )
        << " CONST_items " << CONST_items
        << " NUM_PMTCAT " << NUM_PMTCAT
        << " MPT_sub " << MPT_sub
        << "\n"
        ;

    if(!CONST_items_expect) std::raise(SIGINT);
    assert( CONST_items_expect );

    double dscale = 1e-6 ;  // make energy domain scale consistent

    for(int i=0 ; i < NUM_PMTCAT ; i++)
    {
        //const char* name = MPT->get_subfold_key(i) ;
        NPFold* pmtcat = MPT->get_subfold(i);
        const NP* pmtconst = CONST->get_array(i);

        for(int j=0 ; j < NUM_LAYER ; j++)  // NUM_LAYER:4
        {
            for(int k=0 ; k < NUM_PROP ; k++)   // NUM_PROP:2 (RINDEX,KINDEX)
            {
                const NP* v = nullptr ;
                switch(j)
                {
                   case 0: v = (k == 0 ? PMT_RINDEX->get("PyrexRINDEX")  : NP::ZEROProp<double>(dscale) ) ; break ;
                   case 1: v = pmtcat->get( k == 0 ? "ARC_RINDEX" : "ARC_KINDEX"   )              ; break ;
                   case 2: v = pmtcat->get( k == 0 ? "PHC_RINDEX" : "PHC_KINDEX"   )              ; break ;
                   case 3: v = (k == 0 ? PMT_RINDEX->get("VacuumRINDEX") : NP::ZEROProp<double>(dscale) ) ; break ;
                }

                NP* vc = NP::MakePCopyNotDumb<double>(v);  // avoid dumb constants with > 2 domain values
                vc->pscale(1e6,0);

                NP* vn = NP::MakeWithType<float>(vc);   // narrow

                v_rindex.push_back(vn) ;
            }

            double d = 0. ;
            //double scale = 1e9 ; // express thickness in nm (not meters)
            double scale = 1e6 ;  // HUH: why source scale changed ?
            switch(j)
            {
               case 0: d = 0. ; break ;
               case 1: d = scale*pmtconst->get_named_value<double>("ARC_THICKNESS", -1) ; break ;
               case 2: d = scale*pmtconst->get_named_value<double>("PHC_THICKNESS", -1) ; break ;
               case 3: d = 0. ; break ;
            }
            tt[i*NUM_LAYER + j] = float(d) ;
        }
    }
    rindex = NP::Combine(v_rindex);
    const std::vector<NP::INT>& shape = rindex->shape ;
    assert( shape.size() == 3 );
    rindex->change_shape( NUM_PMTCAT, NUM_LAYER, NUM_PROP, shape[shape.size()-2], shape[shape.size()-1] );
}



/**
SPMT::MakeCatPropArrayFromFold
--------------------------------

1. selects just the LPMT relevant 3 PMTCAT
2. domain scaling is done whilst still in double precision
   and before combination to avoid stomping on last column int
   (HMM NP::Combine may have special handling to preserve that now?)

**/

inline NP* SPMT::MakeCatPropArrayFromFold( const NPFold* fold, const char* _names, std::vector<const NP*>& v_prop, double domain_scale ) // static
{
    if(fold == nullptr) return nullptr ;
    int num_items = fold->num_items();
    if(level > 0) std::cout << "SPMT::MakeCatPropArrayFromFold num_items : " << num_items << "\n" ;

    std::vector<std::string> names ;
    U::Split(_names, ',', names );

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* k = names[i].c_str();
        const NP* v = fold->get(k);
        if(level > 0) std::cout << std::setw(20) << k << " : " << ( v ? v->sstr() : "-" ) << std::endl ;

        NP* vc = v->copy();
        vc->pscale(domain_scale, 0);
        NP* vn = NP::MakeWithType<float>(vc);   // narrow

        v_prop.push_back(vn) ;
    }
    NP* catprop = NP::Combine(v_prop);
    catprop->set_names(names);
    return catprop ;
}


inline void SPMT::init_qeshape()
{
    double domain_scale = 1e6 ; // MeV to eV
    qeshape = MakeCatPropArrayFromFold( QE_shape, QE_shape_PMTCAT_NAMES, v_qeshape, domain_scale );
}
inline void SPMT::init_s_qeshape()
{
    double domain_scale = 1e6 ; // MeV to eV
    s_qeshape = MakeCatPropArrayFromFold( QE_shape, QE_shape_S_PMTCAT_NAMES, v_s_qeshape, domain_scale );
}




inline void SPMT::init_cetheta()
{
    double domain_scale = 1. ;
    cetheta = MakeCatPropArrayFromFold( CE_theta, CE_theta_PMTCAT_NAMES, v_cetheta, domain_scale );
}
inline void SPMT::init_cecosth()
{
    double domain_scale = 1. ;
    cecosth = MakeCatPropArrayFromFold( CE_costh, CE_costh_PMTCAT_NAMES, v_cecosth, domain_scale );
}








/**
SPMT::init_lcqs
-----------------

1. get lpmtCat, qeScale arrays from PMTSimParamData NPFold
2. check appropriate sizes with info for all NUM_CD_LPMT 17612
3. populate v_lcqs vector of LCQS struct holding int:lc
   "local 0/1/2 pmtcat" and float:qeScale
4. convert the vector of LCQS struct into lcqs array


s.lcqs
~~~~~~~

::

    In [3]: s.lcqs[:,1].view(np.float32)
    Out[3]:
    array([1.025, 1.201, 1.238, 1.172, 1.137, 1.177, 1.098, 1.03 , 1.245, 1.026, 1.162, 1.105, 1.046, 1.042, 1.186, 1.059, ..., 0.919, 0.972, 1.005, 0.932, 0.993, 1.011, 1.019, 1.023, 1.026, 1.104,
           0.991, 1.006, 0.98 , 1.053, 1.003, 1.009], shape=(20965,), dtype=float32)

    In [4]: s.lcqs[:,1].view(np.float32).min()
    Out[4]: np.float32(0.38050073)

    In [5]: s.lcqs[:,1].view(np.float32).max()
    Out[5]: np.float32(1.5862682)

    In [6]: s.lcqs[:,0]
    Out[6]:
    array([  1,   2,   1,   2,   1,   2,   1,   1,   2,   1,   2,   1,   1,   1,   2,   1, ..., -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99,   2,   0,   0,   0,   0],
          shape=(20965,), dtype=int32)

    In [7]: 17612 + 2400 + 348 + 600 + 5
    Out[7]: 20965


    In [9]: np.c_[np.unique(s.lcqs[0:17612,0],return_counts=True)]
    Out[9]:
    array([[   0, 2738],
           [   1, 4955],
           [   2, 9919]])

    In [10]: np.c_[np.unique(s.lcqs[17612:17612+2400,0],return_counts=True)]
    Out[10]:
    array([[   0, 1836],
           [   2,  564]])

    In [11]: np.c_[np.unique(s.lcqs[17612+2400:17612+2400+348,0],return_counts=True)]
    Out[11]:
    array([[  0, 132],
           [  2, 216]])

    In [14]: np.c_[np.unique(s.lcqs[17612+2400+348:17612+2400+348+600,0],return_counts=True)]
    Out[14]: array([[-99, 600]])   ## ALL 600 MPMT with cat -99

    In [15]: np.c_[np.unique(s.lcqs[17612+2400+348+600:17612+2400+348+600+5,0],return_counts=True)]
    Out[15]:
    array([[0, 4],
           [2, 1]])


Q: Where is lcqs used ? A: qpmt.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    qpmt<F>::get_lpmtcat_from_lpmtid
    qpmt<F>::get_lpmtcat_from_lpmtidx
    qpmt<F>::get_qescale_from_lpmtidx


pmtCat
~~~~~~~~

pmtCat_v comes from pmtCat array serialized from PMTParamData::m_pmt_categories
whereas most everything else comes from PMTSimParamData

pmtCat is a serialization of m_pmt_categories unordered_map done by NPX::ArrayFromDiscoMapUnordered
which orders the (key,val) array in numerically ascending absolute pmt identifier order.

Ordering is::

    CD_LPMT        17612          OFFSET_CD_LPMT:OFFSET_CD_LPMT_END               0:17612
    CD_SPMT        25600          OFFSET_CD_SPMT:OFFSET_CD_SPMT_END           20000:45600
    WP_LPMT         2400          OFFSET_WP_PMT:OFFSET_WP_PMT_END             50000:52400
    WP_ATM_LPMT      348          OFFSET_WP_ATM_LPMT:OFFSET_WP_ATM_LPMT_END   52400:52748
    WP_ATM_MPMT      600          OFFSET_WP_ATM_MPMT:OFFSET_WP_ATM_MPMT_END   53000:53600
    WP_ATM_WAL         5          OFFSET_WP_WAL_PMT:OFFSET_WP_WAL_PMT_END     54000:54005

Note that MPMT is included even prior to MPMT being fully implemented.

Meanings of quantities used below

lpmtidx
   index in range 0:s_pmt::NUM_LPMTIDX

pmtid
   id obtained from s_pmt::pmtid_from_lpmtidx


**/

inline void SPMT::init_lcqs()
{

    // 0. assert on inputs

    assert( PMTSimParamData );
    assert( PMTParamData );
    assert( s_pmt::NUM_CD_LPMT == total.CD_LPMT );
    assert( s_pmt::NUM_WP   == total.WP );

    // 1. check consistency between start of pmtCat_v and lpmtCat_v
    for(int i=0 ; i < s_pmt::NUM_CD_LPMT ; i++)  // only to 17612
    {
        assert( pmtCat_v[2*i+1] == lpmtCat_v[i] ) ;
    }

    // 2. populate v_lcqs array excluding SPMT

    v_lcqs.resize(s_pmt::NUM_LPMTIDX);
    assert( pmtCat_ni == s_pmt::NUM_OLDCONTIGUOUSIDX ); // MPMT:600 included in pmtCat even prior to full impl

    for(int i=0 ; i < s_pmt::NUM_LPMTIDX ; i++ )
    {
        // outcome depends crucially on the implicit order of lpmtidx from s_pmt.h
        int lpmtidx = i ;
        int pmtid = s_pmt::pmtid_from_lpmtidx( lpmtidx );
        int lpmtidx2 = s_pmt::lpmtidx_from_pmtid( pmtid );
        assert( lpmtidx == lpmtidx2 );

        // *ix:oldcontiguousidx* follows absolute pmtid numerical order but removes the gaps
        // [making it a well founded ordering that applies to all PMTs]
        int oldcontiguousidx = s_pmt::oldcontiguousidx_from_pmtid( pmtid ); // offsets and num
        assert( oldcontiguousidx < s_pmt::NUM_OLDCONTIGUOUSIDX );
        assert( oldcontiguousidx < pmtCat_ni );


        int copyno = pmtCat_v[2*oldcontiguousidx+0] ;
        int cat    = pmtCat_v[2*oldcontiguousidx+1] ;
        bool copyno_pmtid_consistent = copyno == pmtid ;

        //const char* delim = "\n" ;
        const char* delim = "" ;
        if(!copyno_pmtid_consistent)
        std::cerr
           << "SPMT::init_lcqs" << delim
           << " i/lpmtidx "                   << std::setw(6) << lpmtidx << delim
           << " lpmtidx2 "                    << std::setw(6) << lpmtidx2 << delim
           << " oldcontiguousidx "            << std::setw(6) << oldcontiguousidx << delim
           << " s_pmt::NUM_OLDCONTIGUOUSIDX " << std::setw(6) << s_pmt::NUM_OLDCONTIGUOUSIDX << delim
           << " s_pmt::NUM_LPMTIDX "          << std::setw(6) << s_pmt::NUM_LPMTIDX << delim
           << " pmtid "                       << std::setw(6) << pmtid << delim
           << " copyno[pmtCat_v] "            << std::setw(6) << copyno << delim
           << " cat[pmtCat_v] "               << std::setw(6) << cat << delim
           << " copyno_pmtid_consistent "     << ( copyno_pmtid_consistent ? "YES" : "NO " ) << delim
           << "\n"
           ;

        assert( copyno_pmtid_consistent );

        float qesc = get_pmtid_qescale( pmtid );
        v_lcqs[lpmtidx] = { TranslateCat(cat), qesc } ;
    }
    lcqs = NPX::ArrayFromVec<int,LCQS>( v_lcqs ) ;

    if(level > 0) std::cout
       << "SPMT::init_lcqs" << std::endl
       << " NUM_CD_LPMT " << s_pmt::NUM_CD_LPMT << std::endl
       << " pmtCat " << ( pmtCat ? pmtCat->sstr() : "-" ) << std::endl
       << " lpmtCat " << ( lpmtCat ? lpmtCat->sstr() : "-" ) << std::endl
       << " qeScale " << ( qeScale ? qeScale->sstr() : "-" ) << std::endl
       << " lcqs " << ( lcqs ? lcqs->sstr() : "-" ) << std::endl
       ;

    assert( lcqs->shape[0] == s_pmt::NUM_LPMTIDX );

    assert( s_pmt::NUM_CD_LPMT == 17612 );
    assert( s_pmt::NUM_WP == 2400 );
    assert( s_pmt::NUM_WP_ATM_LPMT == 348 );
    assert( s_pmt::NUM_WP_WAL_PMT == 5 );

}






/**
SPMT::init_s_qescale
----------------------

pmtCat
    created by _PMTParamData uses "oldcontiguous" order : CD_LPMT, SPMT, WP

qeScale
    created by _PMTSimParamData::serialize::

        NP* qeScale = NPX::ArrayFromVec<double,double>(data.m_all_pmtID_qe_scale) ;

    ordering used by data.m_all_pmtID_qe_scale is SPMT at end (aka contiguousidx)

**/


inline void SPMT::init_s_qescale()
{
    int ni = s_qescale ? s_qescale->shape[0] : 0 ;
    int nj = s_qescale ? s_qescale->shape[1] : 0 ;
    assert( ni == s_pmt::NUM_SPMT );
    assert( nj == 1 );
    assert( s_qescale_v );

    for(int i=0 ; i < ni ; i++ )
    {
        int spmtidx = i ;
        int pmtid = s_pmt::pmtid_from_spmtidx( spmtidx );

        // pmtCat from _PMTParamData uses s_pmt.h "oldcontiguous" order : CD_LPMT, SPMT, WP
        int oldcontiguousidx = s_pmt::oldcontiguousidx_from_pmtid( pmtid );
        int copyno = pmtCat_v[2*oldcontiguousidx+0] ;
        int cat    = pmtCat_v[2*oldcontiguousidx+1] ;
        assert( copyno == pmtid );
        assert( cat == 2 );

        double qesc = get_pmtid_qescale( pmtid );

        s_qescale_v[spmtidx*nj + 0] = qesc  ;
    }
}












/**
SPMT::TranslateCat : 0,1,3 => 0,1,2
--------------------------------------

::

    In [10]: np.c_[np.unique( t.lpmtCat[:,0], return_counts=True )]
    Out[10]:
    array([[   0, 2720],
           [   1, 4997],
           [   3, 9895]])


    In [4]: t.lcqs[:,0]
    Out[4]: array([1, 1, 2, 1, 2, ..., 1, 1, 1, 1, 1], dtype=int32)

    In [5]: np.unique(t.lcqs[:,0], return_counts=True)
    Out[5]: (array([0, 1, 2], dtype=int32), array([2720, 4997, 9895]))


Including WP should make no difference::

    In [19]: np.unique(f.PMTParamData.pmtCat[:17612,1], return_counts=True)
    Out[19]: (array([0, 1, 3], dtype=int32), array([2738, 4955, 9919]))

    In [20]: np.unique(f.PMTParamData.pmtCat[17612+25600:17612+25600+2400,1],return_counts=True)
    Out[20]: (array([0, 3], dtype=int32), array([1836,  564]))


**/

inline int SPMT::TranslateCat(int lpmtcat) // static
{
    int lcat = -1 ;
    switch( lpmtcat )
    {
        case -1: lcat = -99  ; break ;   // kPMT_Unknown     see "jcv PMTCategory"
        case  0: lcat =  0   ; break ;   // kPMT_NNVT
        case  1: lcat =  1   ; break ;   // kPMT_Hamamatsu
        case  2: lcat =  -99 ; break ;   // kPMT_HZC
        case  3: lcat =  2   ; break ;   // kPMT_NNVT_HighQE
        case  4: lcat = -99  ; break ;   // ?MPMT:3?
        default: lcat = -99  ; break ;
    }
    //assert( lcat >= 0 );
    return lcat ;
}



inline std::string SPMT::descDetail() const
{
    std::stringstream ss ;
    ss << "SPMT::descDetail" << std::endl ;
    ss << ( jpmt ? jpmt->desc() : "NO_JPMT" ) << std::endl ;
    ss << "PMT_RINDEX.desc " << std::endl ;
    ss << ( PMT_RINDEX ? PMT_RINDEX->desc() : "NO_PMT_RINDEX" ) << std::endl ;
    ss << "PMTSimParamData.desc " << std::endl ;
    ss << ( PMTSimParamData ? PMTSimParamData->desc() : "NO_PMTSimParamData" ) << std::endl ;
    ss << "jpmt/PMTSimParamData/MPT " << std::endl << std::endl ;
    ss << ( MPT ? MPT->desc() : "NO_MPT" ) << std::endl ;
    std::string str = ss.str();
    return str ;
}

inline std::string SPMT::desc() const
{
    std::stringstream ss ;
    ss << "SPMT::desc" << std::endl ;
    ss << "jpmt.loaddir " << ( jpmt && jpmt->loaddir ? jpmt->loaddir : "NO_LOAD" ) << std::endl ;
    ss << "rindex " << ( rindex ? rindex->sstr() : "-" ) << std::endl ;
    ss << "thickness " << ( thickness ? thickness->sstr() : "-" ) << std::endl ;
    ss << "qeshape " << ( qeshape ? qeshape->sstr() : "-" ) << std::endl ;
    ss << "s_qeshape " << ( s_qeshape ? s_qeshape->sstr() : "-" ) << std::endl ;
    ss << "s_qescale " << ( s_qescale ? s_qescale->sstr() : "-" ) << std::endl ;
    ss << "cetheta " << ( cetheta ? cetheta->sstr() : "-" ) << std::endl ;
    ss << "cecosth " << ( cecosth ? cecosth->sstr() : "-" ) << std::endl ;
    ss << "lcqs " << ( lcqs ? lcqs->sstr() : "-" ) << std::endl ;

    std::string str = ss.str();
    return str ;
}

inline bool SPMT::is_complete() const
{
    return
       rindex != nullptr &&
       thickness != nullptr &&
       qeshape != nullptr &&
       s_qeshape != nullptr &&
       s_qescale != nullptr &&
       cetheta != nullptr &&
       cecosth != nullptr &&
       lcqs != nullptr
       ;
}

inline NPFold* SPMT::serialize() const   // formerly get_fold
{
    return is_complete() ? serialize_() : nullptr ;
}

inline NPFold* SPMT::serialize_() const   // formerly get_fold
{
    NPFold* fold = new NPFold ;

    if(jpmt) fold->add_subfold("jpmt", const_cast<NPFold*>(jpmt) ) ;

    if(rindex) fold->add("rindex", rindex) ;
    if(thickness) fold->add("thickness", thickness) ;
    if(qeshape) fold->add("qeshape", qeshape) ;
    if(s_qeshape) fold->add("s_qeshape", s_qeshape) ;
    if(s_qescale) fold->add("s_qescale", s_qescale) ;
    if(cetheta) fold->add("cetheta", cetheta) ;
    if(cecosth) fold->add("cecosth", cecosth) ;
    if(lcqs) fold->add("lcqs", lcqs) ;
    return fold ;
}


template<typename T>
inline T SPMT::GetFrac(int i, int ni )  // static
{
   return ni == 1 ? 0.f : T(i)/T(ni-1) ;
}

template<typename T>
inline T SPMT::GetValueInRange(int j, int nj, T X0, T X1 )  // static
{
    assert( j < nj );
    T one(1.);
    T fr = GetFrac<T>(j, nj);
    T x = X0*(one-fr) + X1*fr ;
    return x ;
}

inline float SPMT::get_energy(int j, int nj) const
{
    return GetValueInRange<float>(j, nj, EN0, EN1) ;
}
inline float SPMT::get_wavelength(int j, int nj) const
{
    return GetValueInRange<float>(j, nj, WL0, WL1) ;
}

inline float SPMT::get_minus_cos_theta_linear_angle(int k, int nk) const
{
    assert( k < nk );
    float theta_frac = GetFrac<float>(k, nk);
    float max_theta_pi = 1.f ;
    float theta = theta_frac*max_theta_pi*M_PI ;
    float mct = -cos(theta) ;
    return mct ;
}

inline float SPMT::get_minus_cos_theta_linear_cosine(int k, int nk) const
{
    assert( k < nk );
    float fr = GetFrac<float>(k, nk);
    float MCT0 = -1.f ;
    float MCT1 =  1.f ;
    float mct = MCT0*(1.f-fr) + MCT1*fr ;
    return mct ;
}

inline float SPMT::get_rindex(int cat, int layr, int prop, float energy_eV) const
{
    assert( cat == 0 || cat == 1 || cat == 2 );
    assert( layr == 0 || layr == 1 || layr == 2 || layr == 3 );
    assert( prop == 0 || prop == 1 );
    return rindex->combined_interp_5( cat, layr, prop,  energy_eV ) ;
}

inline NP* SPMT::get_rindex() const
{
    std::cout << "SPMT::get_rindex " << std::endl ;

    int ni = 3 ;  // pmtcat [0,1,2]
    int nj = 4 ;  // layers [0,1,2,3]
    int nk = 2 ;  // props [0,1] (RINDEX,KINDEX)
    int nl = N_EN ; // energies [0..N_EN-1]
    int nn = 2 ;   // payload [energy_eV,rindex_value]

    NP* a = NP::Make<float>(ni,nj,nk,nl,nn) ;
    float* aa = a->values<float>();

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    for(int k=0 ; k < nk ; k++)
    for(int l=0 ; l < nl ; l++)
    {
        float en = get_energy(l, nl );
        float ri = get_rindex(i, j, k, en) ;
        int idx = i*nj*nk*nl*nn+j*nk*nl*nn+k*nl*nn+l*nn ;
        aa[idx+0] = en ;
        aa[idx+1] = ri ;
    }
    return a ;
}


inline float SPMT::get_qeshape(int cat, float energy_eV) const
{
    assert( cat == 0 || cat == 1 || cat == 2 );
    return qeshape->combined_interp_3( cat, energy_eV ) ;
}

inline float SPMT::get_s_qeshape(int cat, float energy_eV) const
{
    assert( cat == 0  );
    return s_qeshape->combined_interp_3( cat, energy_eV ) ;
}





inline NP* SPMT::get_qeshape() const
{
    std::cout << "SPMT::get_qeshape " << std::endl ;

    int ni = 3 ;   // pmtcat [0,1,2]
    int nj = N_EN ; // energies [0..N_EN-1]
    int nk = 2 ;   // payload [energy_eV,qeshape_value]

    NP* a = NP::Make<float>(ni,nj,nk) ;
    float* aa = a->values<float>();

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    {
        float en = get_energy(j, nj );
        float qe = get_qeshape(i, en) ;
        int idx = i*nj*nk+j*nk ;
        aa[idx+0] = en ;
        aa[idx+1] = qe ;
    }
    return a ;
}

inline NP* SPMT::get_s_qeshape() const
{
    std::cout << "SPMT::get_s_qeshape " << std::endl ;

    int ni = 1 ;   // only cat 0 for small PMT
    int nj = N_EN ;
    int nk = 2 ;   // payload [energy_eV,qeshape_value]

    NP* a = NP::Make<float>(ni,nj,nk) ;
    float* aa = a->values<float>();

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    {
        float en = get_energy(j, nj );
        float qe = get_s_qeshape(i, en) ;
        int idx = i*nj*nk+j*nk ;
        aa[idx+0] = en ;
        aa[idx+1] = qe ;
    }
    return a ;
}





float SPMT::get_thickness_nm(int cat, int lay) const
{
    assert( cat == 0 || cat == 1 || cat == 2 );
    assert( lay == 0 || lay == 1 || lay == 2 || lay == 3 );
    return tt[cat*NUM_LAYER+lay] ;
}

NP* SPMT::get_thickness_nm() const
{
    std::cout << "SPMT::get_thickness_nm " << std::endl ;
    int ni = NUM_PMTCAT ;
    int nj = NUM_LAYER ;
    int nk = 1 ;
    NP* a = NP::Make<float>(ni, nj, nk );
    float* aa = a->values<float>();
    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    {
        int idx = i*NUM_LAYER + j ;
        aa[idx] = get_thickness_nm(i, j);
    }
    return a ;
}

/**
SPMT::get_pmtid_stackspec
---------------------------

Expt with using "spare" fourth column .w spec slots to hold onto calculation
intermediates to provide a working context, avoiding API
contortions or repeated lookups.

   +----+-------------+----------+------------+-------------+-------------+
   |    |    x        |   y      |  z         |  w          |  Notes      |
   +====+=============+==========+============+=============+=============+
   | q0 | rindex      |  0.f     | 0.f        | i:pmtcat    | Pyrex       |
   +----+-------------+----------+------------+-------------+-------------+
   | q1 | rindex      | kindex   | thickness  | f:qe_scale  | ARC         |
   +----+-------------+----------+------------+-------------+-------------+
   | q2 | rindex      | kindex   | thickness  | f:qe        | PHC         |
   +----+-------------+----------+------------+-------------+-------------+
   | q3 | rindex 1.f  |  0.f     | 0.f        | f:_qe       | Vacuum      |
   +----+-------------+----------+------------+-------------+-------------+


1. lookup pmtcat, qe_scale for the pmtid
2. using pmtcat and energy_eV do rindex,kindex interpolation and thickness lookups

Note that the rindex, kindex and thickness only depend on the pmtcat.
However the fourth column qe related values do depend on pmtid with
differences coming in via the qe_scale.

**/
void SPMT::get_lpmtid_stackspec( quad4& spec, int lpmtid, float energy_eV) const
{
    spec.zero();

    int& cat = spec.q0.i.w ;
    float& qe_scale = spec.q1.f.w ;
    float& qe = spec.q2.f.w ;
    float& _qe = spec.q3.f.w ;
    // above are refs to locations currently all holding zero

    int lpmtidx = s_pmt::lpmtidx_from_pmtid( lpmtid );
    get_lcqs_from_lpmtidx(cat, qe_scale, lpmtidx);

    assert( cat > -1 && cat < NUM_PMTCAT );
    assert( qe_scale > 0.f );

    qe = get_pmtcat_qe(cat, energy_eV ) ; // qeshape interpolation
    _qe = qe*qe_scale ;

    bool expected_range = _qe > 0.f && _qe < 1.f ;

    if(!expected_range) std::cout
        << "SPMT::get_pmtid_stackspec"
        << " expected_range " << ( expected_range ? "YES" : "NO " )
        << " lpmtid " << lpmtid
        << " energy_eV " << energy_eV
        << " qe " << qe
        << " qe_scale " << qe_scale
        << " _qe " << _qe
        << std::endl
        ;

    assert( expected_range );


    spec.q0.f.x = get_rindex( cat, L0, RINDEX, energy_eV );

    spec.q1.f.x = get_rindex(       cat, L1, RINDEX, energy_eV );
    spec.q1.f.y = get_rindex(       cat, L1, KINDEX, energy_eV );
    spec.q1.f.z = get_thickness_nm( cat, L1 );

    spec.q2.f.x = get_rindex(       cat, L2, RINDEX, energy_eV );
    spec.q2.f.y = get_rindex(       cat, L2, KINDEX, energy_eV );
    spec.q2.f.z = get_thickness_nm( cat, L2 );

    spec.q3.f.x = 1.f ; // Vacuum
}


#ifdef WITH_CUSTOM4
/**
SPMT::get_ARTE : TMM MultiLayerStack calculation using complex rindex, thickness, ...
---------------------------------------------------------------------------------------

Output:ARTE float4
    theAbsorption,theReflectivity,theTransmittance,theEfficiency

pmtid
    integer in range 0->N_LPMT-1 eg N_LPMT 17612

    * used to lookup pmtcat 0,1,2 and qe_scale

energy_eV
    float in range 1.55 to 15.5

    * used for rindex, kindex, qeshape interpolated property lookups

minus_cos_theta
    obtain from dot(mom,nrm) OR OldPolarization*theRecoveredNormal

    * expresses the angle of incidence of the photon onto the surface
    * -1.f:forward normal incidence
    *  0.f:skimming incidence
    * +1.f:backward normal incidence

dot_pol_cross_mom_nrm

    * dot(pol,cross(mom,nrm)) where pol, mom and nrm are all normalized float3 vectors
    * OldPolarization*OldMomentum.cross(theRecoveredNormal) in G4ThreeVector form
    * expresses degree of S vs P polarization of the incident photon
    * pol,mom,nrm all expected to be normalized vectors
    * cross(mom,nrm) vector is transverse to plane of incidence which contains mom and nrm by definition
    * cross(mom,nrm) has magnitude sine(angle_between_mom_and_nrm)
    * cross(mom,nrm) becomes zero at normal incidence, but S/P have no meaning in that case anyhow
    * dot(pol,cross(mom,nrm)) value is cosine(angle_between_pol_and_transverse_vector)*sine(angle_between_mom_and_nrm)
    * NB when testing the value of dot_pol_cross_mom_nrm needs to be consistent with minus_cos_theta

**NB : minus_cos_theta AND dot_pol_cross_mom_nrm ARGS ARE RELATED**

dot_pol_cross_mom_nrm includes cross(mom,nrm) and minus_cos_theta is dot(mom,nrm)
so one incorporates the cross product and the other is the dot product
of the same two vectors. Hence care is needed to prepare correctly
related arguments when test scanning the API.

S_pol incorporation
~~~~~~~~~~~~~~~~~~~~~~

* stack.art.A,R,T now incorporatws S_pol frac obtained from (minus_cos_theta, dot_pol_cross_mom_nrm )


**/

inline void SPMT::get_ARTE(
    SPMTData& pd,
    int   lpmtid,
    float wavelength_nm,
    float minus_cos_theta,
    float dot_pol_cross_mom_nrm ) const
{
    const float energy_eV = hc_eVnm/wavelength_nm ;
    get_lpmtid_stackspec(pd.spec, lpmtid, energy_eV);

    const float* ss = pd.spec.cdata() ;
    const float& _qe = ss[15] ;

    pd.args.x = lpmtid ;
    pd.args.y = energy_eV ;
    pd.args.z = minus_cos_theta ;
    pd.args.w = dot_pol_cross_mom_nrm ;

    if( minus_cos_theta < 0.f ) // only ingoing photons
    {
        pd.stack.calc(wavelength_nm, -1.f, 0.f, ss, 16u );
        pd.ARTE.w = _qe/pd.stack.art.A ;  // aka theEfficiency and escape_fac, no mct dep

        pd.extra.x = 1.f - (pd.stack.art.T_av + pd.stack.art.R_av ) ;  // old An
        pd.extra.y = pd.stack.art.A_av ;
        pd.extra.z = pd.stack.art.A   ;
        pd.extra.w = pd.stack.art.A_s ;
    }
    else
    {
        pd.ARTE.w = 0.f ;
    }

    pd.stack.calc(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );

    const float& A = pd.stack.art.A ;
    const float& R = pd.stack.art.R ;
    const float& T = pd.stack.art.T ;

    pd.ARTE.x = A ;         // aka theAbsorption
    pd.ARTE.y = R/(1.f-A) ; // aka theReflectivity
    pd.ARTE.z = T/(1.f-A) ; // aka theTransmittance
}


/**
stack::calc notes
-------------------

0. backwards _qe is set to zero (+ve minus_cos_theta, ie mom with nrm)
1. DONE : compare An with stackNormal.art.A_av
2. DONE : remove _av as at normal incidence S/P are meaningless and give same values anyhow,
3. DONE : removed stackNormal.calc for minus_cos_theta > 0.f
4. DONE : even better get rid of stackNormal by reusing the one stack instance

**/


/**
SPMT::annotate
----------------

**/

inline void SPMT::annotate( NP* art ) const
{
    std::vector<std::pair<std::string, std::string>> kvs =
    {
        { "title", "SPMT.title" },
        { "brief", "SPMT.brief" },
        { "name",  "SPMT.name"  },
        { "label", "SPMT.label" },
        { "ExecutableName", ExecutableName }
    };

    art->set_meta_kv<std::string>(kvs);
}



inline const NP* SPMT::Make_LPMTID_LIST()  // static
{
    const char* lpmtid_list = ssys::getenvvar("LPMTID_LIST", LPMTID_LIST) ;
    const NP* lpmtid = NPX::FromString<int>(lpmtid_list,',');
    return lpmtid ;
}





/**
SPMT::make_c4scan
-----------------

Scan over (lpmtid, wl, mct, spol )

Potentially could scan over any of (ni,nj,nk,nl)
so should add arrays of the ranges which will have
only one value when not scanned over.

**/


inline NPFold* SPMT::make_c4scan() const
{
    if(level > 0) std::cout << "[SPMT::make_c4scan level " << level << std::endl;
    NPFold* fold = new NPFold ;

    const NP* lpmtid_domain = Make_LPMTID_LIST() ;
    const NP* lpmtcat_domain = get_lpmtcat_from_lpmtid( lpmtid_domain );
    const NP* mct_domain = NP::MinusCosThetaLinearAngle<float>( N_MCT );
    const NP* st_domain = NP::SqrtOneMinusSquare(mct_domain) ;

    if(level > 1) std::cout << " N_MCT " << N_MCT << std::endl ;
    if(level > 1) std::cout << " mct_domain.desc " << mct_domain->desc() << std::endl ;

    fold->add("lpmtid_domain", lpmtid_domain);
    fold->add("lpmtcat_domain", lpmtcat_domain);
    fold->add("mct_domain", mct_domain );
    fold->add("st_domain", st_domain );

    SPMTData pd ;

    int ni = lpmtid_domain->shape[0] ;
    int nj = N_WL ;
    int nk = N_MCT ;
    int nl = N_SPOL ;



    NP* args  = NP::Make<float>(ni, nj, nk, nl, 4 );
    NP* ARTE  = NP::Make<float>(ni, nj, nk, nl, 4 );
    NP* extra = NP::Make<float>(ni, nj, nk, nl, 4 );
    NP* spec  = NP::Make<float>(ni, nj, nk, nl, 4, 4 );

    // Make_ allows arbitrary dimensions
    NP* stack = NP::Make_<float>(ni, nj, nk, nl, 44, 4 );
    NP* ll    = NP::Make_<float>(ni, nj, nk, nl, 4, 4, 4, 2) ;  // (32,4)   4*4*2 = 32
    NP* comp  = NP::Make_<float>(ni, nj, nk, nl, 1, 4, 4, 2) ;  // ( 8,4)   4*2 =  8
    NP* art   = NP::Make_<float>(ni, nj, nk, nl, 4, 4) ;        // ( 4,4)
                                                                // --------
                                                                //  (44,4)

    // stack is composed of (ll,comp,art) but its not easy to use because
    // those have different shapes, hence also split them into separate
    // (ll,comp,art) arrays for easier querying

    annotate(art);

    assert( sizeof(pd.stack)/sizeof(float) == 44*4 );

    fold->add("args", args);
    fold->add("ARTE", ARTE);
    fold->add("extra",extra );
    fold->add("spec", spec);

    fold->add("stack", stack);
    fold->add("ll", ll);
    fold->add("comp", comp);
    fold->add("art", art);

    if(level > 1) std::cout << "SPMT::make_c4scan args.sstr " << args->sstr() << std::endl ;

    const int* lpmtid_domain_v = lpmtid_domain->cvalues<int>();
    const float* mct_domain_v = mct_domain->cvalues<float>();
    const float* st_domain_v = st_domain->cvalues<float>();

    float* args_v = args->values<float>();
    float* ARTE_v = ARTE->values<float>();
    float* extra_v = extra->values<float>() ;
    float* spec_v  = spec->values<float>() ;

    float* stack_v = stack->values<float>() ;
    float* ll_v    = ll->values<float>() ;
    float* comp_v  = comp->values<float>() ;
    float* art_v   = art->values<float>() ;

    for(int i=0 ; i < ni ; i++)
    {
        int lpmtid = lpmtid_domain_v[i] ;
        if( i % 100 == 0 && level > 1) std::cout << "SPMT::make_c4scan lpmtid " << lpmtid << std::endl ;
        for(int j=0 ; j < nj ; j++)
        {
           float wavelength_nm = get_wavelength(j, nj );
           for(int k=0 ; k < nk ; k++)
           {
              float minus_cos_theta = mct_domain_v[k] ;
              float sin_theta = st_domain_v[k] ;
              {
                  float minus_cos_theta_0 = get_minus_cos_theta_linear_angle(k, nk );
                  float minus_cos_theta_diff = std::abs( minus_cos_theta - minus_cos_theta_0 );
                  bool minus_cos_theta_diff_expect = minus_cos_theta_diff < 1e-6 ;
                  assert( minus_cos_theta_diff_expect );
                  if(!minus_cos_theta_diff_expect) std::cerr << "SPMT::make_c4scan minus_cos_theta_diff_expect " << std::endl ;
                  float sin_theta_0 = sqrt( 1.f - minus_cos_theta*minus_cos_theta );
                  float sin_theta_diff = std::abs(sin_theta - sin_theta_0)  ;
                  bool sin_theta_expect = sin_theta_diff < 1e-6 ;
                  if(!sin_theta_expect) std::cout
                      << "SPMT::make_c4scan"
                      << " k " << k
                      << " minus_cos_theta " << std::setw(10) << std::fixed << std::setprecision(5) << minus_cos_theta
                      << " sin_theta " << std::setw(10) << std::fixed << std::setprecision(5) << sin_theta
                      << " sin_theta_0 " << std::setw(10) << std::fixed << std::setprecision(5) << sin_theta_0
                      << " sin_theta_diff " << std::setw(10) << std::fixed << std::setprecision(5) << sin_theta_diff
                      << std::endl
                      ;
                  assert( sin_theta_expect );
              }

              for(int l=0 ; l < nl ; l++)
              {
                  float s_pol_frac = GetFrac<float>(l, nl) ;
                  float dot_pol_cross_mom_nrm = sin_theta*s_pol_frac ;

                  get_ARTE(pd, lpmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm );

                  int idx = i*nj*nk*nl*4 + j*nk*nl*4 + k*nl*4 + l*4 ;

                  int args_idx = idx ;
                  int ARTE_idx = idx ;
                  int extra_idx = idx ;
                  int spec_idx = idx*4 ;

                  int stack_idx = idx*44 ;
                  int ll_idx    = idx*4*4*2 ; // 32
                  int comp_idx  = idx*1*4*2 ; //  8
                  int art_idx   = idx*4 ;     //  4

                  memcpy( args_v + args_idx,   &pd.args.x, sizeof(float)*4 );
                  memcpy( ARTE_v + ARTE_idx,   &pd.ARTE.x, sizeof(float)*4 );
                  memcpy( extra_v + extra_idx, &pd.extra.x,   sizeof(float)*4 );
                  memcpy( spec_v + spec_idx,   pd.spec.cdata(), sizeof(float)*4*4 );

                  memcpy( stack_v + stack_idx,  pd.stack.cdata(),        sizeof(float)*44*4 );
                  memcpy( ll_v    + ll_idx,     pd.stack.ll[0].cdata(),  sizeof(float)*32*4 );
                  memcpy( comp_v  + comp_idx,   pd.stack.comp.cdata(),   sizeof(float)*8*4 );
                  memcpy( art_v   + art_idx,    pd.stack.art.cdata(),    sizeof(float)*4*4 );

                  if(level > 0) std::cout
                      << "SPMT::make_c4scan"
                      << " i " << std::setw(5) << i
                      << " j " << std::setw(5) << j
                      << " k " << std::setw(5) << k
                      << " l " << std::setw(5) << l
                      << " args_idx " << std::setw(10) << args_idx
                      << std::endl
                      ;

              }
           }
        }
    }
    if(level > 0) std::cout << "]SPMT::make_c4scan " << std::endl;
    return fold ;
}
#endif





void SPMT::get_stackspec( quad4& spec, int cat, float energy_eV) const
{
    spec.zero();
    spec.q0.f.x = get_rindex( cat, L0, RINDEX, energy_eV );

    spec.q1.f.x = get_rindex(       cat, L1, RINDEX, energy_eV );
    spec.q1.f.y = get_rindex(       cat, L1, KINDEX, energy_eV );
    spec.q1.f.z = get_thickness_nm( cat, L1 );

    spec.q2.f.x = get_rindex(       cat, L2, RINDEX, energy_eV );
    spec.q2.f.y = get_rindex(       cat, L2, KINDEX, energy_eV );
    spec.q2.f.z = get_thickness_nm( cat, L2 );

    spec.q3.f.x = get_rindex( cat, L3, RINDEX, energy_eV );
    //spec.q3.f.x = 1.f ; // Vacuum, so could just set to 1.f
}

NP* SPMT::get_stackspec() const
{
    int ni = NUM_PMTCAT ;
    int nj = N_EN ;
    int nk = 4 ;
    int nl = 4 ;

    NP* a = NP::Make<float>(ni, nj, nk, nl );
    std::cout << "[ SPMT::get_stackspec " << a->sstr() << std::endl ;

    float* aa = a->values<float>();

    quad4 spec ;
    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    {
       float en = get_energy(j, nj );
       get_stackspec(spec, i, en );
       int idx = i*nj*nk*nl + j*nk*nl ;
       memcpy( aa+idx, spec.cdata(), nk*nl*sizeof(float) );
    }
    std::cout << "] SPMT::get_stackspec " << a->sstr() << std::endl ;
    return a ;
}

/**
SPMT::get_lpmtcat_from_lpmtidx
--------------------------------

For lpmtidx(0->17612+2400-1 ) returns 0, 1 or 2 corresponding to NNVT, HAMA, NNVT_HiQE

**/

inline int SPMT::get_lpmtcat_from_lpmtidx(int lpmtidx) const
{
    assert( lpmtidx >= 0 && lpmtidx < s_pmt::NUM_LPMTIDX );
    const int* lcqs_i = lcqs->cvalues<int>() ;
    return lcqs_i[lpmtidx*2+0] ;
}

inline NP* SPMT::get_lpmtcat_from_lpmtidx() const
{
    std::cout << "SPMT::get_lpmtcat_from_lpmtidx " << std::endl ;
    NP* a = NP::Make<int>( s_pmt::NUM_LPMTIDX ) ;
    int* aa = a->values<int>();
    for(int i=0 ; i < a->shape[0] ; i++) aa[i] = get_lpmtcat_from_lpmtidx(i) ;
    return a ;
}






inline int SPMT::get_lpmtcat_from_lpmtid(int lpmtid) const
{
    int lpmtidx = s_pmt::lpmtidx_from_pmtid(lpmtid);
    return get_lpmtcat_from_lpmtidx(lpmtidx);
}

inline int SPMT::get_lpmtcat_from_lpmtid( int* lpmtcat_ , const int* lpmtid_ , int num ) const
{
    for(int i=0 ; i < num ; i++)
    {
        int lpmtid = lpmtid_[i] ;
        int lpmtidx = s_pmt::lpmtidx_from_pmtid(lpmtid);
        int lpmtcat = get_lpmtcat_from_lpmtidx(lpmtidx) ;
        lpmtcat_[i] = lpmtcat ;
    }
    return num ;
}


inline NP* SPMT::get_lpmtcat_from_lpmtid(const NP* lpmtid ) const
{
    assert( lpmtid->shape.size() == 1 );
    int num = lpmtid->shape[0] ;
    NP* lpmtcat = NP::Make<int>(num);
    int num2 = get_lpmtcat_from_lpmtid( lpmtcat->values<int>(), lpmtid->cvalues<int>(), num ) ;

    bool num_expect = num2 == num ;
    assert( num_expect );
    if(!num_expect) std::raise(SIGINT);

    return lpmtcat ;
}


inline float SPMT::get_qescale_from_lpmtid(int lpmtid) const
{
    int lpmtidx = s_pmt::lpmtidx_from_pmtid(lpmtid);
    return get_qescale_from_lpmtidx(lpmtidx);
}
inline int SPMT::get_copyno_from_lpmtid(int lpmtid) const
{
    int lpmtidx = s_pmt::lpmtidx_from_pmtid(lpmtid);
    int copyno = get_copyno_from_lpmtidx(lpmtidx);
    assert( copyno == lpmtid );
    return copyno ;
}




inline float SPMT::get_qescale_from_lpmtidx(int lpmtidx) const
{
    assert( lpmtidx >= 0 && lpmtidx < s_pmt::NUM_LPMTIDX );
    const float* lcqs_f = lcqs->cvalues<float>() ;
    return lcqs_f[lpmtidx*2+1] ;
}
inline int SPMT::get_copyno_from_lpmtidx(int lpmtidx) const
{
    assert( lpmtidx >= 0 && lpmtidx < s_pmt::NUM_LPMTIDX );
    const int* lcqs_i = lcqs->cvalues<int>() ;
    return lcqs_i[lpmtidx*2+0] ;
}




inline NP* SPMT::get_qescale_from_lpmtidx() const
{
    std::cout << "SPMT::get_qescale_from_lpmtidx " << std::endl ;
    NP* a = NP::Make<float>( s_pmt::NUM_LPMTIDX ) ;
    float* aa = a->values<float>();
    for(int i=0 ; i < a->shape[0] ; i++) aa[i] = get_qescale_from_lpmtidx(i) ;
    return a ;
}
inline NP* SPMT::get_copyno_from_lpmtidx() const
{
    std::cout << "SPMT::get_copyno_from_lpmtidx " << std::endl ;
    NP* a = NP::Make<int>( s_pmt::NUM_LPMTIDX ) ;
    int* aa = a->values<int>();
    for(int i=0 ; i < a->shape[0] ; i++) aa[i] = get_copyno_from_lpmtidx(i) ;
    return a ;
}





/**
SPMT::get_lcqs
----------------

Accesses the lcqs array returning:

* lc(int): local (0,1,2) lpmt category
* qs(float):qescale

::

    In [5]: np.c_[np.unique( s.lcqs[:,0], return_counts=True )]
    Out[5]:
    array([[   0, 2720],
           [   1, 4997],
           [   2, 9895]])




**/
inline void SPMT::get_lcqs_from_lpmtid( int& lc, float& qs, int lpmtid) const
{
    int lpmtidx = s_pmt::lpmtidx_from_pmtid(lpmtid);
    return get_lcqs_from_lpmtidx(lc, qs, lpmtidx);
}
inline void SPMT::get_lcqs_from_lpmtidx(int& lc, float& qs, int lpmtidx) const
{
    assert( lpmtidx >= 0 && lpmtidx < s_pmt::NUM_LPMTIDX );
    const int*   lcqs_i = lcqs->cvalues<int>() ;
    const float* lcqs_f = lcqs->cvalues<float>() ;
    lc = lcqs_i[lpmtidx*2+0] ;
    qs = lcqs_f[lpmtidx*2+1] ;
}
/**
Q: Does not this just duplicate lcqs ?
A: YES, that is the point : to check that it does.
**/
inline NP* SPMT::get_lcqs_from_lpmtidx() const
{
    std::cout << "SPMT::get_lcqs_from_lpmtidx " << std::endl ;
    int ni = s_pmt::NUM_LPMTIDX;
    int nj = 2 ;
    NP* a = NP::Make<int>(ni, nj) ;
    int* ii   = a->values<int>() ;
    float* ff = a->values<float>() ;
    for(int i=0 ; i < ni ; i++) get_lcqs_from_lpmtidx( ii[i*nj+0], ff[i*nj+1], i );
    return a ;
}


inline float SPMT::get_s_qescale_from_spmtid( int pmtid ) const
{
    assert( s_pmt::is_spmtid( pmtid ));
    int spmtidx = s_pmt::spmtidx_from_pmtid(pmtid);
    assert( s_pmt::is_spmtidx( spmtidx ));
    float s_qs = s_qescale_v[spmtidx] ;
    return s_qs ;
}

inline NP* SPMT::get_s_qescale_from_spmtid() const
{
    int ni = s_pmt::NUM_SPMT;
    int nj = 1 ;
    NP* a = NP::Make<float>(ni, nj) ;
    float* aa = a->values<float>() ;
    for(int i=0 ; i < ni ; i++)
    {
        int spmtidx = i ;
        int spmtid = s_pmt::pmtid_from_spmtidx(spmtidx);
        aa[nj*i+0] = get_s_qescale_from_spmtid( spmtid );
    }
    return a ;
}




inline float SPMT::get_pmtcat_qe(int cat, float energy_eV) const
{
    assert( cat == 0 || cat == 1 || cat == 2 );
    return qeshape->combined_interp_3( cat, energy_eV ) ;
}
inline NP* SPMT::get_pmtcat_qe() const
{
    std::cout << "SPMT::get_pmtcat_qe " << std::endl ;
    int ni = 3 ;
    int nj = N_EN ;
    int nk = 2 ;

    NP* a = NP::Make<float>(ni, nj, nk) ;
    float* aa = a->values<float>();

    for(int i=0 ; i < ni ; i++)
    {
        for(int j=0 ; j < nj ; j++)
        {
            float en = get_energy(j, nj );
            aa[i*nj*nk+j*nk+0] = en ;
            aa[i*nj*nk+j*nk+1] = get_pmtcat_qe(i, en) ;
        }
    }
    return a ;
}



/**
SPMT::get_lpmtidx_qe
---------------------

::

    q = t.test.get_pmtid_qe

    In [21]: q.shape
    Out[21]: (17612, 266, 2)

    In [22]: np.argmax(q[:,:,0], axis=1)
    Out[22]: array([265, 265, 265, 265, 265, ..., 265, 265, 265, 265, 265])

    In [23]: np.argmax(q[:,:,1], axis=1)
    Out[23]: array([163, 163, 163, 163, 163, ..., 163, 163, 163, 163, 163])

* surprised that max qe at same energy for all 17612 pmtid ?

**/

inline float SPMT::get_lpmtidx_qe(int lpmtidx, float energy_eV) const
{
    int cat(-1) ;
    float qe_scale(-1.f) ;

    get_lcqs_from_lpmtidx(cat, qe_scale, lpmtidx);

    assert( cat == 0 || cat == 1 || cat == 2 );
    assert( qe_scale > 0.f );

    float qe = get_pmtcat_qe(cat, energy_eV);
    qe *= qe_scale ;

    return qe ;
}




inline NP* SPMT::get_lpmtidx_qe() const
{
    std::cout << "SPMT::get_lpmtidx_qe " << std::endl ;
    int ni = N_LPMT  ;
    int nj = N_EN ;
    int nk = 2 ;

    NP* a = NP::Make<float>(ni, nj, nk) ;
    float* aa = a->values<float>();

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    {
        float en = get_energy(j, nj );
        int idx = i*nj*nk+j*nk ;
        aa[idx+0] = en ;
        aa[idx+1] = get_lpmtidx_qe(i, en) ;
    }
    return a ;
}


/**
SPMT::make_testfold
---------------------

Invoked from main of sysrap/tests/SPMT_test.cc

**/


inline NPFold* SPMT::make_testfold() const
{
    std::cout << "[ SPMT::make_testfold " << std::endl ;

    NPFold* f = new NPFold ;

    f->add("get_lpmtcat_from_lpmtidx", get_lpmtcat_from_lpmtidx() );
    f->add("get_qescale_from_lpmtidx", get_qescale_from_lpmtidx() );
    f->add("get_lcqs_from_lpmtidx", get_lcqs_from_lpmtidx() );
    f->add("get_pmtcat_qe", get_pmtcat_qe() );
    f->add("get_lpmtidx_qe", get_lpmtidx_qe() );

    f->add("get_rindex", get_rindex() );
    f->add("get_qeshape", get_qeshape() );
    f->add("get_s_qeshape", get_s_qeshape() );
    f->add("get_s_qescale_from_spmtid", get_s_qescale_from_spmtid() );
    f->add("get_thickness_nm", get_thickness_nm() );
    f->add("get_stackspec", get_stackspec() );

#ifdef WITH_CUSTOM4
    f->add_subfold("c4scan", make_c4scan() );
#endif

    std::cout << "] SPMT::make_testfold " << std::endl ;
    return f ;
}
