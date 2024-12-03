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
#include "scuda.h"
#include "squad.h"
#include "ssys.h"
#include "sproc.h"

#ifdef WITH_CUSTOM4
#include "C4MultiLayrStack.h"
#endif


struct SPMT
{
    static constexpr const float hc_eVnm = 1239.84198433200208455673  ; 

    enum { L0, L1, L2, L3 } ; 
    enum { RINDEX, KINDEX } ; 

    struct LCQS { int lc ; float qs ; } ; 

    /*
    static constexpr const float EN0 = 1.55f ; 
    static constexpr const float EN1 = 4.20f ;  // 15.5 
    static constexpr const int   N_EN = 420 - 155 + 1 ; 
    */    

    static constexpr const float EN0 = 2.81f ; 
    static constexpr const float EN1 = 2.81f ; 
    static constexpr const int   N_EN = 1 ; 

    static constexpr const float WL0 = 440.f ; 
    static constexpr const float WL1 = 440.f ; 
    static constexpr const int   N_WL = 1 ; 



    /*
    static constexpr const float EN0 = 2.55f ; 
    static constexpr const float EN1 = 3.55f ; 
    static constexpr const int   N_EN = 2 ; 
    */

    /*
    static constexpr const float EN0 = 1.55f ; 
    static constexpr const float EN1 = 15.5f ; 
    static constexpr const int   N_EN = 1550 - 155 + 1 ; 
    */

    static constexpr const bool VERBOSE = false ; 
    static constexpr const char* PATH = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/extra/jpmt" ; 

    static constexpr int NUM_PMTCAT = 3 ; // (NNVT, HAMA, NNVT_HiQE)
    static constexpr int NUM_LAYER = 4 ;  // (Pyrex, ARC, PHC, Vacuum) 
    static constexpr int NUM_PROP = 2 ;   // (RINDEX,KINDEX) real and imaginary parts of the index

    static constexpr const int NUM_LPMT = 17612 ; 

    // below three can be changed via envvars
    static const int N_LPMT ;   // N_LPMT must be less than or equal to NUM_LPMT
    static const int N_MCT ;     
    static const int N_SPOL ; 
 
    static constexpr const char* QEshape_PMTCAT_NAMES = "QEshape_NNVT.npy,QEshape_R12860.npy,QEshape_NNVT_HiQE.npy" ; 
    // follows PMTCategory kPMT enum order but using QEshape array naming convention
    // because there is no consistently used naming convention have to do these dirty things 

    static const NPFold* Serialize(const char* path=nullptr);   
    static SPMT* Load(const char* path=nullptr); 
    SPMT(const NPFold* jpmt); 

    void init(); 
    void init_rindex_thickness(); 
    void init_qeshape(); 
    void init_lcqs(); 
    static int TranslateCat(int lpmtcat); 

    std::string descDetail() const ; 
    std::string desc() const ; 

    bool is_complete() const ; 
    NPFold* serialize() const ;  
    NPFold* serialize_() const ; 

    float get_frac(int i, int ni) const ; 

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
    NP* get_qeshape() const ; 

    float get_thickness_nm(int cat, int layr) const ; 
    NP* get_thickness_nm() const ; 


    void get_lpmtid_stackspec( quad4& spec, int lpmtid, float energy_eV) const ;  // EXPT

    static constexpr const char* LPMTID_LIST = "0,10,55,98,100,137,1000,10000,17611" ; 
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

    void get_ARTE(SPMTData& pd, int pmtid, float wavelength_nm, float minus_cos_theta, float dot_pol_cross_mom_nrm ) const ; 


    NPFold* make_sscan() const ; 
#endif

    void get_stackspec( quad4& spec, int cat, float energy_eV) const ; 
    NP*  get_stackspec() const ; 

    int  get_lpmtcat( int lpmtid ) const ; 
    int  get_lpmtcat( int* lpmtcat, const int* lpmtid , int num ) const ; 
    NP*  get_lpmtcat() const ; 
    NP*  get_lpmtcat(const NP* lpmtid) const ; 


    float get_qescale(int pmtid) const ; 
    NP*   get_qescale() const ; 

    void  get_lcqs( int& lc, float& qs, int pmtid ) const ; 
    NP*   get_lcqs() const ; 

    float get_pmtcat_qe(int cat, float energy_eV) const ; 
    NP*   get_pmtcat_qe() const ; 

    float get_pmtid_qe(int pmtid, float energy_eV) const ; 
    NP*   get_pmtid_qe() const ; 

    NPFold* make_testfold() const ; 



    const char* ExecutableName ;  

    const NPFold* jpmt ; 
    const NPFold* PMT_RINDEX ;       // PyrexRINDEX and VacuumRINDEX 
    const NPFold* PMTSimParamData ; 
    const NPFold* MPT ;              // ARC_RINDEX, ARC_KINDEX, PHC_RINDEX, PHC_KINDEX
    const NPFold* CONST ;            // ARC_THICKNESS PHC_THICKNESS
    const NPFold* QEshape ; 

    std::vector<const NP*> v_rindex ;
    std::vector<const NP*> v_qeshape ; 
    std::vector<LCQS>      v_lcqs ;    // NUM_LPMT  

    NP* rindex ;    // (NUM_PMTCAT, NUM_LAYER, NUM_PROP, N_EN, 2:[energy,value] )
    NP* qeshape ;   // (NUM_PMTCAT, EN_SAMPLES~44, 2:[energy,value] )
    NP* lcqs ;      // (NUM_LPMT, 2)
    NP* thickness ; // (NUM_PMTCAT, NUM_LAYER, 1:value ) 

    float* tt ; 
};


const int SPMT::N_LPMT = ssys::getenvint("N_LPMT", 1 ); // 10 LPMT default for fast scanning  
const int SPMT::N_MCT  = ssys::getenvint("N_MCT",  180 );  // "AOI" (actually mct) scan points from -1. to 1. 
const int SPMT::N_SPOL = ssys::getenvint("N_SPOL", 1 ); // polarization scan points from S-pol to P-pol 


inline const NPFold* SPMT::Serialize(const char* path) // static 
{
    SPMT* spmt = SPMT::Load(path); 
    const NPFold* fold = spmt ? spmt->serialize() : nullptr ;  
    return fold ; 
}

inline SPMT* SPMT::Load(const char* path_)
{
    const char* path = path_ != nullptr ? path_ : PATH ;  
    NPFold* fold = NPFold::LoadIfExists(path) ; 
    if(VERBOSE) printf("SPMT::Load path %s \n", ( path == nullptr ? "path-null" : path ) );  
    if(VERBOSE) printf("SPMT::Load fold %s \n", ( fold == nullptr ? "fold-null" : "fold-ok" ) );  
    return fold == nullptr ? nullptr : new SPMT(fold) ; 
}

inline SPMT::SPMT(const NPFold* jpmt_)
    :
    ExecutableName(sproc::ExecutableName()),
    jpmt(jpmt_),
    PMT_RINDEX(     jpmt ? jpmt->get_subfold("PMT_RINDEX")      : nullptr ),
    PMTSimParamData(jpmt ? jpmt->get_subfold("PMTSimParamData") : nullptr ),
    MPT(            PMTSimParamData ? PMTSimParamData->get_subfold("MPT")   : nullptr ),
    CONST(          PMTSimParamData ? PMTSimParamData->get_subfold("CONST") : nullptr ),
    QEshape(        PMTSimParamData ? PMTSimParamData->get_subfold("QEshape") : nullptr ),
    v_lcqs(NUM_LPMT),
    rindex(nullptr), 
    qeshape(nullptr),
    lcqs(nullptr),
    thickness(NP::Make<float>(NUM_PMTCAT, NUM_LAYER, 1)),
    tt(thickness->values<float>())
{
    init(); 
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
    init_rindex_thickness(); 
    init_qeshape(); 
    init_lcqs(); 
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
SPMT::init_qeshape
-------------------

1. selects just the LPMT relevant 3 PMTCAT 
2. energy domain scaled to use eV whilst still in double 
   and before combination to avoid stomping on last column int 
   (HMM NP::Combine may have special handling to preserve that now?)

**/

inline void SPMT::init_qeshape()
{
    if(QEshape == nullptr) return ; 
    int QEshape_items = QEshape->num_items() ; 
    if(VERBOSE) std::cout << "SPMT::init_qeshape QEshape_items : " << QEshape_items << std::endl ; 

    std::vector<std::string> names ; 
    U::Split(QEshape_PMTCAT_NAMES, ',', names ); 
    
    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* k = names[i].c_str();  
        const NP* v = QEshape->get(k); 
        if(VERBOSE) std::cout << std::setw(20) << k << " : " << ( v ? v->sstr() : "-" ) << std::endl ; 

        NP* vc = v->copy(); 
        vc->pscale(1e6, 0);  // MeV to eV 
        NP* vn = NP::MakeWithType<float>(vc);   // narrow 

        v_qeshape.push_back(vn) ;
    }
    qeshape = NP::Combine(v_qeshape);
    qeshape->set_names(names); 
}

/**
SPMT::init_lcqs
-----------------

1. get lpmtCat, qeScale arrays from PMTSimParamData NPFold
2. check appropriate sizes with info for all NUM_LPMT 17612 
3. populate v_lcqs vector of LCQS struct holding int:lc 
   "local 0/1/2 pmtcat" and float:qeScale
4. convert the vector of LCQS struct into lcqs array 

NB EVEN WHEN TESTING WITH REDUCED N_LPMT STILL NEED TO INCLUDE INFO FOR ALL 17612 LPMT

**/

inline void SPMT::init_lcqs()
{
    assert( PMTSimParamData ); 
    const NP* lpmtCat = PMTSimParamData->get("lpmtCat") ;   
    assert( lpmtCat && lpmtCat->uifc == 'i' && lpmtCat->ebyte == 4 ); 
    assert( lpmtCat->shape[0] == NUM_LPMT ); 
    const int* lpmtCat_v = lpmtCat->cvalues<int>(); 

    const NP* qeScale = PMTSimParamData->get("qeScale") ;   
    assert( qeScale && qeScale->uifc == 'f' && qeScale->ebyte == 8 ); 
    assert( qeScale->shape[0] >= NUM_LPMT );  // SPMT, WPMT info after LPMT 
    const double* qeScale_v = qeScale->cvalues<double>(); 

    for(int i=0 ; i < NUM_LPMT ; i++ )
    {
        v_lcqs[i] = { TranslateCat(lpmtCat_v[i]), float(qeScale_v[i]) } ; 
    }
    lcqs = NPX::ArrayFromVec<int,LCQS>( v_lcqs ) ; 

    if(VERBOSE) std::cout 
       << "SPMT::init_lcqs" << std::endl 
       << " NUM_LPMT " << NUM_LPMT << std::endl 
       << " lpmtCat " << ( lpmtCat ? lpmtCat->sstr() : "-" ) << std::endl
       << " qeScale " << ( qeScale ? qeScale->sstr() : "-" ) << std::endl
       << " lcqs " << ( lcqs ? lcqs->sstr() : "-" ) << std::endl 
       ; 

    assert( lcqs->shape[0] == NUM_LPMT );
    assert( NUM_LPMT == 17612 );  
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
        default: lcat = -99  ; break ; 
    }
    assert( lcat >= 0 ); 
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
void SPMT::get_lpmtid_stackspec( quad4& spec, int pmtid, float energy_eV) const
{
    spec.zero(); 

    int& cat = spec.q0.i.w ; 
    float& qe_scale = spec.q1.f.w ; 
    float& qe = spec.q2.f.w ; 
    float& _qe = spec.q3.f.w ;   
    // above are refs to locations currently all holding zero

    get_lcqs(cat, qe_scale, pmtid);

    assert( cat > -1 && cat < NUM_PMTCAT ); 
    assert( qe_scale > 0.f ); 

    qe = get_pmtcat_qe(cat, energy_eV ) ; // qeshape interpolation 
    _qe = qe*qe_scale ; 

    bool expected_range = _qe > 0.f && _qe < 1.f ; 

    if(!expected_range) std::cout 
        << "SPMT::get_pmtid_stackspec"
        << " expected_range " << ( expected_range ? "YES" : "NO " )
        << " pmtid " << pmtid 
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
SPMT::make_sscan
-----------------

Scan over (lpmtid, wl, mct, spol )

Potentially could scan over any of (ni,nj,nk,nl)
so should add arrays of the ranges which will have 
only one value when not scanned over.

**/


inline NPFold* SPMT::make_sscan() const 
{
    std::cout << "[SPMT::make_sscan " << std::endl; 
    NPFold* fold = new NPFold ; 

    const NP* lpmtid_domain = Make_LPMTID_LIST() ; 
    const NP* lpmtcat_domain = get_lpmtcat( lpmtid_domain );  
    const NP* mct_domain = NP::MinusCosThetaLinearAngle<float>( N_MCT ); 
    const NP* st_domain = NP::SqrtOneMinusSquare(mct_domain) ; 

    std::cout << " N_MCT " << N_MCT << std::endl ; 
    std::cout << " mct_domain.desc " << mct_domain->desc() << std::endl ; 

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

    std::cout << "SPMT::get_ARTE args " << args->sstr() << std::endl ; 

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
        if( i % 100 == 0 ) std::cout << "SPMT::get_ARTE lpmtid " << lpmtid << std::endl ; 
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
                  if(!minus_cos_theta_diff_expect) std::cerr << "SPMT::get_ARTE minus_cos_theta_diff_expect " << std::endl ; 
                  float sin_theta_0 = sqrt( 1.f - minus_cos_theta*minus_cos_theta ); 
                  float sin_theta_diff = std::abs(sin_theta - sin_theta_0)  ; 
                  bool sin_theta_expect = sin_theta_diff < 1e-6 ; 
                  if(!sin_theta_expect) std::cout 
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
                  float s_pol_frac = get_frac(l, nl) ; 
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

                  if(VERBOSE) std::cout 
                      << "SPMT::get_ARTE"
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
    std::cout << "]SPMT::make_sscan " << std::endl; 
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
SPMT::get_lpmtcat
-------------------

For lpmtid (0->17612-1) returns 0, 1 or 2 corresponding to NNVT, HAMA, NNVT_HiQE

**/

inline int SPMT::get_lpmtcat(int lpmtid) const 
{
    assert( lpmtid >= 0 && lpmtid < NUM_LPMT );  
    const int* lcqs_i = lcqs->cvalues<int>() ; 
    return lcqs_i[lpmtid*2+0] ; 
}
inline int SPMT::get_lpmtcat( int* lpmtcat_ , const int* lpmtid_ , int num ) const 
{
    for(int i=0 ; i < num ; i++)
    {
        int lpmtid = lpmtid_[i] ;
        int lpmtcat = get_lpmtcat(lpmtid) ;
        lpmtcat_[i] = lpmtcat ;
    }
    return num ;
}


inline NP* SPMT::get_lpmtcat(const NP* lpmtid ) const 
{
    assert( lpmtid->shape.size() == 1 ); 
    int num = lpmtid->shape[0] ; 
    NP* lpmtcat = NP::Make<int>(num);  
    int num2 = get_lpmtcat( lpmtcat->values<int>(), lpmtid->cvalues<int>(), num ) ; 

    bool num_expect = num2 == num ; 
    assert( num_expect ); 
    if(!num_expect) std::raise(SIGINT); 

    return lpmtcat ; 
}

inline NP* SPMT::get_lpmtcat() const 
{
    std::cout << "SPMT::get_lpmtcat " << std::endl ; 
    NP* a = NP::Make<int>( NUM_LPMT ) ; 
    int* aa = a->values<int>(); 
    for(int i=0 ; i < NUM_LPMT ; i++) aa[i] = get_lpmtcat(i) ; 
    return a ; 
}

inline float SPMT::get_qescale(int lpmtid) const 
{
    assert( lpmtid >= 0 && lpmtid < NUM_LPMT );  
    const float* lcqs_f = lcqs->cvalues<float>() ; 
    return lcqs_f[lpmtid*2+1] ; 
}
inline NP* SPMT::get_qescale() const 
{
    std::cout << "SPMT::get_qescale " << std::endl ; 
    NP* a = NP::Make<float>( NUM_LPMT ) ; 
    float* aa = a->values<float>(); 
    for(int i=0 ; i < NUM_LPMT ; i++) aa[i] = get_qescale(i) ; 
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
inline void SPMT::get_lcqs(int& lc, float& qs, int lpmtid) const 
{
    assert( lpmtid >= 0 && lpmtid < NUM_LPMT );  
    const int*   lcqs_i = lcqs->cvalues<int>() ; 
    const float* lcqs_f = lcqs->cvalues<float>() ; 
    lc = lcqs_i[lpmtid*2+0] ; 
    qs = lcqs_f[lpmtid*2+1] ; 
}
/**
HUH: doesnt this just duplicate lcqs ? 
**/
inline NP* SPMT::get_lcqs() const 
{
    std::cout << "SPMT::get_lcqs " << std::endl ; 
    int ni = NUM_LPMT ; 
    int nj = 2 ; 
    NP* a = NP::Make<int>(ni, nj) ; 
    int* ii   = a->values<int>() ; 
    float* ff = a->values<float>() ; 
    for(int i=0 ; i < ni ; i++) get_lcqs( ii[i*nj+0], ff[i*nj+1], i ); 
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
SPMT::get_pmtid_qe
--------------------

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

inline float SPMT::get_pmtid_qe(int pmtid, float energy_eV) const
{
    int cat(-1) ; 
    float qe_scale(-1.f) ; 

    get_lcqs(cat, qe_scale, pmtid); 

    assert( cat == 0 || cat == 1 || cat == 2 ); 
    assert( qe_scale > 0.f ); 

    float qe = get_pmtcat_qe(cat, energy_eV);
    qe *= qe_scale ; 

    return qe ; 
}




inline NP* SPMT::get_pmtid_qe() const 
{
    std::cout << "SPMT::get_pmtid_qe " << std::endl ; 
    int ni = N_LPMT ;  
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
        aa[idx+1] = get_pmtid_qe(i, en) ; 
    }
    return a ; 
}


inline NPFold* SPMT::make_testfold() const 
{
    std::cout << "[ SPMT::make_testfold " << std::endl ; 

    NPFold* f = new NPFold ; 

/*
    f->add("get_pmtcat", get_pmtcat() ); 
    f->add("get_qescale", get_qescale() ); 
    f->add("get_lcqs", get_lcqs() ); 
    f->add("get_pmtcat_qe", get_pmtcat_qe() ); 
    f->add("get_pmtid_qe", get_pmtid_qe() ); 
    f->add("get_rindex", get_rindex() ); 
    f->add("get_qeshape", get_qeshape() ); 
    f->add("get_thickness_nm", get_thickness_nm() ); 
    f->add("get_stackspec", get_stackspec() ); 
*/

#ifdef WITH_CUSTOM4
    f->add_subfold("sscan", make_sscan() ); 
#endif

    std::cout << "] SPMT::make_testfold " << std::endl ; 
    return f ; 
}
