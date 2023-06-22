/**
SPMT.h : summarize PMTSimParamData NPFold into the few arrays needed on GPU
============================================================================

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

  * _PMTSimParamData::Load from "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/jpmt/PMTSimParamData"
  * test a few simple queries against the loaded PMTSimParamData 
  * does _PMTSimParamData::Scan_pmtid_qe 

* Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/tests/PMTAccessor_test.sh

  * PMTAccessor::Load from "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/jpmt" 
  * standalone CPU use of PMTAccessor to do the stack calc  

* qudarap/tests/QPMTTest.sh 

  * formerl used JPMT NP_PROP_BASE loading rindex and thickness, not qeshape and lcqs
  * DONE: add in SPMT.h rather than JPMT and extend to include qeshape and lcqs
  * DONE: on GPU interpolation check using QPMT


**/

#include "NPFold.h"
#include <cstdio>
#include "scuda.h"
#include "squad.h"
#include "ssys.h"

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
    static constexpr const char* PATH = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/jpmt" ; 

    static constexpr int NUM_PMTCAT = 3 ; // (NNVT, HAMA, NNVT_HiQE)
    static constexpr int NUM_LAYER = 4 ;  // (Pyrex, ARC, PHC, Vacuum) 
    static constexpr int NUM_PROP = 2 ;   // (RINDEX,KINDEX) real and imaginary parts of the index

    static constexpr const int NUM_LPMT = 17612 ;  // N_LPMT must be less than or equal to NUM_LPMT  

    // below three can be changed via envvars
    static const int N_LPMT ;  
    static const int N_MCT ;     
    static const int N_SPOL ; 
 
    static constexpr const char* QEshape_PMTCAT_NAMES = "QEshape_NNVT.npy,QEshape_R12860.npy,QEshape_NNVT_HiQE.npy" ; 
    // follows PMTCategory kPMT enum order but using QEshape array naming convention
    // because there is no consistent naming convention have to do these dirty things 

    static SPMT* Load(const char* path=nullptr); 
    SPMT(const NPFold* jpmt); 

    void init(); 
    void init_rindex_thickness(); 
    void init_qeshape(); 
    void init_lcqs(); 
    static int TranslateCat(int lpmtcat); 

    std::string desc() const ; 
    void save(const char* dir) const ; 

    float get_frac(int i, int ni) const ; 
    float get_energy(int j, int nj) const ; 
    float get_wavelength(int j, int nj) const ; 
    float get_minus_cos_theta(int k, int nk ) const ;

    float get_rindex(int cat, int layr, int prop, float energy_eV) const ; 
    NP* get_rindex() const ; 

    float get_qeshape(int cat, float energy_eV) const ; 
    NP* get_qeshape() const ; 

    float get_thickness_nm(int cat, int layr) const ; 
    NP* get_thickness_nm() const ; 

    void get_pmtid_stackspec( quad4& spec, int pmtid, float energy_eV) const ;  // EXPT

#ifdef WITH_CUSTOM4

    struct SPMTData
    {
        float4             args ;   // (1,4)
        float4             ARTE ;   // (1,4) 
        quad4              spec ;   // (4,4) 
        StackSpec<float,4> ss ;     // (4,4) 
        Stack<float,4>     stack ;        // (44,4)
        Stack<float,4>     stackNormal ;  // (44,4)
                                    // --------
                                    // (98,4)

        const float* cdata() const { return &args.x ; }
    };

    void annotate( NP* art ) const ; 


    void get_ARTE(SPMTData& pd, int pmtid, float wavelength_nm, float minus_cos_theta, float dot_pol_cross_mom_nrm ) const ; 
    NPFold* get_ARTE() const ; 
#endif

    void get_stackspec( quad4& spec, int cat, float energy_eV) const ; 
    NP*  get_stackspec() const ; 

    int   get_pmtcat(int pmtid) const ; 
    NP*   get_pmtcat() const ; 

    float get_qescale(int pmtid) const ; 
    NP*   get_qescale() const ; 

    void  get_lcqs( int& lc, float& qs, int pmtid ) const ; 
    NP*   get_lcqs() const ; 

    float get_pmtcat_qe(int cat, float energy_eV) const ; 
    NP*   get_pmtcat_qe() const ; 

    float get_pmtid_qe(int pmtid, float energy_eV) const ; 
    NP*   get_pmtid_qe() const ; 

    NPFold* make_testfold() const ; 


    const NPFold* jpmt ; 
    const NPFold* PMT_RINDEX ;       // PyrexRINDEX and VacuumRINDEX 
    const NPFold* PMTSimParamData ; 
    const NPFold* MPT ;              // ARC_RINDEX, ARC_KINDEX, PHC_RINDEX, PHC_KINDEX
    const NPFold* CONST ;            // ARC_THICKNESS PHC_THICKNESS
    const NPFold* QEshape ; 

    std::vector<const NP*> v_rindex ;
    std::vector<const NP*> v_qeshape ; 
    std::vector<LCQS>      v_lcqs ; ; 

    NP* rindex ;    // (NUM_PMTCAT, NUM_LAYER, NUM_PROP, N_EN, 2:[energy,value] )
    NP* qeshape ;   // (NUM_PMTCAT, EN_SAMPLES~44, 2:[energy,value] )
    NP* lcqs ;      // (N_LPMT, 2)
    NP* thickness ; // (NUM_PMTCAT, NUM_LAYER, 1:value ) 

    float* tt ; 
};


const int SPMT::N_LPMT = ssys::getenvint("N_LPMT", 1 ); // 10 LPMT default for fast scanning  
const int SPMT::N_MCT  = ssys::getenvint("N_MCT",  1 );  // "AOI" (actually mct) scan points from -1. to 1. 
const int SPMT::N_SPOL = ssys::getenvint("N_SPOL", 1 ); // polarization scan points from S-pol to P-pol 


inline SPMT* SPMT::Load(const char* path_)
{
    const char* path = path_ != nullptr ? path_ : PATH ;  
    NPFold* fold = NPFold::Load(path) ; 
    if(VERBOSE) printf("SPMT::Load path %s \n", ( path == nullptr ? "path-null" : path ) );  
    if(VERBOSE) printf("SPMT::Load fold %s \n", ( fold == nullptr ? "fold-null" : "fold-ok" ) );  
    return fold == nullptr ? nullptr : new SPMT(fold) ; 
}

inline SPMT::SPMT(const NPFold* jpmt_)
    :
    jpmt(jpmt_),
    PMT_RINDEX(     jpmt ? jpmt->get_subfold("PMT_RINDEX")      : nullptr ),
    PMTSimParamData(jpmt ? jpmt->get_subfold("PMTSimParamData") : nullptr ),
    MPT(            PMTSimParamData ? PMTSimParamData->get_subfold("MPT")   : nullptr ),
    CONST(          PMTSimParamData ? PMTSimParamData->get_subfold("CONST") : nullptr ),
    QEshape(        PMTSimParamData ? PMTSimParamData->get_subfold("QEshape") : nullptr ),
    v_lcqs(N_LPMT),
    rindex(nullptr), 
    qeshape(nullptr),
    lcqs(nullptr),
    thickness(NP::Make<float>(NUM_PMTCAT, NUM_LAYER, 1)),
    tt(thickness->values<float>())
{
    init(); 
}

inline void SPMT::init()
{
    init_rindex_thickness(); 
    init_qeshape(); 
    init_lcqs(); 
}

/**
SPMT::init_rindex_thickness
------------------------------

Note similarity to JPMT::init_rindex_thickness

Notice stupid vacuum rindex::

    In [19]: t.rindex[0,3]
    Out[19]: 
    array([[[ 1.55,  1.  ],
            [ 6.2 ,  1.  ],
            [10.33,  1.  ],
            [15.5 ,  1.  ],
            [ 0.  ,  0.  ],
            [ 0.  ,  0.  ],


Fixed this with NP::MakePCopyNotDumb

**/

inline void SPMT::init_rindex_thickness()
{
    if(MPT == nullptr || CONST == nullptr) return ; 
    int MPT_sub = MPT->get_num_subfold() ; 
    int CONST_items = CONST->num_items() ; 

    assert( MPT_sub == NUM_PMTCAT );    // NUM_PMTCAT:3
    assert( CONST_items == NUM_PMTCAT ); 

    double dscale = 1e-6 ;  // make energy domain scale consistent 

    for(int i=0 ; i < NUM_PMTCAT ; i++)
    {
        const char* name = MPT->get_subfold_key(i) ; 
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
    const std::vector<int>& shape = rindex->shape ; 
    assert( shape.size() == 3 );
    rindex->change_shape( NUM_PMTCAT, NUM_LAYER, NUM_PROP, shape[shape.size()-2], shape[shape.size()-1] );
}

/**
SPMT::init_qeshape
-------------------

NB selects just the relevant 3 PMTCAT 

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
        // domain scale whilst still double and before combination
        // to avoid stomping on last column int 

        NP* vn = NP::MakeWithType<float>(vc);   // narrow 

        v_qeshape.push_back(vn) ;
    }
    qeshape = NP::Combine(v_qeshape); 
    qeshape->set_names(names); 
}

/**
SPMT::init_lcqs
-----------------

jcv _PMTSimParamData::

    217     NP* pmtID = NPX::ArrayFromVec<int, int>(data.m_all_pmtID) ;
    218     NP* qeScale = NPX::ArrayFromVec<double,double>(data.m_all_pmtID_qe_scale) ;
    219     NP* lpmtCat = NPX::ArrayFromMap<int, int>(data.m_map_pmt_category) ;
    220     NP* pmtCat = NPX::ArrayFromDiscoMap<int>(data.m_all_pmt_category) ;
    221     NP* pmtCatVec = NPX::ArrayFromVec<int, int>(data.m_all_pmt_catvec) ;

jcv PMTSimParamSvc::

     613 bool PMTSimParamSvc::init_PMTParamSvc()
     614 {
     615     int num_lpmt = m_PMTParamSvc->get_NTotal_CD_LPMT() ;
     616     for(int i = 0 ; i < num_lpmt ; i++)
     617     {
     618         int pmtid = kOFFSET_CD_LPMT + i ;
     619         int cat = m_PMTParamSvc->getPMTCategory(pmtid) ;
     620         m_map_pmt_category[i] = cat ;
     621     }
     622 
     623     m_PmtTotal_WP = m_PMTParamSvc->get_NTotal_WP_LPMT() ;
     624 
     625 
     626    return true;
     627 
     628 }

     /// m_map_pmt_category is a pointless map : contiguous key and base key 0 

::

    In [7]: t.lpmtCat.shape
    Out[7]: (17612, 1)

    In [8]: t.lpmtCat[:,0]
    Out[8]: array([1, 1, 3, 1, 3, ..., 1, 1, 1, 1, 1], dtype=int32)

    In [9]: np.unique( t.lpmtCat[:,0], return_counts=True )
    Out[9]: (array([0, 1, 3], dtype=int32), array([2720, 4997, 9895]))

    In [16]: t.qeScale[17600:17612,0]
    Out[16]: array([1.083, 1.024, 1.095, 1.033, 1.097, 1.031, 1.052, 1.041, 1.048, 1.044, 1.037, 1.076])

    In [17]: t.qeScale[17612:,0]
    Out[17]: array([1.099, 1.099, 1.099, 1.099, 1.099, ..., 1.032, 0.975, 1.005, 1.005, 1.044])

    In [18]: t.qeScale.shape
    Out[18]: (45612, 1)

    In [19]: 17612+2400+25600
    Out[19]: 45612

    In [22]: np.all( t.qeScale[17612:17612+2400,0] == t.qeScale[17612,0] )
    Out[22]: True

    In [24]: t.qeScale[17612+2400:17612+2400+25600].T
    Out[24]: array([[1.022, 1.049, 0.982, 0.977, 1.008, ..., 1.032, 0.975, 1.005, 1.005, 1.044]])



    In [7]: t.lcqs[:,1].view(np.float32)
    Out[7]: array([1.025, 1.062, 1.401, 1.036, 1.286, ..., 1.041, 1.048, 1.044, 1.037, 1.076], dtype=float32)

**/


inline void SPMT::init_lcqs()
{
    const NP* lpmtCat = PMTSimParamData->get("lpmtCat") ;   
    assert( lpmtCat && lpmtCat->uifc == 'i' && lpmtCat->ebyte == 4 ); 
    //assert( lpmtCat->shape[0] == N_LPMT );  // not true with reduced N_LPMT via envvar 
    const int* lpmtCat_v = lpmtCat->cvalues<int>(); 

    const NP* qeScale = PMTSimParamData->get("qeScale") ;   
    assert( qeScale && qeScale->uifc == 'f' && qeScale->ebyte == 8 ); 
    assert( qeScale->shape[0] > N_LPMT );  // HUH ?
    const double* qeScale_v = qeScale->cvalues<double>(); 

    //if(VERBOSE) 
    std::cout 
       << "SPMT::init_lcqs" << std::endl 
       << " lpmtCat " << ( lpmtCat ? lpmtCat->sstr() : "-" ) << std::endl
       << " qeScale " << ( qeScale ? qeScale->sstr() : "-" ) << std::endl
       ; 

    for(int i=0 ; i < N_LPMT ; i++ ) v_lcqs[i] = { TranslateCat(lpmtCat_v[i]), float(qeScale_v[i]) } ; 
    lcqs = NPX::ArrayFromVec<int,LCQS>( v_lcqs ) ; 

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


inline std::string SPMT::desc() const
{
    std::stringstream ss ; 
    ss << "SPMT::desc" << std::endl ; 
    ss << "jpmt.loaddir " << ( jpmt && jpmt->loaddir ? jpmt->loaddir : "NO_LOAD" ) << std::endl ; 

    /*
    ss << "jpmt.desc " << std::endl ; 
    ss << ( jpmt ? jpmt->desc() : "NO_JPMT" ) << std::endl ; 

    ss << "PMT_RINDEX.desc " << std::endl ; 
    ss << ( PMT_RINDEX ? PMT_RINDEX->desc() : "NO_PMT_RINDEX" ) << std::endl ; 
    //ss << "PMTSimParamData.desc " << std::endl ; 
    //ss << ( PMTSimParamData ? PMTSimParamData->desc() : "NO_PMTSimParamData" ) << std::endl ; 
    ss << "jpmt/PMTSimParamData/MPT " << std::endl << std::endl ; 
    ss << ( MPT ? MPT->desc() : "NO_MPT" ) << std::endl ; 
    */
 
    ss << "rindex " << ( rindex ? rindex->sstr() : "-" ) << std::endl ; 
    ss << "thickness " << ( thickness ? thickness->sstr() : "-" ) << std::endl ; 
    ss << "qeshape " << ( qeshape ? qeshape->sstr() : "-" ) << std::endl ; 
    ss << "lcqs " << ( lcqs ? lcqs->sstr() : "-" ) << std::endl ; 


    std::string str = ss.str(); 
    return str ; 
}

inline void SPMT::save(const char* dir) const
{
    std::cout << "[ SPMT::save " << dir << std::endl ; 
    rindex->save(dir, "rindex.npy"); 
    thickness->save(dir, "thickness.npy"); 
    qeshape->save(dir, "qeshape.npy"); 
    lcqs->save(dir, "lcqs.npy"); 
    std::cout << "] SPMT::save " << dir << std::endl ; 
}

inline float SPMT::get_frac(int i, int ni) const
{
   return ni == 1 ? 0.f : float(i)/float(ni-1) ; 
}

inline float SPMT::get_energy(int j, int nj) const
{
    assert( j < nj ); 
    float fr = get_frac(j, nj); 
    float en = EN0*(1.f-fr) + EN1*fr ; 
    return en ; 
}

inline float SPMT::get_wavelength(int j, int nj) const
{
    assert( j < nj ); 
    float fr = get_frac(j, nj); 
    float wl = WL0*(1.f-fr) + WL1*fr ; 
    return wl ; 
}


inline float SPMT::get_minus_cos_theta(int k, int nk ) const 
{
    assert( k < nk ); 
    float fr = get_frac(k, nk); 
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
void SPMT::get_pmtid_stackspec( quad4& spec, int pmtid, float energy_eV) const
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
    
    * dot(pol,cross(mom,nrm)) in float3 form 
    * OldPolarization*OldMomentum.cross(theRecoveredNormal) in G4ThreeVector form 
    * expresses degree of S vs P polarization of the incident photon
    * pol,mom,nrm all expected to be normalized vectors
    * cross(mom,nrm) vector is transverse to plane of incidence which contains mom and nrm by definition
    * cross(mom,nrm) has magnitude sine(angle_between_mom_and_nrm)   
    * cross(mom,nrm) becomes zero at normal incidence, but S/P have no meaning in that case anyhow
    * dot(pol,cross(mom,nrm)) value is cosine(angle_between_pol_and_transverse_vector)*sine(angle_between_mom_and_nrm)
    * NB when testing the value of dot_pol_cross_mom_nrm needs to be consistent with minus_cos_theta

**NB : LAST TWO ARGS ARE RELATED**

dot_pol_cross_mom_nrm includes cross(mom,nrm) and minus_cos_theta is dot(mom,nrm) 
so one incorporates the cross product and the other is the dot product 
of the same two vectors. Hence care is needed to prepare correctly 
related arguments when test scanning the API.

**/

inline void SPMT::get_ARTE(SPMTData& pd, int pmtid, float wavelength_nm, float minus_cos_theta, float dot_pol_cross_mom_nrm ) const
{
    const float energy_eV = hc_eVnm/wavelength_nm ; 
    get_pmtid_stackspec(pd.spec, pmtid, energy_eV); 

    float4& args = pd.args ; 
    float4& ARTE = pd.ARTE ; 
    quad4& spec = pd.spec ; 
    StackSpec<float,4>& ss = pd.ss ;
    Stack<float,4>& stack = pd.stack ; 
    Stack<float,4>& stackNormal = pd.stackNormal ; 

    args.x = pmtid ; 
    args.y = energy_eV ; 
    args.z = minus_cos_theta ; 
    args.w = dot_pol_cross_mom_nrm ; 

    const float& _qe = spec.q3.f.w ; 
    // HMM: annoying shuffling : allows C4MultiLayerStack.h very independent


    ss.ls[0].nr = spec.q0.f.x ; 

    ss.ls[1].nr = spec.q1.f.x ; 
    ss.ls[1].ni = spec.q1.f.y ; 
    ss.ls[1].d  = spec.q1.f.z ;
 
    ss.ls[2].nr = spec.q2.f.x ; 
    ss.ls[2].ni = spec.q2.f.y ; 
    ss.ls[2].d  = spec.q2.f.z ;

    ss.ls[3].nr = spec.q3.f.x ; 
 

    stack.calc(wavelength_nm, minus_cos_theta, ss );


    const float _si = stack.ll[0].st.real() ; // sqrt(one - minus_cos_theta*minus_cos_theta) 
    float E_s2 = _si > 0.f ? dot_pol_cross_mom_nrm/_si : 0.f ;
    E_s2 *= E_s2;

    // E_s2 : S-vs-P power fraction : signs make no difference as squared
    // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 

    const float one = 1.f ;
    const float S = E_s2 ;
    const float P = one - S ;

    const float T = S*stack.art.T_s + P*stack.art.T_p ;  // matched with TransCoeff see sysrap/tests/stmm_vs_sboundary_test.cc
    const float R = S*stack.art.R_s + P*stack.art.R_p ;
    const float A = S*stack.art.A_s + P*stack.art.A_p ;
    //const float A1 = one - (T+R);  // note that A1 matches A 

    stack.art.xx = A ; 
    stack.art.yy = R ; 
    stack.art.zz = T ; 
    stack.art.ww = S ; 
    
    // backwards _qe is set to zero (this is for +ve minus_cos_theta) 
    // so could skip stackNormal.calc for backwards incidence after debugged 

    stackNormal.calc(wavelength_nm, -1.f , ss );
    float An = 1.f - (stackNormal.art.T + stackNormal.art.R) ;
    float E = minus_cos_theta < 0.f ? _qe/An : 0.f ;  // aka theEfficiency and escape_fac 
    //float E = 0.f ; 

    ARTE.x = A ;         // aka theAbsorption
    ARTE.y = R/(1.f-A) ; // aka theReflectivity
    ARTE.z = T/(1.f-A) ; // aka theTransmittance
    ARTE.w = E ; 
}


/**
SPMT::annotate
----------------

Calling this tickles some bug sprinkling nan across the arrays 

HUH : its not happening any more ?  HMM something flakey waiting to strike. 

**/

inline void SPMT::annotate( NP* art ) const 
{
    std::string title = "SPMT.title"  ; 
    std::string brief = "SPMT.brief" ; 
    std::string name = "SPMT.name" ; 
    std::string label = "SPMT.label" ; 

    art->set_meta<std::string>("title", title ); 
    art->set_meta<std::string>("brief", brief ); 
    art->set_meta<std::string>("name", name ); 
    art->set_meta<std::string>("label", label ); 
}


inline NPFold* SPMT::get_ARTE() const 
{
    NPFold* fold = new NPFold ; 
    SPMTData pd ; 

    int ni = N_LPMT ; 
    int nj = N_WL ; 
    int nk = N_MCT ; 
    int nl = N_SPOL ; 

    std::cout << "[ SPMT::get_ARTE "  << std::endl ; 

    NP* args  = NP::Make<float>(ni, nj, nk, nl, 4 ); 
    NP* ARTE  = NP::Make<float>(ni, nj, nk, nl, 4 ); 
    NP* spec  = NP::Make<float>(ni, nj, nk, nl, 4, 4 ); 
    NP* ss    = NP::Make<float>(ni, nj, nk, nl, 4, 4 ); 

    // Make_ allows arbitrary dimensions     
    NP* stack = NP::Make_<float>(ni, nj, nk, nl, 44, 4 ); 
    NP* ll    = NP::Make_<float>(ni, nj, nk, nl, 4, 4, 4, 2) ; 
    NP* comp  = NP::Make_<float>(ni, nj, nk, nl, 1, 4, 4, 2) ; 
    NP* art   = NP::Make_<float>(ni, nj, nk, nl, 4, 4) ; 

    NP* nstack = NP::Make_<float>(ni, nj, nk, nl, 44, 4 ); 
    NP* nll    = NP::Make_<float>(ni, nj, nk, nl, 4, 4, 4, 2) ; 
    NP* ncomp  = NP::Make_<float>(ni, nj, nk, nl, 1, 4, 4, 2) ; 
    NP* nart   = NP::Make_<float>(ni, nj, nk, nl, 4, 4) ; 

    annotate(art); 

    assert( sizeof(pd.stack)/sizeof(float) == 44*4 ); 

    fold->add("args", args);
    fold->add("ARTE", ARTE);
    fold->add("spec", spec);
    fold->add("ss", ss);

    fold->add("stack", stack);
    fold->add("ll", ll);
    fold->add("comp", comp);
    fold->add("art", art);

    fold->add("nstack", nstack);
    fold->add("nll", nll);
    fold->add("ncomp", ncomp);
    fold->add("nart", nart);
 
 
    std::cout << "SPMT::get_ARTE args " << args->sstr() << std::endl ; 

    float* args_v = args->values<float>(); 
    float* ARTE_v = ARTE->values<float>(); 
    float* spec_v = spec->values<float>() ; 
    float* ss_v   = ss->values<float>() ; 

    float* stack_v = stack->values<float>() ; 
    float* ll_v    = ll->values<float>() ; 
    float* comp_v  = comp->values<float>() ; 
    float* art_v   = art->values<float>() ; 

    float* nstack_v = nstack->values<float>() ; 
    float* nll_v    = nll->values<float>() ; 
    float* ncomp_v  = ncomp->values<float>() ; 
    float* nart_v   = nart->values<float>() ; 


    for(int i=0 ; i < ni ; i++)
    {
        int pmtid = i ; 
        if( i % 100 == 0 ) std::cout << "SPMT::get_ARTE pmtid " << pmtid << std::endl ; 
        for(int j=0 ; j < nj ; j++)
        {
           float wavelength_nm = get_wavelength(j, nj ); 
           for(int k=0 ; k < nk ; k++)
           {
              float minus_cos_theta = get_minus_cos_theta(k, nk );  
              float sin_theta = sqrt( 1.f - minus_cos_theta*minus_cos_theta ); 

              for(int l=0 ; l < nl ; l++)
              {
                  float s_pol_frac = get_frac(l, nl) ; 
                  float dot_pol_cross_mom_nrm = sin_theta*s_pol_frac ; 

                  get_ARTE(pd, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm ); 

                  int idx = i*nj*nk*nl*4 + j*nk*nl*4 + k*nl*4 + l*4 ; 

                  int args_idx = idx ; 
                  int ARTE_idx = idx ; 
                  int spec_idx = idx*4 ; 
                  int ss_idx   = idx*4 ; 

                  int stack_idx = idx*44 ; 
                  int ll_idx    = idx*4*4*2 ; // 32 
                  int comp_idx  = idx*1*4*2 ; //  8
                  int art_idx   = idx*4 ;     //  4

                  memcpy( args_v + args_idx,  &pd.args.x, sizeof(float)*4 ); 
                  memcpy( ARTE_v + ARTE_idx,  &pd.ARTE.x, sizeof(float)*4 ); 
                  memcpy( spec_v + spec_idx,   pd.spec.cdata(), sizeof(float)*4*4 ); 
                  memcpy( ss_v   + ss_idx  ,   pd.ss.cdata(),   sizeof(float)*4*4 ); 

                  memcpy( stack_v + stack_idx,  pd.stack.cdata(),        sizeof(float)*44*4 ); 
                  memcpy( ll_v    + ll_idx,     pd.stack.ll[0].cdata(),  sizeof(float)*32*4 ); 
                  memcpy( comp_v  + comp_idx,   pd.stack.comp.cdata(),   sizeof(float)*8*4 ); 
                  memcpy( art_v   + art_idx,    pd.stack.art.cdata(),    sizeof(float)*4*4 ); 

                  memcpy( nstack_v + stack_idx,  pd.stackNormal.cdata(),        sizeof(float)*44*4 ); 
                  memcpy( nll_v    + ll_idx,     pd.stackNormal.ll[0].cdata(),  sizeof(float)*32*4 ); 
                  memcpy( ncomp_v  + comp_idx,   pd.stackNormal.comp.cdata(),   sizeof(float)*8*4 ); 
                  memcpy( nart_v   + art_idx,    pd.stackNormal.art.cdata(),    sizeof(float)*4*4 ); 

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
    std::cout << "] SPMT::get_ARTE " << std::endl ; 
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
SPMT::get_pmtcat
-------------------

For pmtid (0->17612-1) returns 0, 1 or 2 corresponding to NNVT, HAMA, NNVT_HiQE

**/

inline int SPMT::get_pmtcat(int pmtid) const 
{
    assert( pmtid >= 0 && pmtid < N_LPMT );  
    const int* lcqs_i = lcqs->cvalues<int>() ; 
    return lcqs_i[pmtid*2+0] ; 
}
inline NP* SPMT::get_pmtcat() const 
{
    std::cout << "SPMT::get_pmtcat " << std::endl ; 
    NP* a = NP::Make<int>( N_LPMT ) ; 
    int* aa = a->values<int>(); 
    for(int i=0 ; i < N_LPMT ; i++) aa[i] = get_pmtcat(i) ; 
    return a ; 
}

inline float SPMT::get_qescale(int pmtid) const 
{
    assert( pmtid >= 0 && pmtid < N_LPMT );  
    const float* lcqs_f = lcqs->cvalues<float>() ; 
    return lcqs_f[pmtid*2+1] ; 
}
inline NP* SPMT::get_qescale() const 
{
    std::cout << "SPMT::get_qescale " << std::endl ; 
    NP* a = NP::Make<float>( N_LPMT ) ; 
    float* aa = a->values<float>(); 
    for(int i=0 ; i < N_LPMT ; i++) aa[i] = get_qescale(i) ; 
    return a ; 
}

inline void SPMT::get_lcqs(int& lc, float& qs, int pmtid) const 
{
    assert( pmtid >= 0 && pmtid < N_LPMT );  
    const int*   lcqs_i = lcqs->cvalues<int>() ; 
    const float* lcqs_f = lcqs->cvalues<float>() ; 
    lc = lcqs_i[pmtid*2+0] ; 
    qs = lcqs_f[pmtid*2+1] ; 
}
inline NP* SPMT::get_lcqs() const 
{
    std::cout << "SPMT::get_lcqs " << std::endl ; 
    int ni = N_LPMT ; 
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

    NP* _get_pmtcat = get_pmtcat() ; 
    NP* _get_qescale = get_qescale() ;
    NP* _get_lcqs = get_lcqs() ;
    NP* _get_pmtcat_qe = get_pmtcat_qe() ; 
    NP* _get_pmtid_qe = get_pmtid_qe() ; 
    NP* _get_rindex = get_rindex() ; 
    NP* _get_qeshape = get_qeshape() ; 
    NP* _get_thickness_nm = get_thickness_nm() ; 
    NP* _get_stackspec = get_stackspec() ; 


    NPFold* f = new NPFold ; 

/*
    f->add("get_pmtcat", _get_pmtcat ); 
    f->add("get_qescale", _get_qescale ); 
    f->add("get_lcqs", _get_lcqs ); 
    f->add("get_pmtcat_qe", _get_pmtcat_qe ); 
    f->add("get_pmtid_qe", _get_pmtid_qe ); 
    f->add("get_rindex", _get_rindex ); 
    f->add("get_qeshape", _get_qeshape ); 
    f->add("get_thickness_nm", _get_thickness_nm ); 
    f->add("get_stackspec", _get_stackspec ); 
*/

#ifdef WITH_CUSTOM4
    NPFold* _get_ARTE = get_ARTE() ;  
    f->add_subfold("get_ARTE", _get_ARTE ); 
#endif

    std::cout << "] SPMT::make_testfold " << std::endl ; 
    return f ; 
}
