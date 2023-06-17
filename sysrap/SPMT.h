#pragma once
/**
SPMT.h : summarize PMTSimParamData NPFold into the few arrays needed on GPU
============================================================================

Aims
----

1. replace and extend from PMTSim/JPMT.h based off the complete 
   serialized PMT info rather than the partial NP_PROP_BASE rindex, 
   thickness info loaded by JPMT.h. 

   * DONE : (rindex,thickness) matches JPMT after ordering and scale fixes
   * TODO : include some names in source metadata (in _PMTSimParamData?) 
     to makes the order more explicit 

2. DONE : make minimal (if any) use of junosw code. 

3. prepare the inputs to QPMT.h including PMT info arrays to be uploaded to GPU 

   * form a (17612,4) array (pmtcat,qescale,spare,pmtidx) 



Related developments
---------------------

* jcv PMTSimParamData    # core of PMTSimParamSvc
* jcv PMTSimParamSvc     # code that fills the core
* jcv _PMTSimParamData   # code that serializes the core into NPFold 


* j/Layr/JPMT.h 
* j/Layr/JPMTTest.sh

  * loads the props into JPMT and dumps  

* Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/tests/PMTSimParamData.sh 

  * python load the persisted PMTSimParamData 

* Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/tests/PMTSimParamData_test.sh 

  * _PMTSimParamData::Load from "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/jpmt/PMTSimParamData"
  * test a few simple queries against the loaded PMTSimParamData 

* Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/tests/PMTAccessor_test.sh

  * PMTAccessor::Load from "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/jpmt" 
  * standalone CPU use of PMTAccessor to do the stack calc  

* qudarap/tests/QPMTTest.sh 

  * currently used JPMT NP_PROP_BASE loading rindex and thickness
  * TODO: add in SPMT.h source rather than JPMT 
  * on GPU interpolation check using QPMT


**/

#include "NPFold.h"

struct SPMT
{

    struct LCQS { int lc ; float qs ; } ; 
 

    static constexpr const bool VERBOSE = false ; 
    static constexpr const char* PATH = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/jpmt" ; 
    static constexpr int NUM_PMTCAT = 3 ; 
    static constexpr int NUM_LAYER = 4 ; 
    static constexpr int NUM_PROP = 2 ; 
    static constexpr int NUM_LPMT = 17612 ; 
    static constexpr const char* QEshape_PMTCAT_NAMES = "QEshape_NNVT.npy,QEshape_R12860.npy,QEshape_NNVT_HiQE.npy" ; 
    // follows PMTCategory kPMT enum order but using QEshape array naming convention
    // because there is no consistent naming convention have to do these dirty things 


    static SPMT* Load(const char* path=PATH); 
    SPMT(const NPFold* jpmt); 

    void init(); 
    void init_rindex_thickness(); 
    void init_qeshape(); 
    void init_lcqs(); 
    static int TranslateCat(int lpmtcat); 

    std::string desc() const ; 
    void save(const char* dir) const ; 

    const NPFold* jpmt ; 
    const NPFold* PMT_RINDEX ;       // PyrexRINDEX and VacuumRINDEX 
    const NPFold* PMTSimParamData ; 
    const NPFold* MPT ;              // ARC_RINDEX, ARC_KINDEX, PHC_RINDEX, PHC_KINDEX
    const NPFold* CONST ;            // ARC_THICKNESS PHC_THICKNESS
    const NPFold* QEshape ; 

    std::vector<const NP*> v_rindex ;
    std::vector<const NP*> v_qeshape ; 
    std::vector<LCQS>      v_lcqs ; ; 

    NP* rindex ;     // (num_pmtcat, num_layer, num_prop,  num_energies ~15 , num_payload:2 )    # payload is (energy, value)  
    NP* thickness ;  // (num_pmtcat, num_layer, num_payload:1 )
    float* tt ; 
    NP* qeshape ;    // (num_pmtcat, num_energies ~15, num_payload:2 ) 
    NP* lcqs ; 
};

inline SPMT* SPMT::Load(const char* path)
{
    NPFold* f = NPFold::Load(path) ; 
    return new SPMT(f) ; 
}

inline SPMT::SPMT(const NPFold* jpmt_)
    :
    jpmt(jpmt_),
    PMT_RINDEX(     jpmt ? jpmt->get_subfold("PMT_RINDEX")      : nullptr ),
    PMTSimParamData(jpmt ? jpmt->get_subfold("PMTSimParamData") : nullptr ),
    MPT(            PMTSimParamData ? PMTSimParamData->get_subfold("MPT")   : nullptr ),
    CONST(          PMTSimParamData ? PMTSimParamData->get_subfold("CONST") : nullptr ),
    QEshape(        PMTSimParamData ? PMTSimParamData->get_subfold("QEshape") : nullptr ),
    v_lcqs(NUM_LPMT),
    rindex(nullptr), 
    thickness(NP::Make<float>(NUM_PMTCAT, NUM_LAYER, 1)),
    tt(thickness->values<float>()),
    qeshape(nullptr),
    lcqs(nullptr)
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
    assert( lpmtCat->shape[0] == NUM_LPMT ); 
    const int* lpmtCat_v = lpmtCat->cvalues<int>(); 

    const NP* qeScale = PMTSimParamData->get("qeScale") ;   
    assert( qeScale && qeScale->uifc == 'f' && qeScale->ebyte == 8 ); 
    assert( qeScale->shape[0] > NUM_LPMT ); 
    const double* qeScale_v = qeScale->cvalues<double>(); 

    std::cout 
       << "SPMT::init_lcqs" << std::endl 
       << " lpmtCat " << ( lpmtCat ? lpmtCat->sstr() : "-" ) << std::endl
       << " qeScale " << ( qeScale ? qeScale->sstr() : "-" ) << std::endl
       ; 

    for(int i=0 ; i < NUM_LPMT ; i++ ) v_lcqs[i] = { TranslateCat(lpmtCat_v[i]), float(qeScale_v[i]) } ; 
    lcqs = NPX::ArrayFromVec<int,LCQS>( v_lcqs ) ; 

}

/**
SPMT::TranslateCat
--------------------

Translates 0,1,3 into 0,1,2 

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





    In [30]: t.lpmtCat[:200].T
    Out[30]: 
    array([[1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3]], dtype=int32)




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
    rindex->save(dir, "rindex.npy"); 
    thickness->save(dir, "thickness.npy"); 
    qeshape->save(dir, "qeshape.npy"); 
    lcqs->save(dir, "lcqs.npy"); 
}


