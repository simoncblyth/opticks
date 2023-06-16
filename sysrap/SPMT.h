#pragma once
/**
SPMT.h
========

Aims
----

1. replace and extend from PMTSim/JPMT.h based off the complete 
   serialized PMT info rather than the partial NP_PROP_BASE rindex, 
   thickness info loaded by JPMT.h. 

   * first check is that it reproduces the JPMT rindex and thickness arrays

2. make minimal (if any) use of junosw code. 

3. prepare the inputs to QPMT.h including PMT info arrays to be uploaded to GPU 

   * form a (17612,4) array (pmtcat,qescale,spare,pmtidx) 
   * 1st reproduce the JPMT.rindex JPMT.thickness arrays frm PMTAccessor NPFold
   * HMM: dont want to use junosw within opticks so start from NPFold ?



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

  * JPMT NP_PROP_BASE loading rindex and thickness
  * on GPU interpolation check using QPMT
  * TODO: develop SPMT.h to replace JPMT 


**/

#include "NPFold.h"

struct SPMT
{
    static constexpr const char* PATH = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/jpmt" ; 
    static constexpr int NUM_PMTCAT = 3 ; 
    static constexpr int NUM_LAYER = 4 ; 
    static constexpr int NUM_PROP = 2 ; 

    static SPMT* Create(); 
    SPMT(const NPFold* jpmt); 

    void init(); 
    void init_rindex_thickness(); 

    std::string desc() const ; 
    void save(const char* dir) const ; 

    const NPFold* jpmt ; 
    const NPFold* PMT_RINDEX ; 
    const NPFold* PMTSimParamData ; 
    const NPFold* MPT ; 
    const NPFold* CONST ; 

    std::vector<const NP*> v_rindex ;

    NP* rindex ;     // (num_pmtcat, num_layer, num_prop,  num_energies ~15 , num_payload:2 )    # payload is (energy, value)  
    NP* thickness ;  // (num_pmtcat, num_layer, num_payload:1 )
    double* tt ; 


};

inline SPMT* SPMT::Create()
{
    NPFold* f = NPFold::Load(PATH) ; 
    return new SPMT(f) ; 
}

inline SPMT::SPMT(const NPFold* jpmt_)
    :
    jpmt(jpmt_),
    PMT_RINDEX(     jpmt ? jpmt->get_subfold("PMT_RINDEX")      : nullptr ),
    PMTSimParamData(jpmt ? jpmt->get_subfold("PMTSimParamData") : nullptr ),
    MPT(            PMTSimParamData ? PMTSimParamData->get_subfold("MPT")   : nullptr ),
    CONST(          PMTSimParamData ? PMTSimParamData->get_subfold("CONST") : nullptr ),
    rindex(nullptr), 
    thickness(NP::Make<double>(NUM_PMTCAT, NUM_LAYER, 1)),
    tt(thickness->values<double>())
{
    init(); 
}

inline void SPMT::init()
{
    init_rindex_thickness(); 
}

/**
SPMT::init_rindex_thickness
------------------------------

Note similarity to JPMT::init_rindex_thickness

**/

inline void SPMT::init_rindex_thickness()
{
    int MPT_sub = MPT->get_num_subfold() ; 
    int CONST_items = CONST->num_items() ; 

    assert( MPT_sub == NUM_PMTCAT );    // NUM_PMTCAT:3
    assert( CONST_items == NUM_PMTCAT ); 

    for(int i=0 ; i < NUM_PMTCAT ; i++)
    {
        const char* name = MPT->get_subfold_key(i) ; 
        NPFold* pmtcat = MPT->get_subfold(i);
        const NP* pmtconst = CONST->get_array(i); 

        for(int j=0 ; j < NUM_LAYER ; j++)  // NUM_LAYER:4 
        {
            for(int k=0 ; k < NUM_PROP ; k++)   // NUM_PROP:2 (RINDEX,KINDEX) 
            {
                const NP* a = nullptr ;
                switch(j)
                {
                   case 0: a = (k == 0 ? PMT_RINDEX->get("PyrexRINDEX")  : NP::ZEROProp<double>() ) ; break ;
                   case 1: a = pmtcat->get( k == 0 ? "ARC_RINDEX"   : "ARC_KINDEX"   )              ; break ;
                   case 2: a = pmtcat->get( k == 0 ? "PHC_RINDEX"   : "PHC_KINDEX"   )              ; break ;
                   case 3: a = (k == 0 ? PMT_RINDEX->get("VacuumRINDEX") : NP::ZEROProp<double>() ) ; break ;
                }
                v_rindex.push_back(a) ;
            }

            double d = 0. ;
            double scale = 1e9 ; // express thickness in nm (not meters) 
            switch(j)
            {
               case 0: d = 0. ; break ;
               case 1: d = scale*pmtconst->get_named_value<double>("ARC_THICKNESS", -1) ; break ;
               case 2: d = scale*pmtconst->get_named_value<double>("PHC_THICKNESS", -1) ; break ;
               case 3: d = 0. ; break ;
            }
            tt[i*NUM_LAYER + j] = d ;
        }
    }
    rindex = NP::Combine(v_rindex); 
    const std::vector<int>& shape = rindex->shape ; 
    assert( shape.size() == 3 );
    rindex->change_shape( NUM_PMTCAT, NUM_LAYER, NUM_PROP, shape[shape.size()-2], shape[shape.size()-1] );
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
    std::string str = ss.str(); 
    return str ; 
}

inline void SPMT::save(const char* dir) const
{
    rindex->save(dir, "rindex.npy"); 
    thickness->save(dir, "thickness.npy"); 
}


