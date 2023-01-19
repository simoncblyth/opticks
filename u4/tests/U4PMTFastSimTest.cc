
#include "OPTICKS_LOG.hh"

#ifdef WITH_PMTFASTSIM
#include "Geometry/PMT.h"
#include "Geometry/PMTCategory.h"
#include "PMTSimParamSvc/_PMTSimParamData.h"
#include "PMTSimParamSvc/PMTSimParamData.h"
#include "IPMTAccessor.h"
#include "JPMT.h"
#include "G4Material.hh"

#endif

struct PMTAccessor : public IPMTAccessor
{
    static constexpr const char* TypeName = "PMTAccessor" ; 
    const PMTSimParamData* data ; 
    const G4Material* Pyrex  ; 
    const G4Material* Vacuum ; 

    PMTAccessor(const PMTSimParamData* data); 
    std::string desc() const ; 


    double get_pmtid_qe( int pmtid, double energy ) const ; 
    int    get_pmtcat( int pmtid  ) const ; 
    void   get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy_eV ) const ; 
    const char* get_typename() const ; 
};

inline PMTAccessor::PMTAccessor(const PMTSimParamData* data_ )
    :
    data(data_),
    Pyrex(G4Material::GetMaterial("Pyrex")),
    Vacuum(G4Material::GetMaterial("Vacuum"))
{
}

inline std::string PMTAccessor::desc() const 
{
    std::stringstream ss ; 
    ss << "PMTAccessor::desc"
       << " data " << data 
       << " Pyrex " << ( Pyrex ? "YES" : "NO" ) 
       << " Vacuum " << ( Vacuum ? "YES" : "NO" )  
       << " TypeName " << get_typename()
       ; 
    std::string str = ss.str(); 
    return str ; 
}

inline double PMTAccessor::get_pmtid_qe( int pmtid, double energy ) const
{
    return data->get_pmtid_qe(pmtid, energy) ; 
}
inline int PMTAccessor::get_pmtcat( int pmtid  ) const
{
    return data->get_pmtcat(pmtid) ; 
}
inline void PMTAccessor::get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy_eV ) const
{
    // TODO: follow JPMT.h collecting the rindex, kindex, thickness into ss 

}
inline const char* PMTAccessor::get_typename() const 
{ 
    return TypeName ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef WITH_PMTFASTSIM
    LOG(info) << "PMTCategory::List() " << PMTCategory::List() ; 

    for(int i=kPMT_Unknown ; i <= kPMT_NNVT_HighQE ; i++)
    {
        std::cout << " PMTCategory::Name(" << std::setw(2) <<  i << ") " << PMTCategory::Name(i) << std::endl ; 
    }

    PMTSimParamData data ;
    const char* LOAD = getenv("PMTSimParamData_BASE") ; 
    //if(LOAD) _PMTSimParamData::Load(data, LOAD) ;  // TODO: convenience static 
    if(LOAD){ _PMTSimParamData _data(data) ; _data.load(LOAD); }
    
    LOG(info) << " data " << data ; 


    PMTAccessor acc(&data); 
    LOG(info) << " acc " << acc.desc() ; 


#else
    LOG(fatal) << "not WITH_PMTFASTSIM : nothing to do " ;     
#endif
    return 0 ; 
}

