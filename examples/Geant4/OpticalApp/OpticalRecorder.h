#pragma once

#include "G4OpticalPhoton.hh"
#include "G4OpBoundaryProcess.hh"
#include "np.h"

#define FFS(x)   (ffs(x))

// OpticksPhoton.h 
enum
{
    CERENKOV          = 0x1 <<  0,    
    SCINTILLATION     = 0x1 <<  1,    
    MISS              = 0x1 <<  2,  
    BULK_ABSORB       = 0x1 <<  3,  
    BULK_REEMIT       = 0x1 <<  4,  
    BULK_SCATTER      = 0x1 <<  5,  
    SURFACE_DETECT    = 0x1 <<  6,  
    SURFACE_ABSORB    = 0x1 <<  7,  
    SURFACE_DREFLECT  = 0x1 <<  8,  
    SURFACE_SREFLECT  = 0x1 <<  9,  
    BOUNDARY_REFLECT  = 0x1 << 10, 
    BOUNDARY_TRANSMIT = 0x1 << 11, 
    TORCH             = 0x1 << 12, 
    NAN_ABORT         = 0x1 << 13, 
    EFFICIENCY_CULL    = 0x1 << 14, 
    EFFICIENCY_COLLECT = 0x1 << 15, 
    __NATURAL         = 0x1 << 16, 
    __MACHINERY       = 0x1 << 17, 
    __EMITSOURCE      = 0x1 << 18, 
    PRIMARYSOURCE     = 0x1 << 19, 
    GENSTEPSOURCE     = 0x1 << 20, 
    DEFER_FSTRACKINFO = 0x1 << 21
}; 



struct OpticalRecorder
{
    static constexpr const unsigned BITS = 4 ;
    static constexpr const uint64_t MASK = ( 0x1ull << BITS ) - 1ull ;

    static constexpr const char* fGeomBoundary_  = "fGeomBoundary" ;
    static constexpr const char* fUndefined_ = "fUndefined" ;
    static constexpr const char* fERROR_ = "fERROR" ;
    static const char* StepStatus(unsigned status) ; 

    static constexpr const char* Undefined_ = "Undefined" ;
    static constexpr const char* Transmission_ = "Transmission" ;
    static constexpr const char* FresnelRefraction_ = "FresnelRefraction" ;
    static constexpr const char* FresnelReflection_  = "FresnelReflection" ;
    static constexpr const char* TotalInternalReflection_ = "TotalInternalReflection" ;
    static constexpr const char* LambertianReflection_ = "LambertianReflection" ;
    static constexpr const char* LobeReflection_ = "LobeReflection" ;
    static constexpr const char* SpikeReflection_ = "SpikeReflection" ;
    static constexpr const char* BackScattering_ = "BackScattering" ;
    static constexpr const char* Absorption_ = "Absorption" ;
    static constexpr const char* Detection_ = "Detection" ;
    static constexpr const char* NotAtBoundary_ = "NotAtBoundary" ;
    static constexpr const char* SameMaterial_ = "SameMaterial" ;
    static constexpr const char* StepTooSmall_ = "StepTooSmall" ;
    static constexpr const char* NoRINDEX_ = "NoRINDEX" ;
    static constexpr const char* Other_ = "Other" ;   
    static const char* OpBoundaryProcessStatus(unsigned status); 


    static constexpr const char* _ZERO              = "  " ;
    static constexpr const char* _CERENKOV          = "CK" ;
    static constexpr const char* _SCINTILLATION     = "SI" ; 
    static constexpr const char* _TORCH             = "TO" ; 
    static constexpr const char* _MISS              = "MI" ;
    static constexpr const char* _BULK_ABSORB       = "AB" ;
    static constexpr const char* _BULK_REEMIT       = "RE" ;
    static constexpr const char* _BULK_SCATTER      = "SC" ;
    static constexpr const char* _SURFACE_DETECT    = "SD" ;
    static constexpr const char* _SURFACE_ABSORB    = "SA" ;
    static constexpr const char* _SURFACE_DREFLECT  = "DR" ;
    static constexpr const char* _SURFACE_SREFLECT  = "SR" ;
    static constexpr const char* _BOUNDARY_REFLECT  = "BR" ;
    static constexpr const char* _BOUNDARY_TRANSMIT = "BT" ;
    static constexpr const char* _NAN_ABORT         = "NA" ;
    static constexpr const char* _EFFICIENCY_COLLECT = "EC" ;
    static constexpr const char* _EFFICIENCY_CULL    = "EX" ;
    static constexpr const char* _BAD_FLAG           = "XX" ;
    static const char* Flag(unsigned flag); 


    template <typename T>
    static T* GetOpBoundaryProcess();

    void writePoint( const G4StepPoint* point, unsigned flag ); 
    static void WritePoint( double* p , const G4StepPoint* point, unsigned flag ); 
    static std::string Desc( const double* p, int num ); 
 
    void BeginOfRunAction(const G4Run* run);
    void EndOfRunAction(const G4Run* run);

    void BeginOfEventAction(const G4Event* evt);
    void EndOfEventAction(const G4Event* evt);

    void PreUserTrackingAction(const G4Track* trk);
    void PostUserTrackingAction(const G4Track* trk);
    static int TrackIdx( const G4Track* trk ); 

    static unsigned PointFlag( const G4StepPoint* point ); 
    static unsigned BoundaryFlag(unsigned status);  // BT BR NA SA SD SR DR

    void UserSteppingAction(const G4Step* step);


    static bool Valid(int trk_idx, int point_idx);
    double* getRecord(int _point_idx) const ;
    void recordPoint( const G4StepPoint* point ); 

    static double DeltaTime( const double* a, const double* b );
    static double DeltaPos( const double* a, const double* b );
    std::string descPoint(int _point_idx) const ; 

    static constexpr const int MAX_PHOTON = 100000 ; 
    static constexpr const int MAX_POINT  = 10 ; 
    static constexpr const int MAX_VALUE  = MAX_PHOTON*MAX_POINT*16  ; 
  
    OpticalRecorder(); 
    void clear(); 
    void alloc(); 

    int trk_idx = 0  ; 
    int point_idx = 0  ; 

    double* pp = nullptr ; 
    uint64_t* qq = nullptr ; 
    std::string desc ; 

}; 

inline OpticalRecorder::OpticalRecorder(){}

inline void OpticalRecorder::clear()
{
    delete [] pp ; 
    pp = nullptr ; 

    delete [] qq ; 
    qq = nullptr ; 
}
inline void OpticalRecorder::alloc()
{
    pp = new double[MAX_VALUE] ; 
    for(int i=0 ; i < MAX_VALUE ; i++) pp[i] = 0. ; 

    qq = new uint64_t[MAX_PHOTON*2*2] ; 
    for(int i=0 ; i < MAX_PHOTON*2*2 ; i++) qq[i] = 0ull ; 
}


// U4StepStatus::Name
inline const char* OpticalRecorder::StepStatus(unsigned status)
{
    const char* str = nullptr ; 
    switch(status)
    {   
        case fGeomBoundary:           str=fGeomBoundary_           ;break; 
        case fUndefined:              str=fUndefined_              ;break; 
        default:                      str=fERROR_                  ;break;
    }   
    return str ; 
}

// U4OpBoundaryProcessStatus::Name
inline const char* OpticalRecorder::OpBoundaryProcessStatus(unsigned status)
{
    const char* str = nullptr ; 
    switch(status)
    {   
       case Undefined:                str = Undefined_ ; break ; 
       case Transmission:             str = Transmission_ ; break ; 
       case FresnelRefraction:        str = FresnelRefraction_ ; break ; 
       case FresnelReflection:        str = FresnelReflection_ ; break ;
       case TotalInternalReflection:  str = TotalInternalReflection_ ; break ;
       case LambertianReflection:     str = LambertianReflection_ ; break ;
       case LobeReflection:           str = LobeReflection_ ; break ;
       case SpikeReflection:          str = SpikeReflection_ ; break ;
       case BackScattering:           str = BackScattering_ ; break ;
       case Absorption:               str = Absorption_ ; break ;
       case Detection:                str = Detection_ ; break ;
       case NotAtBoundary:            str = NotAtBoundary_ ; break ;
       case SameMaterial:             str = SameMaterial_ ; break ;
       case StepTooSmall:             str = StepTooSmall_ ; break ;
       case NoRINDEX:                 str = NoRINDEX_ ; break ;
       default:                       str = Other_ ; break ;
    }
    return str ;
}

// OpticksPhoton::Flag
const char* OpticalRecorder::Flag(const unsigned int flag)
{
    const char* str = 0 ;
    switch(flag)
    {
        case 0:                str=_ZERO;break;
        case CERENKOV:         str=_CERENKOV ;break;
        case SCINTILLATION:    str=_SCINTILLATION ;break;
        case MISS:             str=_MISS ;break;
        case BULK_ABSORB:      str=_BULK_ABSORB ;break;
        case BULK_REEMIT:      str=_BULK_REEMIT ;break;
        case BULK_SCATTER:     str=_BULK_SCATTER ;break;
        case SURFACE_DETECT:   str=_SURFACE_DETECT ;break;
        case SURFACE_ABSORB:   str=_SURFACE_ABSORB ;break;
        case SURFACE_DREFLECT: str=_SURFACE_DREFLECT ;break;
        case SURFACE_SREFLECT: str=_SURFACE_SREFLECT ;break;
        case BOUNDARY_REFLECT: str=_BOUNDARY_REFLECT ;break;
        case BOUNDARY_TRANSMIT:str=_BOUNDARY_TRANSMIT ;break;
        case TORCH:            str=_TORCH ;break;
        case NAN_ABORT:        str=_NAN_ABORT ;break;
        case EFFICIENCY_CULL:    str=_EFFICIENCY_CULL ;break;
        case EFFICIENCY_COLLECT: str=_EFFICIENCY_COLLECT ;break;
        default:                 str=_BAD_FLAG  ;
    }
    return str;
}




// U4OpBoundaryProcess::Get
template <typename T>
inline T* OpticalRecorder::GetOpBoundaryProcess()
{           
    T* bp = nullptr ;
    G4ProcessManager* mgr = G4OpticalPhoton::OpticalPhoton()->GetProcessManager() ;
    assert(mgr); 

    G4int pmax = mgr ? mgr->GetPostStepProcessVector()->entries() : 0 ;
    G4ProcessVector* pvec = mgr ? mgr->GetPostStepProcessVector(typeDoIt) : nullptr ;

    for (int i=0; i < pmax ; i++)
    {
        G4VProcess* p = (*pvec)[i];
        T* t = dynamic_cast<T*>(p);
        if(t)
        {
            bp = t ;
            break;
        }
    }
    return bp ;
}

// U4StepPoint::Update
void OpticalRecorder::writePoint( const G4StepPoint* point, unsigned flag )
{
    double* p = getRecord(point_idx); 
    if(!p) return ; 

    WritePoint(p, point, flag ); 

    uint64_t& q0 = qq[4*trk_idx+0] ; 

    // sseq::add_nibble
    unsigned shift = 4*point_idx ; 
    q0 |= (( FFS(flag) & MASK ) << shift );  
    
}


void OpticalRecorder::WritePoint( double* p, const G4StepPoint* point, unsigned flag )
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& mom = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = CLHEP::h_Planck*CLHEP::c_light/energy ;
    
    p[0] = pos.x(); 
    p[1] = pos.y(); 
    p[2] = pos.z(); 
    p[3] = time/ns ; 

    p[4] = mom.x(); 
    p[5] = mom.y(); 
    p[6] = mom.z(); 
    p[7] = 0. ; 

    p[8] = pol.x();
    p[9] = pol.y();
    p[10] = pol.z();
    p[11] = wavelength/nm ;


    union uif64_t {
        uint64_t  u ; 
        int64_t   i ; 
        double    f ; 
    };  

    uif64_t uif ; 
    uif.u = flag ;  

    p[12] = 0. ; 
    p[13] = 0. ; 
    p[14] = 0. ; 
    p[15] = uif.f ; 
}


double OpticalRecorder::DeltaTime( const double* a, const double* b )
{
    return a && b ? b[3] - a[3] : -1 ; 
}
double OpticalRecorder::DeltaPos( const double* a, const double* b )
{
    if( a == nullptr || b == nullptr ) return -1 ; 
    G4ThreeVector dpos( b[0] - a[0], b[1] - a[1], b[2] - a[2] ); 
    return dpos.mag() ; 
}




std::string OpticalRecorder::Desc( const double* p, int num )
{
    std::stringstream ss ; 
    assert( num == 16 ); 
    for(int i=0 ; i < num ; i++) 
        ss  
            << ( i % 4 == 0 && num > 4 ? ".\n" : "" ) 
            << " " << std::fixed << std::setw(10) << std::setprecision(4) << p[i] 
            << ( i == num-1 && num > 4 ? ".\n" : "" ) 
            ;   

    std::string str = ss.str(); 
    return str ; 
}



void OpticalRecorder::BeginOfRunAction(const G4Run* ){         std::cout << "OpticalRecorder::BeginOfRunAction\n" ;    }
void OpticalRecorder::EndOfRunAction(const G4Run* ){           std::cout << "OpticalRecorder::EndOfRunAction\n" ;  }


void OpticalRecorder::BeginOfEventAction(const G4Event* evt)
{  
    std::cout << "OpticalRecorder::BeginOfEventAction evt " << evt->GetEventID() << "\n" ;
    alloc();  
} 

void OpticalRecorder::EndOfEventAction(const G4Event* evt)
{    
    int eid = evt->GetEventID() ; 
    std::cout << "OpticalRecorder::EndOfEventAction eid " << eid << "\n" ; 

    const char* FOLD = getenv("FOLD") ; 

    std::vector<int> pp_shape = { MAX_PHOTON, MAX_POINT, 4, 4 } ; 
    np::Write( FOLD, "record.npy",  pp_shape, pp, "<f8" ); 

    std::vector<int> qq_shape = { MAX_PHOTON, 2, 2 } ; 
    np::Write( FOLD, "seq.npy",  qq_shape, qq, "<u8" ); 

    if(!desc.empty()) np::WriteString( FOLD, "NPFold_meta.txt", desc.c_str() ); 

    clear(); 
}
void OpticalRecorder::PreUserTrackingAction(const G4Track* trk )
{ 
    trk_idx = TrackIdx(trk) ;
    point_idx = 0 ;
 
    if(0) std::cout 
        << "OpticalRecorder::PreUserTrackingAction"
        << " trk_idx :" << trk_idx 
        << " point_idx:" << point_idx 
        << "\n" 
        ;  
}
void OpticalRecorder::PostUserTrackingAction(const G4Track* trk)
{
    assert( TrackIdx(trk) == trk_idx ); 
 
    if(0) std::cout 
        << "OpticalRecorder::PostUserTrackingAction"
        << " trk_idx :" << trk_idx 
        << " point_idx:" << point_idx 
        << "\n" 
        ; 
    
}

int OpticalRecorder::TrackIdx( const G4Track* track )
{
   return track->GetTrackID() - 1 ;  // 0-based unlike 1-based TrackID  
}

// U4StepPoint::Flag
unsigned OpticalRecorder::PointFlag( const G4StepPoint* point )
{
    G4StepStatus status = point->GetStepStatus()  ;

    // U4StepPoint::ProcessDefinedStepType
    const G4VProcess* proc = point->GetProcessDefinedStep() ;
    const char* procName = proc ? proc->GetProcessName().c_str() : nullptr  ; 

    unsigned flag = 0 ; 

    if( status == fPostStepDoItProc && strcmp(procName, "OpAbsorption") == 0 )
    {
        flag = BULK_ABSORB ;
    }
    else if( status == fPostStepDoItProc && strcmp(procName, "OpRayleigh") == 0 )
    {
        flag = BULK_SCATTER ;
    }    
    else if( status == fGeomBoundary && strcmp(procName, "Transportation") == 0  )
    {
        G4OpBoundaryProcess* bp = GetOpBoundaryProcess<G4OpBoundaryProcess>(); 
        G4OpBoundaryProcessStatus bp_status = bp ? bp->GetStatus() : Undefined ;
        flag = BoundaryFlag( bp_status ); 
    }
    else if( status == fWorldBoundary && strcmp(procName, "Transportation") == 0  )
    {
        flag = MISS ;  
    }
    return flag  ; 
}

// U4StepPoint::BoundaryFlag
unsigned OpticalRecorder::BoundaryFlag(unsigned status) // BT BR NA SA SD SR DR 
{
    unsigned flag = 0 ; 
    switch(status)
    {   
        case FresnelRefraction:
        case SameMaterial:
        case Transmission:
                               flag=BOUNDARY_TRANSMIT;
                               break;
        case TotalInternalReflection:
        case       FresnelReflection:
                               flag=BOUNDARY_REFLECT;
                               break;
        case StepTooSmall:
                               flag=NAN_ABORT;
                               break;
        case Absorption:
                               flag=SURFACE_ABSORB ; 
                               break;
        case Detection:
                               flag=SURFACE_DETECT ; 
                               break;
        case SpikeReflection:
                               flag=SURFACE_SREFLECT ; 
                               break;
        case LobeReflection:
        case LambertianReflection:
                               flag=SURFACE_DREFLECT ; 
                               break;
        case NoRINDEX:
                               flag=SURFACE_ABSORB ;
                               //flag=NAN_ABORT;
                               break;
        default:
                               flag = 0 ; 
                               break;
    }
    return flag ; 
}


void OpticalRecorder::UserSteppingAction(const G4Step* step )
{    
    const G4Track* track = step->GetTrack();
    assert( trk_idx == TrackIdx(track) ); 

    const G4StepPoint* pre = step->GetPreStepPoint() ;
    const G4StepPoint* post = step->GetPostStepPoint() ;

    if(point_idx == 0) recordPoint(pre) ; 
    recordPoint(post); 
}

bool OpticalRecorder::Valid(int _trk_idx, int _point_idx)
{
    return 
          _trk_idx > -1 && _trk_idx < MAX_PHOTON 
          && 
          _point_idx > -1 && _point_idx < MAX_POINT 
          ;  
}

double* OpticalRecorder::getRecord(int _point_idx) const
{
    double* rec = Valid(trk_idx, _point_idx) ? pp + 16*MAX_POINT*trk_idx + 16*_point_idx  : nullptr ; 
    return rec ;     
}

void OpticalRecorder::recordPoint( const G4StepPoint* point )
{
    unsigned flag = point_idx == 0 ? TORCH : PointFlag(point) ;  

    if( flag == NAN_ABORT ) return ; 
 
    writePoint( point, flag ); 
     

    if( trk_idx < 5 ) std::cout 
        << "OpticalRecorder::recordPoint" 
        << " trk_idx " << trk_idx 
        << " point_idx " << point_idx 
        << " flag [" << Flag(flag) << "]"  
        << "\n"
        << descPoint( point_idx )
        << "\n"
        ;  

    point_idx += 1 ; 
}


std::string OpticalRecorder::descPoint(int _point_idx) const
{
    double* curr = getRecord( _point_idx ); 
    double* prev = getRecord( _point_idx - 1 ); 

    double dt  = DeltaTime( prev, curr ); 
    double dp  = DeltaPos(  prev, curr ); 
    double speed = dt > 0. && dp > 0. ? dp/dt : -1. ; 

    std::stringstream ss ; 
    ss << Desc( curr, 16 ) << "\n" << " dt " << dt << " dp " << dp << " dp/dt " << speed << "\n" ;  
    std::string str = ss.str(); 
    return str ; 
}

