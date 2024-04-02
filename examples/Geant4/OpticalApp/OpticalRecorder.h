#pragma once

#include "G4OpticalPhoton.hh"
#include "G4OpBoundaryProcess.hh"

struct OpticalRecorder
{
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

    template <typename T>
    static T* GetOpBoundaryProcess();

    static void GetPoint( double* p , const G4StepPoint* point); 
    static std::string Desc( const double* p, int num ); 
 
    void BeginOfRunAction(const G4Run* run);
    void EndOfRunAction(const G4Run* run);

    void BeginOfEventAction(const G4Event* evt);
    void EndOfEventAction(const G4Event* evt);

    void PreUserTrackingAction(const G4Track* trk);
    void PostUserTrackingAction(const G4Track* trk);

    void UserSteppingAction(const G4Step* step);

    OpticalRecorder(); 

    static constexpr const int MAX_POINT = 100 ; 
    double* pp ; 
    int num_point ; 

}; 


inline OpticalRecorder::OpticalRecorder()
    :
    pp(new double[MAX_POINT*16]),
    num_point(0)
{
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
void OpticalRecorder::GetPoint( double* p , const G4StepPoint* point)  // static
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

    p[12] = 0. ; 
    p[13] = 0. ; 
    p[14] = 0. ; 
    p[15] = 0. ; 
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
void OpticalRecorder::BeginOfEventAction(const G4Event* ){     std::cout << "OpticalRecorder::BeginOfEventAction\n"; } 
void OpticalRecorder::EndOfEventAction(const G4Event* ){       std::cout << "OpticalRecorder::EndOfEventAction\n" ; }
void OpticalRecorder::PreUserTrackingAction(const G4Track* ){  std::cout << "OpticalRecorder::PreUserTrackingAction\n" ;   }
void OpticalRecorder::PostUserTrackingAction(const G4Track* ){ std::cout << "OpticalRecorder::PostUserTrackingAction\n" ;  }
void OpticalRecorder::UserSteppingAction(const G4Step* step )
{    
    const G4Track* track = step->GetTrack();
    G4VPhysicalVolume* pv = track->GetVolume() ;

    const G4StepPoint* pre = step->GetPreStepPoint() ;
    const G4StepPoint* post = step->GetPostStepPoint() ;

    // U4StepPoint::ProcessDefinedStepType
    const G4VProcess* post_proc = post->GetProcessDefinedStep() ;
    const char* post_procName = post_proc ? post_proc->GetProcessName().c_str() : nullptr  ; 

    G4OpBoundaryProcessStatus bp_status = Undefined ; 
    if( post->GetStepStatus() == fGeomBoundary && strcmp(post_procName, "Transportation") == 0 )
    {
        G4OpBoundaryProcess* bp = GetOpBoundaryProcess<G4OpBoundaryProcess>(); 
        bp_status = bp ? bp->GetStatus() : Undefined ;
    }

    std::cout 
        << "OpticalRecorder::UserSteppingAction"
        << " pre_status " << StepStatus( pre->GetStepStatus())
        << " post_status " << StepStatus( post->GetStepStatus() )
        << " post_procName " << ( post_procName ? post_procName : "-" ) 
        << " bp_status " << OpBoundaryProcessStatus(bp_status)
        << " pv " << pv->GetName() 
        << "\n" 
        ; 


    double* p = num_point < MAX_POINT ? pp + 16*num_point : nullptr ; 

    if(p)
    {
        GetPoint( p, post ); 
        std::cout << Desc(p, 16 ) << "\n" ; 

        double* prior_p = num_point > 0 ? pp + 16*(num_point - 1 ) : nullptr ; 

        if(prior_p) 
        {
            double dt = p[3] - prior_p[3] ; 
            G4ThreeVector dp( p[0] - prior_p[0], p[1] - prior_p[1], p[2] - prior_p[2] ); 
            double dpm = dp.mag() ; 
            double dpm_dt = dt > 0 && dpm > 0 ? dpm/dt : 0 ; 

            std::cout << " dt " << dt << " dpm " << dpm << " dpm/dt " << dpm_dt << "\n" ; 
              
        }

   

    }

    num_point += 1 ; 


}

