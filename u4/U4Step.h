#pragma once

class G4Step ; 
class G4StepPoint ; 
class G4LogicalSurface ; 
class G4VPhysicalVolume ; 

struct sphoton ; 
struct SCF ; 


enum { 
   U4Step_UNSET, 
   U4Step_NOT_AT_BOUNDARY,
   U4Step_MOTHER_TO_CHILD, 
   U4Step_CHILD_TO_MOTHER,
   U4Step_CHILD_TO_CHILD,
   U4Step_UNEXPECTED
}; 

struct U4Step
{
    static const SCF* CF ; 

    static constexpr const char* UNSET = "UNSET" ; 
    static constexpr const char* NOT_AT_BOUNDARY = "NOT_AT_BOUNDARY" ; 
    static constexpr const char* MOTHER_TO_CHILD = "MOTHER_TO_CHILD" ; // AKA enteredDaughter
    static constexpr const char* CHILD_TO_MOTHER = "CHILD_TO_MOTHER" ; 
    static constexpr const char* CHILD_TO_CHILD  = "CHILD_TO_CHILD" ;  // ABOMINATION SUGGESTING BROKEN GEOMETRY 
    static constexpr const char* UNEXPECTED      = "UNEXPECTED" ; 

    static const char* MockOpticksBoundaryIdentity_NOTE ; 
    static void MockOpticksBoundaryIdentity(sphoton& current_photon,  const G4Step* step, unsigned idx); 

    static unsigned PackIdentity(unsigned prim_idx, unsigned instance_id); 
    static int KludgePrimIdx(const G4Step* step, unsigned type, unsigned idx); 

    static const char* Name(unsigned type); 
    static bool IsProblem(unsigned type);  
    static unsigned Classify(const G4Step* step); 
    static bool IsOnBoundary( const G4Step* step ); 
    static std::string BoundarySpec( const G4Step* step ); 
    static std::string BoundarySpec_(const G4Step* step ); 
    static const G4VSolid* Solid(const G4StepPoint* point ); 
    static G4LogicalSurface* GetLogicalSurface(const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV); 
    static std::string Spec(const G4Step* step); 
};


const SCF* U4Step::CF = SCF::Create() ; 


/**
U4Step::MockOpticksBoundaryIdentity
---------------------------------------

This only partially mimicks the Opticks identity, using the primname index as stand in for real prim_idx.
That should match Opticks only with simple geom without repeated prim or instances.

cx/CSGOptiX7.cu::

    406 extern "C" __global__ void __closesthit__ch()
    407 {
    408     unsigned iindex = optixGetInstanceIndex() ;    // 0-based index within IAS
    409     unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build and InstanceId.h 
    410     unsigned prim_idx = optixGetPrimitiveIndex() ; // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    411     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ;

TODO: find way to fully reproduce the Opticks identity with instance index, 
probably that would mean dealing with long lists of volume names : the difficulty
is the factorization which means multiple volumes are within each instance so would
have to list all volumes 

**/

const char* U4Step::MockOpticksBoundaryIdentity_NOTE = R"(
U4Step::MockOpticksBoundaryIdentity 
====================================

Mocking Opticks requires CFBASE envvar which allows instanciation of SCF
This means that when changing the U4VolumeMaker geometry it is necessary to
run the Opticks gxs.sh GPU simulation first and grab the CSGFoundry geometry 
back to the machine running the Geant4 simulation. 

eg::

    gx
    ./gxs.sh run    # workstation
    ./gxs.sh grab   # laptop 

    u4
    ./u4s.sh run     # CFBASE is set by the script to pick up the CF geometry

)"; 


void U4Step::MockOpticksBoundaryIdentity(sphoton& current_photon,  const G4Step* step, unsigned idx)  // static
{
    if(CF == nullptr) std::cerr << MockOpticksBoundaryIdentity_NOTE ; 
    assert(CF); 

    std::string spec = BoundarySpec(step) ; // empty when not boundary   
    unsigned boundary = spec.empty() ? 0 : CF->getBoundary(spec.c_str()) ; 
    unsigned type = U4Step::Classify(step); 
    int kludge_prim_idx = KludgePrimIdx(step, type, idx) ; 
    unsigned kludge_prim_idx_ = kludge_prim_idx == -1 ? 0xffff : kludge_prim_idx ; 

    float cosThetaSign = 0.f ; // aka orient 

    switch(type)
    {
        case U4Step_NOT_AT_BOUNDARY: cosThetaSign =  0.f ; break ;
        case U4Step_MOTHER_TO_CHILD: cosThetaSign = -1.f ; break ;    // photon direction against the normal 
        case U4Step_CHILD_TO_MOTHER: cosThetaSign =  1.f ; break ;    // photon direction with the normal 
    }

    /*
    if( U4Step::IsProblem(type) || type ==  U4Step_NOT_AT_BOUNDARY || kludge_prim_idx == -1  )
    {
        std::cerr
             << "U4Step::MockOpticksBoundaryIdentity"
             << " problem step "
             << " idx " << idx
             << " type " << type 
             << " U4Step::Name " << U4Step::Name(type)
             << " cosThetaSign " << cosThetaSign
             << " spec " << spec 
             << " boundary " << boundary 
             << " kludge_prim_idx " << kludge_prim_idx
             << " kludge_prim_idx_ " << kludge_prim_idx_
             << std::endl 
             ;  

        std::cerr << " pre  " << U4StepPoint::DescPositionTime(step->GetPreStepPoint()) << std::endl ; 
        std::cerr << " post " << U4StepPoint::DescPositionTime(step->GetPostStepPoint()) << std::endl ; 
    }
    */
 

    // HMM: what does Opticks do for not at boundary ? 
    current_photon.set_orient( cosThetaSign );   // sets a bit : would be 0 if not set
    current_photon.set_boundary( boundary);
    current_photon.identity = PackIdentity( kludge_prim_idx_, 0u) ;
}
unsigned U4Step::PackIdentity(unsigned prim_idx, unsigned instance_id) 
{
    unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ;
    return identity ; 
}

/**
U4Step::KludgePrimIdx
------------------------

::

     +---Rock--------------------------------+
     |                                       |
     |                                       |
     |    +--------------Air----------+      |
     |    |                           |      |
     |    |            Water          |      |
     |    |          /     \          |      |
     |    |  +----->|-    >|         >|      |
     |    |          \    /           |      |
     |    |           ---             |      |
     |    |                           |      |
     |    +---------------------------+      |
     |                                       |
     |                                       |
     +---------------------------------------+


Opticks provides the primIdx of the hit surface, without regard 
for the direction of the photon : because that is a characteristic
of the geometry that is essentially a label on the geometry.

* Opticks knows the boundary, and the orientation wrt the normal, 
  so it knows the material but not the next prim until it does 
  another intersection

* this difference arises from the boundary based Opticks geometry model 
  vs volume based Geant4 geometry model

The first attempt to mimic this with Geant4 always used the "post-solid" 
but that does not match Opticks when leaving solids. To mimic Opticks it is 
necessary to use pre-solid or the post-solid depending on the orientation. 

**/


int U4Step::KludgePrimIdx(const G4Step* step, unsigned type, unsigned idx)
{

    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 
 
    const G4VSolid* pre_so = U4Step::Solid(pre) ;  
    const G4VSolid* post_so = U4Step::Solid(post) ;
    G4String pre_so_name = pre_so ? pre_so->GetName() : "" ; 
    G4String post_so_name = post_so ? post_so->GetName() : "" ;

    if( pre_so_name.empty() || post_so_name.empty() ) 
         std::cerr 
             << "U4Step::KludgePrimIdx EMPTY NAME " 
             << " pre_so_name " << pre_so_name
             << " post_so_name " << post_so_name
             << std::endl
             ;

 
    int pre_prim_idx = CF->getPrimIdx(pre_so_name.c_str()) ; 
    int post_prim_idx = CF->getPrimIdx(post_so_name.c_str()) ; 

    int kludge_prim_idx = 0 ;    // this will only match Opticks for very simple geometries 
    switch(type)
    {
        case U4Step_MOTHER_TO_CHILD: kludge_prim_idx = post_prim_idx ; break ; 
        case U4Step_CHILD_TO_MOTHER: kludge_prim_idx = pre_prim_idx  ; break ; 
        default:                     kludge_prim_idx = 0             ; break ;      
    }

    /*
    std::cout 
        << " U4Step::Name " << U4Step::Name(type)
        << " pre_so_name " << std::setw(20) << pre_so_name 
        << " pre_prim_idx " << std::setw(4) << pre_prim_idx 
        << " post_so_name " << std::setw(20) << post_so_name 
        << " post_prim_idx " << std::setw(4) << post_prim_idx 
        << " kludge_prim_idx " << std::setw(4) << kludge_prim_idx
        << std::endl 
        ; 
    */ 

    return kludge_prim_idx ; 
}







const char* U4Step::Name(unsigned type)
{
    const char* s = nullptr ; 
    switch(type)
    {
       case U4Step_UNSET:           s = UNSET           ; break ; 
       case U4Step_NOT_AT_BOUNDARY: s = NOT_AT_BOUNDARY ; break ; 
       case U4Step_MOTHER_TO_CHILD: s = MOTHER_TO_CHILD ; break ; 
       case U4Step_CHILD_TO_MOTHER: s = CHILD_TO_MOTHER ; break ; 
       case U4Step_CHILD_TO_CHILD:  s = CHILD_TO_CHILD  ; break ; 
       case U4Step_UNEXPECTED:      s = UNEXPECTED      ; break ; 
    }
    return s ; 
} 

bool U4Step::IsProblem(unsigned type)
{
    return type == U4Step_UNSET || type == U4Step_CHILD_TO_CHILD || type == U4Step_UNEXPECTED ; 
}



/**
U4Step::Classify
-----------------------------

::

      +------------------------------------------+
      | thePrePV                                 |
      |                                          |
      |                                          |
      |       +------------------------+         |
      |       | thePostPV              |         |
      |       |                        |         |
      |   +--->                        |         |
      |       |                        |         |
      |       |                        |         |
      |       +------------------------+         |
      |                                          |
      |                                          |
      |                                          |
      +------------------------------------------+


      enteredDaughter = thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume()   "MOTHER_TO_CHILD"

      enteredDaughter:True 
          "inwards" photons


      +------------------------------------------+
      | thePostPV                                |
      |                                          |
      |                                          |
      |       +------------------------+         |
      |       | thePrePV               |         |
      |       |                        |         |
      |       <---+                    |         |
      |       |                        |         |
      |       |                        |         |
      |       +------------------------+         |
      |                                          |
      |                                          |
      |                                          |
      +------------------------------------------+


      enteredDaughter = thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume() 

      enteredDaughter:False
          "outwards" photons

**/


unsigned U4Step::Classify(const G4Step* step)
{
    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 
    bool  not_post_boundary = post->GetStepStatus() != fGeomBoundary ;
    if(not_post_boundary) return U4Step_NOT_AT_BOUNDARY ; 

    const G4VPhysicalVolume* thePrePV = pre->GetPhysicalVolume();
    const G4VPhysicalVolume* thePostPV = post->GetPhysicalVolume();

    bool pre_is_post_mother = thePrePV->GetLogicalVolume() == thePostPV->GetMotherLogical() ;   // aka enteredDaughter
    bool pre_mother_is_post = thePrePV->GetMotherLogical() == thePostPV->GetLogicalVolume() ; 
    bool pre_mother_is_post_mother = thePrePV->GetMotherLogical() == thePostPV->GetMotherLogical() ; 

    bool expect_exclusive = int(pre_is_post_mother) + int(pre_mother_is_post) + int(pre_mother_is_post_mother) <= 1 ; 
    assert( expect_exclusive ) ;   // otherwise broken geometry ? 

    unsigned type = U4Step_UNSET ; 
    if(      pre_is_post_mother )        type = U4Step_MOTHER_TO_CHILD ; 
    else if( pre_mother_is_post )        type = U4Step_CHILD_TO_MOTHER ; 
    else if( pre_mother_is_post_mother ) type = U4Step_CHILD_TO_CHILD ;   // abomination suggesting broken geometry 
    else                                 type = U4Step_UNEXPECTED ; 

    return type ; 
}


bool U4Step::IsOnBoundary( const G4Step* step ) // static 
{
    const G4StepPoint* post = step->GetPostStepPoint() ; 
    G4bool isOnBoundary = post->GetStepStatus() == fGeomBoundary ;
    return isOnBoundary ; 
}



/**
U4Step::BoundarySpec_
------------------------------

Spec is a string composed of 4 elements delimted by 3 "/"::

    omat/osur/isur/imat

The osur and isur can be blank, the omat and imat cannot be blank


enteredDaughter
    True:  "inwards" photons 
    False: "outwards" photons 
 
**/

std::string U4Step::BoundarySpec(const G4Step* step ) // static
{
    return IsOnBoundary(step) ? BoundarySpec_(step) : "" ;  
}
std::string U4Step::BoundarySpec_(const G4Step* step) // static 
{
    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 
    const G4Material* m1 = pre->GetMaterial();
    const G4Material* m2 = post->GetMaterial();
    const char* n1 = m1->GetName().c_str() ;  
    const char* n2 = m2->GetName().c_str() ;  

    const G4VPhysicalVolume* thePrePV = pre->GetPhysicalVolume();
    const G4VPhysicalVolume* thePostPV = post->GetPhysicalVolume();
    G4bool enteredDaughter = thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume();

    const char* omat = enteredDaughter ? n1 : n2 ; 
    const char* imat = enteredDaughter ? n2 : n1 ; 

    const G4LogicalSurface* surf1 = GetLogicalSurface(thePrePV, thePostPV ); 
    const G4LogicalSurface* surf2 = GetLogicalSurface(thePostPV, thePrePV ); 

    const char* osur = nullptr ; 
    const char* isur = nullptr ; 

    if( enteredDaughter )
    {
        osur = surf1 ? surf1->GetName().c_str() : nullptr ; 
        isur = surf2 ? surf2->GetName().c_str() : nullptr ; 
    }
    else
    {
        osur = surf2 ? surf2->GetName().c_str() : nullptr ; 
        isur = surf1 ? surf1->GetName().c_str() : nullptr ; 
    }

    std::stringstream ss ; 
    ss 
       << omat 
       << "/" 
       << ( osur ? osur : "" ) 
       << "/" 
       << ( isur ? isur : "" ) 
       << "/" 
       << imat
       ; 

    std::string s = ss.str(); 
    return s ; 
}


inline G4LogicalSurface* U4Step::GetLogicalSurface(const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV)
{
    G4LogicalSurface* Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
    if (Surface == nullptr)
    {
        G4bool enteredDaughter = thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume();

        if(enteredDaughter)
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
            if(Surface == nullptr) Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
        }    
        else  // "leavingDaughter"
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
            if(Surface == NULL) Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
        }
    }
    return Surface ; 
}



const G4VSolid* U4Step::Solid(const G4StepPoint* point ) // static
{
    const G4VPhysicalVolume* pv  = point->GetPhysicalVolume();
    const G4LogicalVolume* lv = pv ? pv->GetLogicalVolume() : nullptr ;
    const G4VSolid* so = lv ? lv->GetSolid() : nullptr ; 
    return so ; 
}

/**
U4Step::Spec
-------------

Used for fake skipping from U4Recorder::UserSteppingAction_Optical using U4Recorder::IsFake
configured via envvars::

    U4Recorder__FAKES
    U4Recorder__FAKES_SKIP

**/

std::string U4Step::Spec(const G4Step* step) // static
{
    const G4Track* track = step->GetTrack(); 

    G4VPhysicalVolume* pv = track->GetVolume() ; 
    G4VPhysicalVolume* next_pv = track->GetNextVolume() ; 

    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    const G4VPhysicalVolume* pre_pv = pre->GetPhysicalVolume();
    const G4VPhysicalVolume* post_pv = post->GetPhysicalVolume();

    assert( pv == pre_pv );  
    assert( next_pv == post_pv );  


    const G4Material* pre_mat = pre->GetMaterial(); 
    const G4Material* post_mat = post->GetMaterial(); 

    std::stringstream ss ; 
    ss 
       << ( pre_mat ? pre_mat->GetName() : "-" ) 
       << "/" 
       << ( post_mat ? post_mat->GetName() : "-" ) 
       << ":"
       << ( pv ? pv->GetName() : "-" ) 
       << "/" 
       << ( next_pv ? next_pv->GetName() : "-" ) 
       ; 

    std::string str = ss.str() ; 
    return str ; 
}






