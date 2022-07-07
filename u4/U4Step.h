#pragma once

class G4Step ; 
class G4StepPoint ; 
class G4LogicalSurface ; 
class G4VPhysicalVolume ; 


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
    static constexpr const char* UNSET = "UNSET" ; 
    static constexpr const char* NOT_AT_BOUNDARY = "NOT_AT_BOUNDARY" ; 
    static constexpr const char* MOTHER_TO_CHILD = "MOTHER_TO_CHILD" ; // AKA enteredDaughter
    static constexpr const char* CHILD_TO_MOTHER = "CHILD_TO_MOTHER" ; 
    static constexpr const char* CHILD_TO_CHILD  = "CHILD_TO_CHILD" ;  // ABOMINATION SUGGESTING BROKEN GEOMETRY 
    static constexpr const char* UNEXPECTED      = "UNEXPECTED" ; 
    static const char* Name(unsigned type); 
    static bool IsProblem(unsigned type);  
    static unsigned Classify(const G4Step* step); 
    static bool IsOnBoundary( const G4Step* step ); 
    static std::string BoundarySpec( const G4Step* step ); 
    static std::string BoundarySpec_(const G4Step* step ); 
    static const G4VSolid* Solid(const G4StepPoint* point ); 
    static G4LogicalSurface* GetLogicalSurface(const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV); 
};

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
    const G4LogicalVolume* lv = pv->GetLogicalVolume();
    const G4VSolid* so = lv->GetSolid(); 
    return so ; 

}


