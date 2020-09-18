Hi Sam, 

> The geometry which I'm using uses a groundfrontpainted finish in some places. 
> This would be m_finish = 4 see 
>    https://bitbucket.org/simoncblyth/opticks/src/00080a4771d0f3afeecf06d9c2d1c4f78b9d7077/ggeo/GOpticalSurface.cc#lines-211 .

> Do you know what the implications of including this type of surface would be in
> the rest of opticks or is it a safe change?
> For now to get further I have changed the surfaces to be polished (0) in my geometry where needed.


To understand how Geant4 handles different finish see the Geant4 source::

    g4-
    g4-cls G4OpBoundaryProcess
    g4-cls G4RandomTools
    g4-cls G4RandomDirection

G4OpBoundaryProcess.cc::

    509              else {
    510                 if ( theFinish == polishedfrontpainted ) {
    511                    DoReflection();
    512                 }
    513                 else if ( theFinish == groundfrontpainted ) {
    514                    theStatus = LambertianReflection;
    515                    DoReflection();
    516                 }
    517                 else {
    518                    DielectricDielectric();
    519                 }
    520              }

G4OpBoundaryProcess.hh::

    344 inline
    345 void G4OpBoundaryProcess::DoReflection()
    346 {
    347         if ( theStatus == LambertianReflection ) {
    348 
    349           NewMomentum = G4LambertianRand(theGlobalNormal);
    350           theFacetNormal = (NewMomentum - OldMomentum).unit();
    351 
    352         }
    353         else if ( theFinish == ground ) {
    ...
    372         }
    373         G4double EdotN = OldPolarization * theFacetNormal;
    374         NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
    375 }


Opticks doesnt yet translate this surface finish, but it looks 
(at a quick glance anyhow) to be straightforward to do so.  
The natural place would be: 

    optixrap/cu/propagate.h:propagate_at_boundary_geant4_style 

The self contained nature of modifying the normal "G4LambertianRand"
is easy to translate as an almost direct translation of the Geant4 code can be done.

The most difficult aspect is getting the finish "label" onto the geometry via the 
boundary GPU texture as this requires "threading" the information through quite a
bit of code.


Simon



