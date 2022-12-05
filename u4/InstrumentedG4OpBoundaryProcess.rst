InstrumentedG4OpBoundaryProcess
===================================


capture theGlobalExitNormal before any flips
------------------------------------------------

::

     434     G4int hNavId = G4ParallelWorldProcess::GetHypNavigatorID();
     435     std::vector<G4Navigator*>::iterator iNav =
     436                 G4TransportationManager::GetTransportationManager()->
     437                                          GetActiveNavigatorsIterator();
     438     theGlobalNormal =
     439                    (iNav[hNavId])->GetGlobalExitNormal(theGlobalPoint,&valid);
     440 
     441     theGlobalExitNormal = theGlobalNormal ;  // BEFORE ANY FLIPS 


Making sense of the normals
------------------------------


CustomART::

    1223     //dbg.nrm = oriented_normal ;  
    1224     //dbg.nrm = surface_normal ;     // verified that surface_normal always outwards
    1225     dbg.nrm = theGlobalExitNormal ;  // inwards first, the rest outwards : oriented into direction of incident photon
    1226     //dbg.nrm = theGlobalNormal ;    // this has been oriented : outwards first, the rest inwards  


Even theGlobalExitNormal has been flipped by the G4Navigator to point into the 2nd volume

This means::

     theGlobalExitNormal*OldMomentum always +ve 


Is it possible to reliably recover the unflipped global normal somehow ? 
----------------------------------------------------------------------------





theGlobalNormal and flips
----------------------------

::

     367 G4VParticleChange* InstrumentedG4OpBoundaryProcess::PostStepDoIt_(const G4Track& aTrack, const G4Step& aStep)
     368 {
     ...
     426     //[theGlobalNorml
     427     theGlobalPoint = pStep->GetPostStepPoint()->GetPosition();
     428 
     429     G4bool valid;
     430     // Use the new method for Exit Normal in global coordinates,
     431     // which provides the normal more reliably.
     432     // ID of Navigator which limits step
     433 
     434     G4int hNavId = G4ParallelWorldProcess::GetHypNavigatorID();
     435     std::vector<G4Navigator*>::iterator iNav =
     436                 G4TransportationManager::GetTransportationManager()->
     437                                          GetActiveNavigatorsIterator();
     438     theGlobalNormal =
     439                    (iNav[hNavId])->GetGlobalExitNormal(theGlobalPoint,&valid);
     440 #ifdef DEBUG_PIDX
     441     {
     442         quad2& prd = SEvt::Get()->current_prd ;
     443         prd.q0.f.x = theGlobalNormal.x() ;
     444         prd.q0.f.y = theGlobalNormal.y() ;
     445         prd.q0.f.z = theGlobalNormal.z() ;
     446 
     447         // TRY USING PRE->POST POSITION CHANGE TO GET THE PRD DISTANCE ? 
     448         G4ThreeVector theGlobalPoint_pre = pStep->GetPreStepPoint()->GetPosition();
     449         G4ThreeVector theGlobalPoint_delta = theGlobalPoint - theGlobalPoint_pre  ;
     450         prd.q0.f.w = theGlobalPoint_delta.mag() ;
     451 
     452         // HMM: PRD intersect identity ? how to mimic what Opticks does ? 
     453     }
     454 #endif
     455 
     456     if (valid)
     457     {
     458         theGlobalNormal = -theGlobalNormal;
     459     }



G4Navigator not providing the simple consistent Geometry Normal Needed for CustomBoundaryART
----------------------------------------------------------------------------------------------


g4-cls G4Navigator::

    227   virtual G4ThreeVector GetGlobalExitNormal(const G4ThreeVector& point,
    228                                                   G4bool* valid);
    229     // Return Exit Surface Normal and validity too.
    230     // Can only be called if the Navigator's last Step has crossed a
    231     // volume geometrical boundary.
    232     // It returns the Normal to the surface pointing out of the volume that
    233     // was left behind and/or into the volume that was entered.
    234     // Convention:
    235     //   The *local* normal is in the coordinate system of the *final* volume.
    236     // Restriction:
    237     //   Normals are not available for replica volumes (returns valid= false)
    238     // These methods takes full care about how to calculate this normal,
    239     // but if the surfaces are not convex it will return valid=false.
    240 


    1647        localNormal = GetLocalExitNormalAndCheck(IntersectPointGlobal,
    1648                                                 &validNormal);
    1649        *pNormalCalculated = fCalculatedExitNormal;
    1650 
    1651        G4AffineTransform localToGlobal = GetLocalToGlobalTransform();
    1652        globalNormal = localToGlobal.TransformAxis( localNormal );
    1653     }


    1329 // ********************************************************************
    1330 // GetLocalExitNormal
    1331 //
    1332 // Obtains the Normal vector to a surface (in local coordinates)
    1333 // pointing out of previous volume and into current volume
    1334 // ********************************************************************
    1335 //
    1336 G4ThreeVector G4Navigator::GetLocalExitNormal( G4bool* valid )
    1337 {
    1338   G4ThreeVector    ExitNormal(0.,0.,0.);
    1339   G4VSolid        *currentSolid=0;
    1340   G4LogicalVolume *candidateLogical;
    1341 
    1342   if ( fLastTriedStepComputation )
    1343   {
    1344     // use fLastLocatedPointLocal and next candidate volume
    1345     //
    1346     G4ThreeVector nextSolidExitNormal(0.,0.,0.);
    1347 
    1348     if( fEntering && (fBlockedPhysicalVolume!=0) )
    1349     {
    1350       candidateLogical= fBlockedPhysicalVolume->GetLogicalVolume();
    1351       if( candidateLogical )
    1352       {
    1353         // fLastStepEndPointLocal is in the coordinates of the mother
    1354         // we need it in the daughter's coordinate system.
    1355 
    1356         // The following code should also work in case of Replica
    1357         {
    1358           // First transform fLastLocatedPointLocal to the new daughter
    1359           // coordinates
    1360           //
    1361           G4AffineTransform MotherToDaughterTransform=
    1362             GetMotherToDaughterTransform( fBlockedPhysicalVolume,
    1363                                           fBlockedReplicaNo,
    1364                                           VolumeType(fBlockedPhysicalVolume) );
    1365           G4ThreeVector daughterPointOwnLocal=
    1366             MotherToDaughterTransform.TransformPoint( fLastStepEndPointLocal );
    1367 
    1368           // OK if it is a parameterised volume
    1369           //


    1454     if ( EnteredDaughterVolume() )
    1455     {
    1456       G4VSolid* daughterSolid =fHistory.GetTopVolume()->GetLogicalVolume()
    1457                                                       ->GetSolid();
    1458       ExitNormal= -(daughterSolid->SurfaceNormal(fLastLocatedPointLocal));



* daughterSolid->SurfaceNormal is outwards, but that gets flipped inwards 
  when the incident photon is going from mother to daughter (Pyrex->Vacuum)

  * so this is flipping based on volume hierarchy  
    




Easiest to use G4VSolid::SurfaceNormal(localPoint) : to avoid all the flipping that hides the real normal 
-------------------------------------------------------------------------------------------------------------

But that normal is inherently local. Preferable to avoid having to transform into local 
and back. 




::

    127     virtual G4ThreeVector SurfaceNormal(const G4ThreeVector& p) const = 0;
    128       // Returns the outwards pointing unit normal of the shape for the
    129       // surface closest to the point at offset p.


    276 //////////////////////////////////////////////////////////////////////////
    277 //
    278 // Return unit normal of surface closest to p
    279 
    280 G4ThreeVector G4Orb::SurfaceNormal( const G4ThreeVector& p ) const
    281 {
    282   return (1/p.mag())*p;
    283 }

Simply the gradient operator applied to the implicit eqn of the ellipsoid, 
gives the normal::

    290 ///////////////////////////////////////////////////////////////////////////////
    291 //
    292 // Return unit normal of surface closest to p not protected against p=0
    293 
    294 G4ThreeVector G4Ellipsoid::SurfaceNormal( const G4ThreeVector& p) const
    295 {
    296   G4double distR, distZBottom, distZTop;
    297 
    298   // normal vector with special magnitude:  parallel to normal, units 1/length
    299   // norm*p == 1.0 if on surface, >1.0 if outside, <1.0 if inside
    300   //
    301   G4ThreeVector norm(p.x()/(xSemiAxis*xSemiAxis),
    302                      p.y()/(ySemiAxis*ySemiAxis),
    303                      p.z()/(zSemiAxis*zSemiAxis));
    304   G4double radius = 1.0/norm.mag();

    /// simple sphere case the axes are all 1 ->  norm(x,y,z)  radius=1 
    /// so norm is the same as p, making distR zero for p on surface 

    305 
    306   // approximate distance to curved surface
    307   //
    308   distR = std::fabs( (p*norm - 1.0) * radius ) / 2.0;
    309 
    310   // Distance to z-cut plane
    311   //
    312   distZBottom = std::fabs( p.z() - zBottomCut );
    313   distZTop = std::fabs( p.z() - zTopCut );

    ///  extrema cuts are -r +r 

    314 
    315   if ( (distZBottom < distR) || (distZTop < distR) )
    316   {
    317     return G4ThreeVector(0.,0.,(distZBottom < distZTop) ? -1.0 : 1.0);
    318   }
    319   return ( norm *= radius );
    320 }


