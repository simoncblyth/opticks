sur_oldsur_matching
======================

Overview
-----------

The implicit criterior is triggering too much. 
The criterior needs to look at more details
to better predict G4OpBoundaryProcess. 
 
It is not sufficient just to check for RINDEX-NoRINDEX material
pairs. Also need to examine details of pre-existing optical 
surfaces on the boundary including the finish and the 
properties that are present (eg REFLECTIVITY)
that will sway what G4OpBoundaryProcess will do. 

In summary need to be able to predict when NoRINDEX is not 
a problem, in which case need to treat as ordinary surface 
rather than as an implicit. 

MAYBE : Just dont implicit override when pre-existing surface ?


Thoughts
-----------

When the original osur is -1 there is no problem, because 
when there is no preexisting surface adding the implicit should do 
what is required in making Opticks behave like Geant4. 

Where there is a preexisting osur it is a potential problem 
for the Geant4 simulation because it will not be honoured 
due to NoRINDEX fStopAndKill.  

* THIS NEEDS CONFIRMATION FOR SPECIFIC SURFACES AS DEPENDS ON DETAILS: FINISH, Surface MPT RINDEX etc..

* BASICALLY I NEED TO PREDICT WHAT G4OpBoundaryProcess WILL DO 
  IN ORDER TO CORRECTLY ASSIGN IMPLICITS 


However in the sense of matching Opticks to Geant4 it is 
not a problem because its easy for Opticks to match the 
NoRINDEX fStopAndKill with the implicit surface. 
 


g4-cls G4OpBoundaryProcess : RINDEX-NoRINDEX fStopAndKill aint so black and white
-------------------------------------------------------------------------------------

* it depends on finish, RINDEX in the surface MPT, ...
* polishedbackpainted groundbackpainted needs RINDEX in the surface MPT to avoid fStopAndKill
* TODO : some tests targetting suspect surfaces 

::

     169 G4VParticleChange*
     170 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     171 {


Material1 Must have MPT with RINDEX property to avoid NoRINDEX/fStopAndKill
-----------------------------------------------------------------------------

::

     278     G4MaterialPropertiesTable* aMaterialPropertiesTable;
     279         G4MaterialPropertyVector* Rindex;
     280 
     281     aMaterialPropertiesTable = Material1->GetMaterialPropertiesTable();
     282         if (aMaterialPropertiesTable) {
     283         Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
     284     }
     285     else {
     286                 theStatus = NoRINDEX;
     287                 if ( verboseLevel > 0) BoundaryProcessVerbose();
     288                 aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     289                 aParticleChange.ProposeTrackStatus(fStopAndKill);
     290                 return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     291     }
     292 
     293         if (Rindex) {
     294            Rindex1 = Rindex->Value(thePhotonMomentum);
     295         }
     296         else {
     297             theStatus = NoRINDEX;
     298                 if ( verboseLevel > 0) BoundaryProcessVerbose();
     299                 aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     300                 aParticleChange.ProposeTrackStatus(fStopAndKill);
     301                 return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     302     }
     303 





Properties relevant to G4OpBoundaryProcess::

    kRINDEX
    kREFLECTIVITY
    kEFFICIENCY

    kGROUPVEL        ## used in tail for (theStatus == FresnelRefraction || theStatus == Transmission)

    kREALRINDEX
    kIMAGINARYRINDEX
    kTRANSMITTANCE
    kSPECULARLOBECONSTANT
    kSPECULARSPIKECONSTANT
    kBACKSCATTERCONSTANT

::

    epsilon:geant4.10.04.p02 blyth$ grep GetProperty source/processes/optical/src/G4OpBoundaryProcess.cc
    Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
                  Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
                      aMaterialPropertiesTable->GetProperty(kREFLECTIVITY);
                      aMaterialPropertiesTable->GetProperty(kREALRINDEX);
                      aMaterialPropertiesTable->GetProperty(kIMAGINARYRINDEX);
              aMaterialPropertiesTable->GetProperty(kEFFICIENCY);
              aMaterialPropertiesTable->GetProperty(kTRANSMITTANCE);
                 aMaterialPropertiesTable->GetProperty(kSPECULARLOBECONSTANT);
                 aMaterialPropertiesTable->GetProperty(kSPECULARSPIKECONSTANT);
                 aMaterialPropertiesTable->GetProperty(kBACKSCATTERCONSTANT);
                 Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
           Material2->GetMaterialPropertiesTable()->GetProperty(kGROUPVEL);
                      aMaterialPropertiesTable->GetProperty(kREALRINDEX);
                      aMaterialPropertiesTable->GetProperty(kIMAGINARYRINDEX);



DONE : U4Tree::initSurfaces : collect full surface metadata
-------------------------------------------------------------

* will allow to constrain the surface details to study to only those in use


Finish : ground for most metals,  polishedfrontpainted with "mirror" in name, otherwise polished 
----------------------------------------------------------------------------------------------------

::

    epsilon:surface blyth$ pwd
    /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree/surface

    epsilon:surface blyth$ grep FinishName */NPFold_meta.txt

    CDInnerTyvekSurface/NPFold_meta.txt:FinishName:ground
    CDTyvekSurface/NPFold_meta.txt:FinishName:ground
    HamamatsuMaskOpticalSurface/NPFold_meta.txt:FinishName:ground
    NNVTMaskOpticalSurface/NPFold_meta.txt:FinishName:ground
    Steel_surface/NPFold_meta.txt:FinishName:ground
    Strut2AcrylicOpSurface/NPFold_meta.txt:FinishName:ground
    StrutAcrylicOpSurface/NPFold_meta.txt:FinishName:ground
    UpperChimneyTyvekSurface/NPFold_meta.txt:FinishName:ground
    VETOTyvekSurface/NPFold_meta.txt:FinishName:ground

    HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt:FinishName:polishedfrontpainted
    NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt:FinishName:polishedfrontpainted
    PMT_20inch_mirror_logsurf1/NPFold_meta.txt:FinishName:polishedfrontpainted
    PMT_20inch_mirror_logsurf2/NPFold_meta.txt:FinishName:polishedfrontpainted
    PMT_20inch_veto_mirror_logsurf1/NPFold_meta.txt:FinishName:polishedfrontpainted
    PMT_20inch_veto_mirror_logsurf2/NPFold_meta.txt:FinishName:polishedfrontpainted

    HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/NPFold_meta.txt:FinishName:polished
    HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/NPFold_meta.txt:FinishName:polished
    HamamatsuR12860_PMT_20inch_grid_opsurface/NPFold_meta.txt:FinishName:polished
    HamamatsuR12860_PMT_20inch_inner_edge_opsurface/NPFold_meta.txt:FinishName:polished
    HamamatsuR12860_PMT_20inch_inner_ring_opsurface/NPFold_meta.txt:FinishName:polished
    HamamatsuR12860_PMT_20inch_outer_edge_opsurface/NPFold_meta.txt:FinishName:polished
    HamamatsuR12860_PMT_20inch_shield_opsurface/NPFold_meta.txt:FinishName:polished
    NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NPFold_meta.txt:FinishName:polished
    NNVTMCPPMT_PMT_20inch_mcp_opsurface/NPFold_meta.txt:FinishName:polished
    NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NPFold_meta.txt:FinishName:polished
    NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NPFold_meta.txt:FinishName:polished
    PMT_20inch_photocathode_logsurf1/NPFold_meta.txt:FinishName:polished
    PMT_20inch_photocathode_logsurf2/NPFold_meta.txt:FinishName:polished
    PMT_20inch_veto_photocathode_logsurf1/NPFold_meta.txt:FinishName:polished
    PMT_20inch_veto_photocathode_logsurf2/NPFold_meta.txt:FinishName:polished
    PMT_3inch_absorb_logsurf1/NPFold_meta.txt:FinishName:polished
    PMT_3inch_absorb_logsurf2/NPFold_meta.txt:FinishName:polished
    PMT_3inch_absorb_logsurf3/NPFold_meta.txt:FinishName:polished
    PMT_3inch_absorb_logsurf4/NPFold_meta.txt:FinishName:polished
    PMT_3inch_absorb_logsurf5/NPFold_meta.txt:FinishName:polished
    PMT_3inch_absorb_logsurf6/NPFold_meta.txt:FinishName:polished
    PMT_3inch_absorb_logsurf7/NPFold_meta.txt:FinishName:polished
    PMT_3inch_absorb_logsurf8/NPFold_meta.txt:FinishName:polished
    PMT_3inch_photocathode_logsurf1/NPFold_meta.txt:FinishName:polished
    PMT_3inch_photocathode_logsurf2/NPFold_meta.txt:FinishName:polished


ModelName : struts etc.. unified, PMT glisur
-----------------------------------------------------

::

    epsilon:surface blyth$ grep ModelName */NPFold_meta.txt
    CDInnerTyvekSurface/NPFold_meta.txt:ModelName:unified
    CDTyvekSurface/NPFold_meta.txt:ModelName:unified
    HamamatsuMaskOpticalSurface/NPFold_meta.txt:ModelName:unified
    NNVTMaskOpticalSurface/NPFold_meta.txt:ModelName:unified
    Steel_surface/NPFold_meta.txt:ModelName:unified
    Strut2AcrylicOpSurface/NPFold_meta.txt:ModelName:unified
    StrutAcrylicOpSurface/NPFold_meta.txt:ModelName:unified
    UpperChimneyTyvekSurface/NPFold_meta.txt:ModelName:unified
    VETOTyvekSurface/NPFold_meta.txt:ModelName:unified


unified is that prob_sl, prob_ss, prob_bs guff used by::

    410           if ( theModel == unified ) {
     411                  PropertyPointer =
     412                  aMaterialPropertiesTable->GetProperty(kSPECULARLOBECONSTANT);
     413                  if (PropertyPointer) {
     414                          prob_sl =
     415                          PropertyPointer->Value(thePhotonMomentum);
     416                  } else {
     417                          prob_sl = 0.0;
     418                  }
     419 
     420                  PropertyPointer =
     421                  aMaterialPropertiesTable->GetProperty(kSPECULARSPIKECONSTANT);
     422              if (PropertyPointer) {
     423                          prob_ss =
     424                          PropertyPointer->Value(thePhotonMomentum);
     425                  } else {
     426                          prob_ss = 0.0;
     427                  }
     428 
     429                  PropertyPointer =
     430                  aMaterialPropertiesTable->GetProperty(kBACKSCATTERCONSTANT);
     431                  if (PropertyPointer) {
     432                          prob_bs =
     433                          PropertyPointer->Value(thePhotonMomentum);
     434                  } else {
     435                          prob_bs = 0.0;
     436                  }
     437               }
     438            }


::

      |-----prob_ss-------|---prob_sl--------|------prob_bs--------|  

          SpikeReflection     LobeReflection    BackScattering

When all three prob are zero, as I expect will happen will get : LambertianReflection



::

    301 inline
    302 void G4OpBoundaryProcess::ChooseReflection()
    303 {
    304                  G4double rand = G4UniformRand();
    305                  if ( rand >= 0.0 && rand < prob_ss ) {
    306                     theStatus = SpikeReflection;
    307                     theFacetNormal = theGlobalNormal;
    308                  }
    309                  else if ( rand >= prob_ss &&
    310                            rand <= prob_ss+prob_sl) {
    311                     theStatus = LobeReflection;
    312                  }
    313                  else if ( rand > prob_ss+prob_sl &&
    314                            rand < prob_ss+prob_sl+prob_bs ) {
    315                     theStatus = BackScattering;
    316                  }
    317                  else {
    318                     theStatus = LambertianReflection;
    319                  }
    320 }



    HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/NPFold_meta.txt:ModelName:glisur
    HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/NPFold_meta.txt:ModelName:glisur
    HamamatsuR12860_PMT_20inch_grid_opsurface/NPFold_meta.txt:ModelName:glisur
    HamamatsuR12860_PMT_20inch_inner_edge_opsurface/NPFold_meta.txt:ModelName:glisur
    HamamatsuR12860_PMT_20inch_inner_ring_opsurface/NPFold_meta.txt:ModelName:glisur
    HamamatsuR12860_PMT_20inch_outer_edge_opsurface/NPFold_meta.txt:ModelName:glisur
    HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt:ModelName:glisur
    HamamatsuR12860_PMT_20inch_shield_opsurface/NPFold_meta.txt:ModelName:glisur

    NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NPFold_meta.txt:ModelName:glisur
    NNVTMCPPMT_PMT_20inch_mcp_opsurface/NPFold_meta.txt:ModelName:glisur
    NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NPFold_meta.txt:ModelName:glisur
    NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NPFold_meta.txt:ModelName:glisur
    NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt:ModelName:glisur

    PMT_20inch_mirror_logsurf1/NPFold_meta.txt:ModelName:glisur
    PMT_20inch_mirror_logsurf2/NPFold_meta.txt:ModelName:glisur
    PMT_20inch_photocathode_logsurf1/NPFold_meta.txt:ModelName:glisur
    PMT_20inch_photocathode_logsurf2/NPFold_meta.txt:ModelName:glisur
    PMT_20inch_veto_mirror_logsurf1/NPFold_meta.txt:ModelName:glisur
    PMT_20inch_veto_mirror_logsurf2/NPFold_meta.txt:ModelName:glisur
    PMT_20inch_veto_photocathode_logsurf1/NPFold_meta.txt:ModelName:glisur
    PMT_20inch_veto_photocathode_logsurf2/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_absorb_logsurf1/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_absorb_logsurf2/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_absorb_logsurf3/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_absorb_logsurf4/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_absorb_logsurf5/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_absorb_logsurf6/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_absorb_logsurf7/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_absorb_logsurf8/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_photocathode_logsurf1/NPFold_meta.txt:ModelName:glisur
    PMT_3inch_photocathode_logsurf2/NPFold_meta.txt:ModelName:glisur


TypeName : all dielectric_metal
-----------------------------------

::

    epsilon:surface blyth$ grep TypeName */NPFold_meta.txt

    CDInnerTyvekSurface/NPFold_meta.txt:TypeName:dielectric_metal
    CDTyvekSurface/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuMaskOpticalSurface/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuR12860_PMT_20inch_grid_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuR12860_PMT_20inch_inner_edge_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuR12860_PMT_20inch_inner_ring_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuR12860_PMT_20inch_outer_edge_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt:TypeName:dielectric_metal
    HamamatsuR12860_PMT_20inch_shield_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    NNVTMCPPMT_PMT_20inch_mcp_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NPFold_meta.txt:TypeName:dielectric_metal
    NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt:TypeName:dielectric_metal
    NNVTMaskOpticalSurface/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_20inch_mirror_logsurf1/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_20inch_mirror_logsurf2/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_20inch_photocathode_logsurf1/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_20inch_photocathode_logsurf2/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_20inch_veto_mirror_logsurf1/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_20inch_veto_mirror_logsurf2/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_20inch_veto_photocathode_logsurf1/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_20inch_veto_photocathode_logsurf2/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_absorb_logsurf1/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_absorb_logsurf2/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_absorb_logsurf3/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_absorb_logsurf4/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_absorb_logsurf5/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_absorb_logsurf6/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_absorb_logsurf7/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_absorb_logsurf8/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_photocathode_logsurf1/NPFold_meta.txt:TypeName:dielectric_metal
    PMT_3inch_photocathode_logsurf2/NPFold_meta.txt:TypeName:dielectric_metal
    Steel_surface/NPFold_meta.txt:TypeName:dielectric_metal
    Strut2AcrylicOpSurface/NPFold_meta.txt:TypeName:dielectric_metal
    StrutAcrylicOpSurface/NPFold_meta.txt:TypeName:dielectric_metal
    UpperChimneyTyvekSurface/NPFold_meta.txt:TypeName:dielectric_metal
    VETOTyvekSurface/NPFold_meta.txt:TypeName:dielectric_metal
    epsilon:surface blyth$ 


dielectric_metal
-----------------

::

     718 void G4OpBoundaryProcess::DielectricMetal()
     719 {
     720         G4int n = 0;
     721         G4double rand, PdotN, EdotN;
     722         G4ThreeVector A_trans, A_paral;
     723 
     724         do {
     725 
     726            n++;
     727 
     728            rand = G4UniformRand();
     729            if ( rand > theReflectivity && n == 1 ) {
     730               if (rand > theReflectivity + theTransmittance) {
     731                 DoAbsorption();
     732               } else {

     /// theTransmittance nomally zero : so the below bizarre fall thru Transmission
     /// doesnt happen : > theReflectivity on dielectric_metal causes DoAbsorption 

     733                 theStatus = Transmission;
     734                 NewMomentum = OldMomentum;
     735                 NewPolarization = OldPolarization;
     736               }
     737               break;
     738            }
     739            else {



DONE : review where are Implicit and perfect surfaces added in X4/GGeo workflow as well as U4Tree/stree workflow
------------------------------------------------------------------------------------------------------------------

::

    epsilon:extg4 blyth$ opticks-f Implicit_RINDEX_NoRINDEX
    ./extg4/X4PhysicalVolume.cc:    static const char* IMPLICIT_PREFIX = "Implicit_RINDEX_NoRINDEX" ; 
    ./sysrap/stree.h:* ~/opticks/notes/issues/stree_bd_names_and_Implicit_RINDEX_NoRINDEX.rst
    ./sysrap/SBnd.h:    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air
    ./sysrap/SBnd.h:    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air
    ./ggeo/GSurfaceLib.cc:    ss << "Implicit_RINDEX_NoRINDEX_" << spv1 << "_" << spv2 ;  
    ./u4/U4Tree.h:* see ~/opticks/notes/issues/stree_bd_names_and_Implicit_RINDEX_NoRINDEX.rst
    epsilon:opticks blyth$ 



Observation : no significant diff in payload group 0 of the 40 sur in common
-------------------------------------------------------------------------------

::

    In [35]: np.abs( a[:40,0,:,:]-b[:40,0,:,:] ).max()
    Out[35]: 6.100014653676045e-07


DONE : first attempt at add implicit handling to U4Tree/stree : IS FINDING TOO MANY IMPLICITS
------------------------------------------------------------------------------------------------

HMM : getting too many 107 implicits...
-------------------------------------------

Must be some bug, surely there cannot be so many materials without RINDEX. 
Thats not the problem. Need to make more detailed examination of a 
border to judge if need to override with an implicit. 

::

    epsilon:stree blyth$ head -30 /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree/implicit.txt 
    Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock
    Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox
    Implicit_RINDEX_NoRINDEX_pExpHall_pPoolCover
    Implicit_RINDEX_NoRINDEX_lUpperChimney_phys_pUpperChimneySteel
    Implicit_RINDEX_NoRINDEX_lUpperChimney_phys_pUpperChimneyTyvek
    Implicit_RINDEX_NoRINDEX_pPlane_0_ff__pPanel_0_f_
    Implicit_RINDEX_NoRINDEX_pPlane_0_ff__pPanel_1_f_
    Implicit_RINDEX_NoRINDEX_pPlane_0_ff__pPanel_2_f_
    Implicit_RINDEX_NoRINDEX_pPlane_0_ff__pPanel_3_f_
    Implicit_RINDEX_NoRINDEX_pPlane_1_ff__pPanel_0_f_
    Implicit_RINDEX_NoRINDEX_pPlane_1_ff__pPanel_1_f_
    Implicit_RINDEX_NoRINDEX_pPlane_1_ff__pPanel_2_f_
    Implicit_RINDEX_NoRINDEX_pPlane_1_ff__pPanel_3_f_
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pPoolLining
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLw1.up10_up11_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLw1.up09_up10_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLw1.up08_up09_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLw1.up07_up08_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLw1.up06_up07_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLw1.up05_up06_HBeam_phys
    ...


    epsilon:stree blyth$ tail -30  /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree/implicit.txt 
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.A03_B04_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.A04_B05_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.A05_B06_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.A06_B07_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.B01_B01_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.B03_B03_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.B05_B05_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.A03_A03_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_ZC2.A05_A05_HBeam_phys
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pCentralDetector
    Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector
    Implicit_RINDEX_NoRINDEX_pInnerWater_lSteel_phys
    Implicit_RINDEX_NoRINDEX_pInnerWater_lSteel2_phys
    Implicit_RINDEX_NoRINDEX_pInnerWater_lFasteners_phys
    Implicit_RINDEX_NoRINDEX_pInnerWater_lUpper_phys
    Implicit_RINDEX_NoRINDEX_pLPMT_Hamamatsu_R12860_HamamatsuR12860pMaskTail
    Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_plate_phy
    Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_outer_edge_phy
    Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_inner_edge_phy
    Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_inner_ring_phy
    Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_dynode_tube_phy
    Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_grid_phy
    Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_shield_phy
    Implicit_RINDEX_NoRINDEX_pLPMT_NNVT_MCPPMT_NNVTMCPPMTpMaskTail
    Implicit_RINDEX_NoRINDEX_NNVTMCPPMT_PMT_20inch_inner_phys_NNVTMCPPMT_PMT_20inch_edge_phy
    Implicit_RINDEX_NoRINDEX_NNVTMCPPMT_PMT_20inch_inner_phys_NNVTMCPPMT_PMT_20inch_plate_phy
    Implicit_RINDEX_NoRINDEX_NNVTMCPPMT_PMT_20inch_inner_phys_NNVTMCPPMT_PMT_20inch_tube_phy
    Implicit_RINDEX_NoRINDEX_NNVTMCPPMT_PMT_20inch_inner_phys_NNVTMCPPMT_PMT_20inch_mcp_phy
    Implicit_RINDEX_NoRINDEX_PMT_3inch_log_phys_PMT_3inch_cntr_phys
    Implicit_RINDEX_NoRINDEX_lLowerChimney_phys_pLowerChimneySteel
    epsilon:stree blyth$ 


::

    epsilon:DetSim blyth$ find Material -name RINDEX 
    Material/Pyrex/RINDEX
    Material/photocathode_Ham20inch/RINDEX
    Material/ETFE/RINDEX
    Material/FEP/RINDEX
    Material/VacuumT/RINDEX
    Material/AcrylicMask/RINDEX
    Material/Water/RINDEX
    Material/photocathode_HZC9inch/RINDEX
    Material/Vacuum/RINDEX
    Material/LAB/RINDEX
    Material/vetoWater/RINDEX
    Material/Air/RINDEX
    Material/photocathode_MCP8inch/RINDEX
    Material/MineralOil/RINDEX
    Material/PA/RINDEX
    Material/Mylar/RINDEX
    Material/Acrylic/RINDEX
    Material/PE_PA/RINDEX
    Material/LS/RINDEX
    Material/photocathode_3inch/RINDEX
    Material/photocathode_Ham8inch/RINDEX
    Material/photocathode_MCP20inch/RINDEX
    Material/photocathode/RINDEX
    epsilon:DetSim blyth$ 


::

    epsilon:stree blyth$ cat mtname_no_rindex.txt
    Rock
    Galactic
    Steel
    Tyvek
    Scintillator
    TiO2Coating
    Adhesive
    Aluminium
    LatticedShellSteel
    StrutSteel
    CDReflectorSteel

    epsilon:stree blyth$ pwd
    /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree

    epsilon:stree blyth$ cat mtname.txt
    Air
    Rock                ##
    Galactic            ##
    Steel               ##
    LS                  
    Tyvek               ## 
    Scintillator        ##
    TiO2Coating         ##
    Adhesive            ##
    Aluminium           ##
    LatticedShellSteel  ##
    Acrylic
    PE_PA
    StrutSteel          ## 
    AcrylicMask
    CDReflectorSteel    ##
    Vacuum
    Pyrex
    Water
    vetoWater
    epsilon:stree blyth$ 



Only 2 isur implicit overrides, loads of osur implicit overrides
--------------------------------------------------------------------

::

    U4Tree::initNodes_r changing isur from 2 to 53 num_surfaces 40 implicit_idx 13
    U4Tree::initNodes_r changing osur from 0 to 126 num_surfaces 40 implicit_idx 86
    U4Tree::initNodes_r changing isur from 1 to 127 num_surfaces 40 implicit_idx 87
    U4Tree::initNodes_r changing osur from 33 to 128 num_surfaces 40 implicit_idx 88
    U4Tree::initNodes_r changing osur from 33 to 128 num_surfaces 40 implicit_idx 88
    U4Tree::initNodes_r changing osur from 33 to 128 num_surfaces 40 implicit_idx 88
    U4Tree::initNodes_r changing osur from 33 to 128 num_surfaces 40 implicit_idx 88
    U4Tree::initNodes_r changing osur from 33 to 128 num_surfaces 40 implicit_idx 88


Changed to verbose implicit naming for debug::

    epsilon:stree blyth$ cat implicit.txt
    Implicit__RINDEX__pDomeAir__Air__NoRINDEX__pDomeRock__Rock
    Implicit__RINDEX__pExpHall__Air__NoRINDEX__pExpRockBox__Rock
    Implicit__RINDEX__pExpHall__Air__NoRINDEX__pPoolCover__Steel
    Implicit__RINDEX__lUpperChimney_phys__Air__NoRINDEX__pUpperChimneySteel__Steel
    Implicit__RINDEX__lUpperChimney_phys__Air__NoRINDEX__pUpperChimneyTyvek__Tyvek
    Implicit__RINDEX__pPlane_0_ff___Air__NoRINDEX__pPanel_0_f___Aluminium
    Implicit__RINDEX__pPlane_0_ff___Air__NoRINDEX__pPanel_1_f___Aluminium
    Implicit__RINDEX__pPlane_0_ff___Air__NoRINDEX__pPanel_2_f___Aluminium
    Implicit__RINDEX__pPlane_0_ff___Air__NoRINDEX__pPanel_3_f___Aluminium
    Implicit__RINDEX__pPlane_1_ff___Air__NoRINDEX__pPanel_0_f___Aluminium
    Implicit__RINDEX__pPlane_1_ff___Air__NoRINDEX__pPanel_1_f___Aluminium
    Implicit__RINDEX__pPlane_1_ff___Air__NoRINDEX__pPanel_2_f___Aluminium
    Implicit__RINDEX__pPlane_1_ff___Air__NoRINDEX__pPanel_3_f___Aluminium
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__pPoolLining__Tyvek
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up10_up11_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up09_up10_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up08_up09_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up07_up08_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up06_up07_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up05_up06_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up04_up05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up03_up04_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up02_up03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up01_up02_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw2.equ_up01_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw2.equ_bt01_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw3.bt01_bt02_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw3.bt02_bt03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw2.bt03_bt04_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw2.bt04_bt05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.bt05_bt06_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.bt06_bt07_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.bt07_bt08_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.bt08_bt09_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.bt09_bt10_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.bt10_bt11_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb3.up11_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb4.up10_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb3.up09_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb2.up08_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb2.up07_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb2.up06_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.up05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.up04_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.up03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.up02_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.up01_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb2.equ_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb2.bt01_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.bt02_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb2.bt03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb2.bt04_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.bt05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.bt06_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.bt07_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb1.bt08_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb3.bt09_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb3.bt10_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLb3.bt11_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.A01_02_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.A02_03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.A03_04_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.A04_05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.A05_06_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.A06_07_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.B01_02_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.B02_03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.B03_04_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.B04_05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.B05_06_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GZ1.B06_07_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A02_B02_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A03_B03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A04_B04_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A05_B05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A06_B06_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A02_B03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A03_B04_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A04_B05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A05_B06_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A06_B07_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.B01_B01_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.B03_B03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.B05_B05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A03_A03_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__ZC2.A05_A05_HBeam_phys__LatticedShellSteel
    Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__pCentralDetector__Tyvek
    Implicit__RINDEX__pInnerWater__Water__NoRINDEX__pCentralDetector__Tyvek
    Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lSteel_phys__StrutSteel
    Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lSteel2_phys__StrutSteel
    Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lSteel_phys__Steel
    Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lFasteners_phys__Steel
    Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lUpper_phys__Steel
    Implicit__RINDEX__pLPMT_Hamamatsu_R12860__Water__NoRINDEX__HamamatsuR12860pMaskTail__CDReflectorSteel
    Implicit__RINDEX__HamamatsuR12860_PMT_20inch_inner_phys__Vacuum__NoRINDEX__HamamatsuR12860_PMT_20inch_plate_phy__Steel
    Implicit__RINDEX__HamamatsuR12860_PMT_20inch_inner_phys__Vacuum__NoRINDEX__HamamatsuR12860_PMT_20inch_outer_edge_phy__Steel
    Implicit__RINDEX__HamamatsuR12860_PMT_20inch_inner_phys__Vacuum__NoRINDEX__HamamatsuR12860_PMT_20inch_inner_edge_phy__Steel
    Implicit__RINDEX__HamamatsuR12860_PMT_20inch_inner_phys__Vacuum__NoRINDEX__HamamatsuR12860_PMT_20inch_inner_ring_phy__Steel
    Implicit__RINDEX__HamamatsuR12860_PMT_20inch_inner_phys__Vacuum__NoRINDEX__HamamatsuR12860_PMT_20inch_dynode_tube_phy__Steel
    Implicit__RINDEX__HamamatsuR12860_PMT_20inch_inner_phys__Vacuum__NoRINDEX__HamamatsuR12860_PMT_20inch_grid_phy__Steel
    Implicit__RINDEX__HamamatsuR12860_PMT_20inch_inner_phys__Vacuum__NoRINDEX__HamamatsuR12860_PMT_20inch_shield_phy__Steel
    Implicit__RINDEX__pLPMT_NNVT_MCPPMT__Water__NoRINDEX__NNVTMCPPMTpMaskTail__CDReflectorSteel
    Implicit__RINDEX__NNVTMCPPMT_PMT_20inch_inner_phys__Vacuum__NoRINDEX__NNVTMCPPMT_PMT_20inch_edge_phy__Steel
    Implicit__RINDEX__NNVTMCPPMT_PMT_20inch_inner_phys__Vacuum__NoRINDEX__NNVTMCPPMT_PMT_20inch_plate_phy__Steel
    Implicit__RINDEX__NNVTMCPPMT_PMT_20inch_inner_phys__Vacuum__NoRINDEX__NNVTMCPPMT_PMT_20inch_tube_phy__Steel
    Implicit__RINDEX__NNVTMCPPMT_PMT_20inch_inner_phys__Vacuum__NoRINDEX__NNVTMCPPMT_PMT_20inch_mcp_phy__Steel
    Implicit__RINDEX__PMT_3inch_log_phys__Water__NoRINDEX__PMT_3inch_cntr_phys__Steel
    Implicit__RINDEX__lLowerChimney_phys__Water__NoRINDEX__pLowerChimneySteel__Steel
    epsilon:stree blyth$ 



implicit_isur implicit_osur
-----------------------------

These are collected in U4Tree::initNodes they contain the bd indices
before and after the implicit swaps. 

::

     592     if( implicit_idx > -1 )
     593     {
     594         int num_surfaces = surfaces.size() ;
     595         if(implicit_outwards) // from imat to omat : isur is relevant 
     596         {
     597             //assert(isur == -1 );          // only 2 of these
     598             if( isur != -1 ) std::cerr
     599                 << "U4Tree::initNodes_r"
     600                 << " changing isur from " << isur
     601                 << " to " << ( num_surfaces + implicit_idx )
     602                 << " num_surfaces " << num_surfaces
     603                 << " implicit_idx " << implicit_idx
     604                 << std::endl
     605                 ;
     606 
     607             st->implicit_isur.push_back( {omat, osur, isur, imat} );
     608             isur = num_surfaces + implicit_idx ;
     609             st->implicit_isur.push_back( {omat, osur, isur, imat} );
     610         }
     611         else if(implicit_inwards) // from omat to imat : osur is relevant
     612         {
     613             //assert(osur == -1 );           // loads of these
     614             if( osur != -1 ) std::cerr
     615                 << "U4Tree::initNodes_r"
     616                 << " changing osur from " << osur
     617                 << " to " << ( num_surfaces + implicit_idx )
     618                 << " num_surfaces " << num_surfaces
     619                 << " implicit_idx " << implicit_idx
     620                 << std::endl
     621                 ;
     622 
     623             st->implicit_osur.push_back( {omat, osur, isur, imat} );
     624             osur = num_surfaces + implicit_idx ;
     625             st->implicit_osur.push_back( {omat, osur, isur, imat} );
     626 
     627         }
     628     }

::

    st
    ./stree_sur_test.sh ana




isur : s.implicit_isur.reshape(-1,8) : "outwards" implicits
-------------------------------------------------------------

::

    In [85]: isur = s.implicit_isur.reshape(-1,8) ; isur                                                                         

    In [91]: isur
    Out[91]: 
    array([[  1,  -1,  -1,   0,   1,  -1,  40,   0],
           [  1,  -1,  -1,   0,   1,  -1,  41,   0],
           [  5,  -1,   2,  19,   5,  -1,  53,  19],
           [  5,  -1,   1,  18,   5,  -1, 127,  18]], dtype=int32)



    Out[85]: isur  ## line by line 
    array([[  1,  -1,  -1,   0,   1,  -1,  40,   0],
           [  1,  -1,  -1,   0,   1,  -1,  41,   0],

            In [86]:  mtn[np.array([1,0])]
            Out[86]: array(['Rock', 'Air'], dtype='<U18')

            In [87]: np.c_[sun[np.array([40,41])]]                                                                                       
            Out[87]: 
            array([['Implicit__RINDEX__pDomeAir__Air__NoRINDEX__pDomeRock__Rock'],
                   ['Implicit__RINDEX__pExpHall__Air__NoRINDEX__pExpRockBox__Rock']], dtype='<U124')

            ## HANDLING Air TO Rock WITH ISUR : NO PROBLEM 


           [  5,  -1,   2,  19,   5,  -1,  53,  19],

            In [88]:  mtn[np.array([5,19])]                                                                                              
            Out[88]: array(['Tyvek', 'vetoWater'], dtype='<U18')

            In [90]: np.c_[sun[np.array([2,53])]]
            Out[90]: 
            array([['VETOTyvekSurface'],
                   ['Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__pPoolLining__Tyvek']], dtype='<U124')

            ## HANDLING vetoWater TO Tyvek WITH ISUR : PROBLEM : FALSE IMPLICIT ?


           [  5,  -1,   1,  18,   5,  -1, 127,  18]], dtype=int32)

            In [89]:  mtn[np.array([5,18])]                                                                                              
            Out[89]: array(['Tyvek', 'Water'], dtype='<U18')



 


osur : implicit_osur.reshape(-1,8) : "inwards" implicits
-----------------------------------------------------------

The bd int4 (omat,osur,isur,imat) before and after implicit swaps.
The osur column changes in every case, as that is the implicit_inwards requirement 
that triggers the collection::

    In [22]: osur = s.implicit_osur.reshape(-1,8) ; osur
    Out[22]: 
    array([[  0,  -1,  -1,   3,   0,  42,  -1,   3],
           [  0,  -1,  -1,   3,   0,  43,  -1,   3],
           [  0,  -1,  -1,   5,   0,  44,  -1,   5],
           [  0,  -1,  -1,   9,   0,  45,  -1,   9],
           [  0,  -1,  -1,   9,   0,  46,  -1,   9],
           ...,
           [ 18,  -1,  -1,   3,  18, 146,  -1,   3],
           [ 18,  -1,  -1,   3,  18, 146,  -1,   3],
           [ 18,  -1,  -1,   3,  18, 146,  -1,   3],
           [ 18,  -1,  -1,   3,  18, 146,  -1,   3],
           [ 18,  39,  39,   3,  18, 147,  39,   3]], dtype=int32)


After rearrange code, check get same::

    In [2]: osur = s.implicit_osur.reshape(-1,8) ; osur
    Out[2]: 
    array([[  0,  -1,  -1,   3,   0,  42,  -1,   3],
           [  0,  -1,  -1,   3,   0,  43,  -1,   3],
           [  0,  -1,  -1,   5,   0,  44,  -1,   5],
           [  0,  -1,  -1,   9,   0,  45,  -1,   9],
           [  0,  -1,  -1,   9,   0,  46,  -1,   9],
           ...,
           [ 18,  -1,  -1,   3,  18, 146,  -1,   3],
           [ 18,  -1,  -1,   3,  18, 146,  -1,   3],
           [ 18,  -1,  -1,   3,  18, 146,  -1,   3],
           [ 18,  -1,  -1,   3,  18, 146,  -1,   3],
           [ 18,  39,  39,   3,  18, 147,  39,   3]], dtype=int32)

    In [3]: osur.shape
    Out[3]: (133640, 8)

::

    In [23]: osur.shape
    Out[23]: (133640, 8)





Examine counts of the 104 unique osur swaps::

    In [29]: u_osur,i_osur,n_osur = np.unique(osur,axis=0,return_index=True,return_counts=True)

    In [65]: u_osur.shape
    Out[65]: (104, 8)

    In [39]: np.set_printoptions(edgeitems=200)

    In [43]: mtn = np.array(s.mtname)  

    In [50]: sun = np.array(s.sur_names)    

    In [40]: np.c_[n_osur, i_osur, u_osur][:120]
    Out[40]: 
    array([[     1,      0,      0,     -1,     -1,      3,      0,     42,     -1,      3],
           [     1,      1,      0,     -1,     -1,      3,      0,     43,     -1,      3],
                               omat    osur    isur    imat    omat    osur    isur     imat  

            In [46]: mtn[np.array([0,3])]
            Out[46]: array(['Air', 'Steel'], dtype='<U18')

            In [53]: sun[42]
            Out[53]: 'Implicit__RINDEX__pExpHall__Air__NoRINDEX__pPoolCover__Steel'

            In [54]: sun[43]
            Out[54]: 'Implicit__RINDEX__lUpperChimney_phys__Air__NoRINDEX__pUpperChimneySteel__Steel'

           [     1,      2,      0,     -1,     -1,      5,      0,     44,     -1,      5],

            In [47]: mtn[np.array([0,5])]
            Out[47]: array(['Air', 'Tyvek'], dtype='<U18')

            In [55]: sun[44]
            Out[55]: 'Implicit__RINDEX__lUpperChimney_phys__Air__NoRINDEX__pUpperChimneyTyvek__Tyvek'


           [    63,      3,      0,     -1,     -1,      9,      0,     45,     -1,      9],
           [    63,      4,      0,     -1,     -1,      9,      0,     46,     -1,      9],
           [    63,      5,      0,     -1,     -1,      9,      0,     47,     -1,      9],
           [    63,      6,      0,     -1,     -1,      9,      0,     48,     -1,      9],
           [    63,      7,      0,     -1,     -1,      9,      0,     49,     -1,      9],
           [    63,      8,      0,     -1,     -1,      9,      0,     50,     -1,      9],
           [    63,      9,      0,     -1,     -1,      9,      0,     51,     -1,      9],
           [    63,     10,      0,     -1,     -1,      9,      0,     52,     -1,      9],

            In [56]: mtn[np.array([0,9])]
            Out[56]: array(['Air', 'Aluminium'], dtype='<U18')

            In [58]: np.c_[sun[np.array([45,46,47,48,49,50,51,52])]]
            Out[58]: 
            array([['Implicit__RINDEX__pPlane_0_ff___Air__NoRINDEX__pPanel_0_f___Aluminium'],
                   ['Implicit__RINDEX__pPlane_0_ff___Air__NoRINDEX__pPanel_1_f___Aluminium'],
                   ['Implicit__RINDEX__pPlane_0_ff___Air__NoRINDEX__pPanel_2_f___Aluminium'],
                   ['Implicit__RINDEX__pPlane_0_ff___Air__NoRINDEX__pPanel_3_f___Aluminium'],
                   ['Implicit__RINDEX__pPlane_1_ff___Air__NoRINDEX__pPanel_0_f___Aluminium'],
                   ['Implicit__RINDEX__pPlane_1_ff___Air__NoRINDEX__pPanel_1_f___Aluminium'],
                   ['Implicit__RINDEX__pPlane_1_ff___Air__NoRINDEX__pPanel_2_f___Aluminium'],
                   ['Implicit__RINDEX__pPlane_1_ff___Air__NoRINDEX__pPanel_3_f___Aluminium']], dtype='<U124')



           [  4997,   4989,     16,      7,     35,      3,     16,    134,     35,      3],
           [  4997,   4992,     16,      8,     35,      3,     16,    137,     35,      3],
           [  4997,   4990,     16,      9,     35,      3,     16,    135,     35,      3],
           [  4997,   4991,     16,     10,     35,      3,     16,    136,     35,      3],
           [  4997,   4993,     16,     11,     35,      3,     16,    138,     35,      3],
           [  4997,   4994,     16,     12,     35,      3,     16,    139,     35,      3],
           [  4997,   4995,     16,     13,     35,      3,     16,    140,     35,      3],


            In [66]: mtn[np.array([16,3])]
            Out[66]: array(['Vacuum', 'Steel'], dtype='<U18')

            In [69]: np.c_[sun[np.array([35])]]
            Out[69]: array([['HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf']], dtype='<U124')

            In [70]: np.c_[sun[np.array([7,8,9,10,11,12])]]
            Out[70]: 
            array([['HamamatsuR12860_PMT_20inch_dynode_plate_opsurface'],
                   ['HamamatsuR12860_PMT_20inch_inner_ring_opsurface'],
                   ['HamamatsuR12860_PMT_20inch_outer_edge_opsurface'],
                   ['HamamatsuR12860_PMT_20inch_inner_edge_opsurface'],
                   ['HamamatsuR12860_PMT_20inch_dynode_tube_opsurface'],
                   ['HamamatsuR12860_PMT_20inch_grid_opsurface']], dtype='<U124')

            ## Because Steel has no RINDEX all these opsurface are ignored ? 
            ## NO : it aint so black and white, it depends on finish, surface MPT content 

Those opsurface have only REFLECTIVITY::

    epsilon:surface blyth$ l HamamatsuR*/*.npy
    8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/REFLECTIVITY.npy
    8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/REFLECTIVITY.npy
    8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 HamamatsuR12860_PMT_20inch_grid_opsurface/REFLECTIVITY.npy
    8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 HamamatsuR12860_PMT_20inch_inner_edge_opsurface/REFLECTIVITY.npy
    8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 HamamatsuR12860_PMT_20inch_shield_opsurface/REFLECTIVITY.npy
    8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/REFLECTIVITY.npy
    8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 HamamatsuR12860_PMT_20inch_inner_ring_opsurface/REFLECTIVITY.npy
    8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 HamamatsuR12860_PMT_20inch_outer_edge_opsurface/REFLECTIVITY.npy
    epsilon:surface blyth$ pwd
    /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree/surface
    epsilon:surface blyth$ 

TODO : add finish etc.. to surface metadata::

    epsilon:HamamatsuR12860_PMT_20inch_outer_edge_opsurface blyth$ cat NPFold_meta.txt
    OpticalSurfaceName:outerEdgeOpSurface
    pv1:HamamatsuR12860_PMT_20inch_inner_phys
    pv2:HamamatsuR12860_PMT_20inch_outer_edge_phy
    type:Border
    epsilon:HamamatsuR12860_PMT_20inch_outer_edge_opsurface blyth$ 




           [ 12615,   5006,     16,     14,     37,      3,     16,    143,     37,      3],
           [ 12615,   5005,     16,     15,     37,      3,     16,    142,     37,      3],
           [ 12615,   5007,     16,     16,     37,      3,     16,    144,     37,      3],
           [ 12615,   5008,     16,     17,     37,      3,     16,    145,     37,      3],

            In [66]: mtn[np.array([16,3])]                     ## bits of Steel inside PMT Vacuum 
            Out[66]: array(['Vacuum', 'Steel'], dtype='<U18')

            In [71]: np.c_[sun[np.array([14,15,16,17])]]
            Out[71]: 
            array([['NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface'],
                   ['NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface'],
                   ['NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface'],
                   ['NNVTMCPPMT_PMT_20inch_mcp_opsurface']], dtype='<U124')

            In [72]: np.c_[sun[np.array([37])]]
            Out[72]: array([['NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf']], dtype='<U124')

            In [1]: a = np.load("NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/REFLECTIVITY.npy") ; a 
            Out[2]: 
            array([[0.  , 0.92],
                   [0.  , 0.92]])

            In [3]: a[:,0]*1e6
            Out[3]: array([ 1.55, 15.5 ])


            In [1]: a = np.load("NNVTMCPPMT_PMT_20inch_mcp_opsurface/REFLECTIVITY.npy")
            Out[2]: 
            array([[0., 0.],
                   [0., 0.]])

            In [4]: a[:,0]*1e6
            Out[4]: array([ 1.55, 15.5 ])

            ## TODO: check code for this osur + isur 

            epsilon:surface blyth$ cat NNVTMCPPMT_PMT_20inch_mcp_opsurface/NPFold_meta.txt
            OpticalSurfaceName:mcpOpSurface
            pv1:NNVTMCPPMT_PMT_20inch_inner_phys
            pv2:NNVTMCPPMT_PMT_20inch_mcp_phy
            type:Border

            epsilon:surface blyth$ cat NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt 
            OpticalSurfaceName:@NNVTMCPPMT_PMT_20inch_Mirror_opsurf
            lv:NNVTMCPPMT_PMT_20inch_inner_log
            type:Skin
            epsilon:surface blyth$ 




           [   590,   3218,     18,     -1,     -1,      3,     18,    130,     -1,      3],
           [   590,   3808,     18,     -1,     -1,      3,     18,    131,     -1,      3],
           [   590,   4398,     18,     -1,     -1,      3,     18,    132,     -1,      3],

            In [73]:  mtn[np.array([18,3])]
            Out[73]: array(['Water', 'Steel'], dtype='<U18')

            In [74]: np.c_[sun[np.array([130,131,132])]]
            Out[74]: 
            array([['Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lSteel_phys__Steel'],
                   ['Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lFasteners_phys__Steel'],
                   ['Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lUpper_phys__Steel']], dtype='<U124')



           [ 25600, 108039,     18,     -1,     -1,      3,     18,    146,     -1,      3],

            In [78]: np.c_[sun[np.array([146])]]
            Out[78]: array([['Implicit__RINDEX__PMT_3inch_log_phys__Water__NoRINDEX__PMT_3inch_cntr_phys__Steel']], dtype='<U124')


           [   370,   2628,     18,     33,     33,     13,     18,    128,     33,     13],
           [   220,   2998,     18,     34,     34,     13,     18,    129,     34,     13],

            In [75]:  mtn[np.array([18,13])]
            Out[75]: array(['Water', 'StrutSteel'], dtype='<U18')

            In [76]: np.c_[sun[np.array([33,34])]]
            Out[76]: 
            array([['StrutAcrylicOpSurface'],
                   ['Strut2AcrylicOpSurface']], dtype='<U124')

            ## CURIOUS : surface with ABSLENGTH 

            epsilon:surface blyth$ l Strut*/*.npy
            8 -rw-rw-r--  1 blyth  staff  192 Jul  4 21:49 Strut2AcrylicOpSurface/ABSLENGTH.npy
            8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 Strut2AcrylicOpSurface/REFLECTIVITY.npy
            8 -rw-rw-r--  1 blyth  staff  192 Jul  4 21:49 StrutAcrylicOpSurface/ABSLENGTH.npy
            8 -rw-rw-r--  1 blyth  staff  160 Jul  4 21:49 StrutAcrylicOpSurface/REFLECTIVITY.npy


            In [77]: np.c_[sun[np.array([128,129])]]
            Out[77]: 
            array([['Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lSteel_phys__StrutSteel'],
                   ['Implicit__RINDEX__pInnerWater__Water__NoRINDEX__lSteel2_phys__StrutSteel']], dtype='<U124')



           [  4997,   4988,     18,     36,     36,     15,     18,    133,     36,     15],
           [ 12615,   5004,     18,     38,     38,     15,     18,    141,     38,     15],

            In [79]:  mtn[np.array([18,15])]
            Out[79]: array(['Water', 'CDReflectorSteel'], dtype='<U18')

            In [80]: np.c_[sun[np.array([36,133,38,141])]]
            Out[80]: 
            array([['HamamatsuMaskOpticalSurface'],
                   ['Implicit__RINDEX__pLPMT_Hamamatsu_R12860__Water__NoRINDEX__HamamatsuR12860pMaskTail__CDReflectorSteel'],
                   ['NNVTMaskOpticalSurface'],
                   ['Implicit__RINDEX__pLPMT_NNVT_MCPPMT__Water__NoRINDEX__NNVTMCPPMTpMaskTail__CDReflectorSteel']], dtype='<U124')



           [     1, 133639,     18,     39,     39,      3,     18,    147,     39,      3],

            In [81]: mtn[np.array([18,3])]
            Out[81]: array(['Water', 'Steel'], dtype='<U18')

            In [82]: np.c_[sun[np.array([39,147])]]
            Out[82]: 
            array([['Steel_surface'],
                   ['Implicit__RINDEX__lLowerChimney_phys__Water__NoRINDEX__pLowerChimneySteel__Steel']], dtype='<U124')



           [    10,    507,     19,     -1,     -1,     10,     19,     54,     -1,     10],
           [    30,    517,     19,     -1,     -1,     10,     19,     55,     -1,     10],
           [    30,    547,     19,     -1,     -1,     10,     19,     56,     -1,     10],
           [    30,    577,     19,     -1,     -1,     10,     19,     57,     -1,     10],
           [    30,    607,     19,     -1,     -1,     10,     19,     58,     -1,     10],
           [    30,    637,     19,     -1,     -1,     10,     19,     59,     -1,     10],

            In [83]:  mtn[np.array([19,10])]
            Out[83]: array(['vetoWater', 'LatticedShellSteel'], dtype='<U18')

            In [84]: np.c_[sun[np.array([54,55,56,57,58,59])]]
            Out[84]: 
            array([['Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up10_up11_HBeam_phys__LatticedShellSteel'],
                   ['Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up09_up10_HBeam_phys__LatticedShellSteel'],
                   ['Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up08_up09_HBeam_phys__LatticedShellSteel'],
                   ['Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up07_up08_HBeam_phys__LatticedShellSteel'],
                   ['Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up06_up07_HBeam_phys__LatticedShellSteel'],
                   ['Implicit__RINDEX__pOuterWaterPool__vetoWater__NoRINDEX__GLw1.up05_up06_HBeam_phys__LatticedShellSteel']], dtype='<U124')



