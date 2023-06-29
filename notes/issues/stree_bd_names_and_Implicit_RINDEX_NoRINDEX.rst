stree_bd_names_and_Implicit_RINDEX_NoRINDEX
==============================================


Looks like the Implicit_RINDEX_NoRINDEX fixup not yet in direct workflow ?
--------------------------------------------------------------------------------

::

    epsilon:opticks blyth$ opticks-f Implicit_RINDEX_NoRINDEX
    ./extg4/X4PhysicalVolume.cc:    static const char* IMPLICIT_PREFIX = "Implicit_RINDEX_NoRINDEX" ; 
    ./sysrap/SBnd.h:    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air
    ./sysrap/SBnd.h:    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air
    ./ggeo/GSurfaceLib.cc:    ss << "Implicit_RINDEX_NoRINDEX_" << spv1 << "_" << spv2 ;  
    epsilon:opticks blyth$ 


Q: How to add perfect surface in direct workflow ?
-----------------------------------------------------

1st: need to develop the addition of standard surfaces







X4PhysicalVolume::convertImplicitSurfaces_r
---------------------------------------------

::

     526 /**
     527 X4PhysicalVolume::convertImplicitSurfaces_r
     528 ---------------------------------------------
     529 
     530 Recursively look for "implicit" surfaces, eg from the mis-use of Geant4 NoRINDEX 
     531 behaviour to effectively mimic surface absorption without actually setting an 100% 
     532 absorption property. See::
     533 
     534    g4-cls G4OpBoundaryProcess
     535 
     536 Photons at the border from a transparent material with RINDEX (eg Water) 
     537 to a NoRINDEX material (eg Tyvek) get fStopAndKill-ed by G4OpBoundaryProcess::PostStepDoIt
     538 
     539 RINDEX_NoRINDEX
     540     parent with RINDEX but daughter without, 
     541     there tend to be large numbers of these 
     542     eg from every piece of metal in Air or Water etc..
     543 
     544 NoRINDEX_RINDEX
     545     parent without RINDEX and daughter with,
     546     there tend to be only a few of these : but they are optically important as "containers" 
     547     eg Tyvek or Rock containing Water   
     548 
     549 **/
     550 
     551 void X4PhysicalVolume::convertImplicitSurfaces_r(const G4VPhysicalVolume* const parent_pv, int depth)
     552 {
     553     const G4LogicalVolume* parent_lv = parent_pv->GetLogicalVolume() ;
     554     const G4Material* parent_mt = parent_lv->GetMaterial() ;
     555     const G4String& parent_mtName = parent_mt->GetName();
     556 




GSurfaceLib::addImplicitBorderSurface_RINDEX_NoRINDEX
--------------------------------------------------------


::

     717 void GSurfaceLib::addImplicitBorderSurface_RINDEX_NoRINDEX( const char* pv1, const char* pv2 )
     718 {
     719     GBorderSurface* prior_bs = findBorderSurface(pv1, pv2);
     720     if( prior_bs != nullptr )
     721     {   
     722         LOG(fatal) 
     723             << " pv1 " << pv1
     724             << " pv2 " << pv2  
     725             << " prior_bs " << prior_bs
     726             << " there is a prior GBorderSurface from pv1->pv2 "
     727             ;
     728         assert(0);
     729     }
     730 
     731 
     732     std::string spv1 = SGDML::Strip(pv1);
     733     std::string spv2 = SGDML::Strip(pv2);
     734     std::stringstream ss ;
     735     ss << "Implicit_RINDEX_NoRINDEX_" << spv1 << "_" << spv2 ;
     736     std::string s = ss.str();
     737     const char* name = s.c_str();
     738 
     739     // placeholders
     740     const char* type = "1" ;
     741     const char* model = "1" ;
     742     const char* finish = "1" ;
     743     const char* value = "1" ;
     744     GOpticalSurface* os = new GOpticalSurface(name, type, model, finish, value);
     745 
     746     unsigned index = 2000 ; // TODO: eliminate this index, or automate it 
     747     GBorderSurface* bs = new GBorderSurface( name, index, os );
     748     bs->setBorderSurface(pv1, pv2);
     749 
     750     double detect_ = 0.f ;




Issue
---------

::

    epsilon:~ blyth$ cd /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/
    epsilon:SSim blyth$ l
    total 20416
        0 drwxr-xr-x   6 blyth  staff       192 Jun 15 19:34 jpmt
        8 -rw-rw-r--   1 blyth  staff        34 Jun 15 19:34 NPFold_index.txt
        8 -rw-rw-r--   1 blyth  staff       109 Jun 15 19:34 icdf_meta.txt
        8 -rw-rw-r--   1 blyth  staff         3 Jun 15 19:34 icdf_names.txt
      200 -rw-rw-r--   1 blyth  staff     98432 Jun 15 19:34 icdf.npy
        8 -rw-rw-r--   1 blyth  staff      3520 Jun 15 19:34 optical.npy
        8 -rw-rw-r--   1 blyth  staff        69 Jun 15 19:34 bnd_meta.txt
        8 -rw-rw-r--   1 blyth  staff      2734 Jun 15 19:34 bnd_names.txt
    20168 -rw-rw-r--   1 blyth  staff  10325392 Jun 15 19:34 bnd.npy
        0 drwxr-xr-x  28 blyth  staff       896 Jun  7 14:17 stree
        0 drwxr-xr-x  12 blyth  staff       384 Jun  7 14:17 .
        0 drwxr-xr-x  13 blyth  staff       416 Jun  7 14:17 ..
    epsilon:SSim blyth$ 



The other difference looks like geometry version inconsistency ?::

    epsilon:SSim blyth$ diff bnd_names.txt stree/bd_names.txt 
    4c4
    < Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air
    ---
    > Rock///Air
    6d5
    < Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air
    26d24
    < Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/Steel
    epsilon:SSim blyth$ 




    epsilon:SSim blyth$ diff -y bnd_names.txt stree/bd_names.txt 
    Galactic///Galactic						Galactic///Galactic
    Galactic///Rock							Galactic///Rock
    Rock///Galactic							Rock///Galactic
    Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air	      |	Rock///Air
    Rock///Rock							Rock///Rock
    Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air	      <
    Air///Steel							Air///Steel
    Air///Air							Air///Air
    Air///LS							Air///LS
    Air///Tyvek							Air///Tyvek
    Air///Aluminium							Air///Aluminium
    Aluminium///Adhesive						Aluminium///Adhesive
    Adhesive///TiO2Coating						Adhesive///TiO2Coating
    TiO2Coating///Scintillator					TiO2Coating///Scintillator
    Rock///Tyvek							Rock///Tyvek
    Tyvek//VETOTyvekSurface/vetoWater				Tyvek//VETOTyvekSurface/vetoWater
    vetoWater///LatticedShellSteel					vetoWater///LatticedShellSteel
    vetoWater/CDTyvekSurface//Tyvek					vetoWater/CDTyvekSurface//Tyvek
    Tyvek//CDInnerTyvekSurface/Water				Tyvek//CDInnerTyvekSurface/Water
    Water///Acrylic							Water///Acrylic
    Acrylic///LS							Acrylic///LS
    LS///Acrylic							LS///Acrylic
    LS///PE_PA							LS///PE_PA
    Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel	Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel
    Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutStee	Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutStee
    Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/Steel	      <
    Water///Steel							Water///Steel
    Water///Water							Water///Water
    Water///AcrylicMask						Water///AcrylicMask
    Water/HamamatsuMaskOpticalSurface/HamamatsuMaskOpticalSurface	Water/HamamatsuMaskOpticalSurface/HamamatsuMaskOpticalSurface
    Water///Pyrex							Water///Pyrex
    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/	Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/
    Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/Hama	Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/Hama
    Vacuum/HamamatsuR12860_PMT_20inch_outer_edge_opsurface/Hamama	Vacuum/HamamatsuR12860_PMT_20inch_outer_edge_opsurface/Hamama
    Vacuum/HamamatsuR12860_PMT_20inch_inner_edge_opsurface/Hamama	Vacuum/HamamatsuR12860_PMT_20inch_inner_edge_opsurface/Hamama
    Vacuum/HamamatsuR12860_PMT_20inch_inner_ring_opsurface/Hamama	Vacuum/HamamatsuR12860_PMT_20inch_inner_ring_opsurface/Hamama
    Vacuum/HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/Hamam	Vacuum/HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/Hamam
    Vacuum/HamamatsuR12860_PMT_20inch_grid_opsurface/HamamatsuR12	Vacuum/HamamatsuR12860_PMT_20inch_grid_opsurface/HamamatsuR12
    Vacuum/HamamatsuR12860_PMT_20inch_shield_opsurface/HamamatsuR	Vacuum/HamamatsuR12860_PMT_20inch_shield_opsurface/HamamatsuR
    Water/NNVTMaskOpticalSurface/NNVTMaskOpticalSurface/CDReflect	Water/NNVTMaskOpticalSurface/NNVTMaskOpticalSurface/CDReflect
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTM	Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTM
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NNVTMCPPMT_PM	Vacuum/NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NNVTMCPPMT_PM
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NNVTMCPPMT_P	Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NNVTMCPPMT_P
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NNVTMCPPMT_PM	Vacuum/NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NNVTMCPPMT_PM
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_opsurface/NNVTMCPPMT_PMT_20i	Vacuum/NNVTMCPPMT_PMT_20inch_mcp_opsurface/NNVTMCPPMT_PMT_20i
    Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_	Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_
    Pyrex/PMT_3inch_absorb_logsurf2/PMT_3inch_absorb_logsurf1/Vac	Pyrex/PMT_3inch_absorb_logsurf2/PMT_3inch_absorb_logsurf1/Vac
    Water///LS							Water///LS
    Water/Steel_surface/Steel_surface/Steel				Water/Steel_surface/Steel_surface/Steel
    vetoWater///Water						vetoWater///Water
    Pyrex///Pyrex							Pyrex///Pyrex
    Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_p	Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_p
    Pyrex/PMT_20inch_veto_mirror_logsurf2/PMT_20inch_veto_mirror_	Pyrex/PMT_20inch_veto_mirror_logsurf2/PMT_20inch_veto_mirror_
    epsilon:SSim blyth$ 




