optical_ems_4_getting_too_many_from_non_sensor_Vacuum_Steel_borders
======================================================================

Context
-----------

* from :doc:`QSimTest_shakedown_following_QPMT_extension`


FIXED  : too many ems 4, getting them for non-sensor  Vacuum/Steel borders 
------------------------------------------------------------------------------------

* some surfaces inside the PMT are getting ems 4 


BEFORE::

    st ; ./stree_py_test.sh 


    In [3]: np.where( f.standard.optical[:,:,1] == 4 )
    Out[3]:
    (array([30, 30, 31, 32, 33, 34, 35, 36, 37, 39, 39, 40, 41, 42, 43]),
     array([1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2]))

    In [4]: np.where( f.standard.optical[:,:,1] == 4 )[0]
    Out[4]: array([30, 30, 31, 32, 33, 34, 35, 36, 37, 39, 39, 40, 41, 42, 43])

    In [5]: np.unique(np.where( f.standard.optical[:,:,1] == 4 )[0])
    Out[5]: array([30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43])

    In [6]: f.standard.bnd_names[np.unique(np.where( f.standard.optical[:,:,1] == 4 )[0])]
    Out[6]:
    array(['Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum',

           'Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/HamamatsuR12860_PMT_20inch_outer_edge_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/HamamatsuR12860_PMT_20inch_inner_edge_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/HamamatsuR12860_PMT_20inch_inner_ring_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/HamamatsuR12860_PMT_20inch_grid_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/HamamatsuR12860_PMT_20inch_shield_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',

           'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum',

           'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel',
           'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel'], dtype='<U122')

::

    In [19]: f.standard.optical[np.unique(np.where( f.standard.optical[:,:,1] == 4 )[0])]
    Out[19]: 
    array([[[18,  0,  0,  0],
            [36,  4,  1, 99],
            [36,  4,  1, 99],
            [17,  0,  0,  0]],


           [[17,  0,  0,  0],
            [ 8,  2,  0, 99],
            [36,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [10,  2,  0, 99],
            [36,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [11,  2,  0, 99],
            [36,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [ 9,  2,  0, 99],
            [36,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [12,  2,  0, 99],
            [36,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [13,  2,  0, 99],
            [36,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [14,  2,  0, 99],
            [36,  4,  1, 99],
            [ 4,  0,  0,  0]],


           [[18,  0,  0,  0],
            [38,  4,  1, 99],
            [38,  4,  1, 99],
            [17,  0,  0,  0]],


           [[17,  0,  0,  0],
            [16,  2,  0, 99],
            [38,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [15,  2,  0, 99],
            [38,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [17,  2,  0, 99],
            [38,  4,  1, 99],
            [ 4,  0,  0,  0]],

           [[17,  0,  0,  0],
            [18,  2,  0, 99],
            [38,  4,  1, 99],
            [ 4,  0,  0,  0]]], dtype=int32)


That is based on the first char of the OpticalSurfaceName in surface metadata::

    epsilon:surface blyth$ grep OpticalSurfaceName */NPFold_meta.txt
    CDInnerTyvekSurface/NPFold_meta.txt:OpticalSurfaceName:CDInnerTyvekOpticalSurface
    CDTyvekSurface/NPFold_meta.txt:OpticalSurfaceName:CDTyvekOpticalSurface
    HamamatsuMaskOpticalSurface/NPFold_meta.txt:OpticalSurfaceName:opHamamatsuMask
    HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/NPFold_meta.txt:OpticalSurfaceName:plateOpSurface
    HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/NPFold_meta.txt:OpticalSurfaceName:tubeOpSurface
    HamamatsuR12860_PMT_20inch_grid_opsurface/NPFold_meta.txt:OpticalSurfaceName:gridOpSurface
    HamamatsuR12860_PMT_20inch_inner_edge_opsurface/NPFold_meta.txt:OpticalSurfaceName:outerEdgeOpSurface
    HamamatsuR12860_PMT_20inch_inner_ring_opsurface/NPFold_meta.txt:OpticalSurfaceName:plateOpSurface
    HamamatsuR12860_PMT_20inch_outer_edge_opsurface/NPFold_meta.txt:OpticalSurfaceName:outerEdgeOpSurface

    HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt:OpticalSurfaceName:@HamamatsuR12860_PMT_20inch_Mirror_opsurf

    HamamatsuR12860_PMT_20inch_shield_opsurface/NPFold_meta.txt:OpticalSurfaceName:shieldOpSurface
    NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NPFold_meta.txt:OpticalSurfaceName:edgeOpSurface
    NNVTMCPPMT_PMT_20inch_mcp_opsurface/NPFold_meta.txt:OpticalSurfaceName:mcpOpSurface
    NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NPFold_meta.txt:OpticalSurfaceName:plateOpSurface
    NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NPFold_meta.txt:OpticalSurfaceName:tubeOpSurface

    NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NPFold_meta.txt:OpticalSurfaceName:@NNVTMCPPMT_PMT_20inch_Mirror_opsurf

    NNVTMaskOpticalSurface/NPFold_meta.txt:OpticalSurfaceName:opNNVTMask
    PMT_20inch_mirror_logsurf1/NPFold_meta.txt:OpticalSurfaceName:Mirror_opsurf
    PMT_20inch_mirror_logsurf2/NPFold_meta.txt:OpticalSurfaceName:Mirror_opsurf
    PMT_20inch_photocathode_logsurf1/NPFold_meta.txt:OpticalSurfaceName:Photocathode_opsurf
    PMT_20inch_photocathode_logsurf2/NPFold_meta.txt:OpticalSurfaceName:Photocathode_opsurf


Probably its because of the skin surface on the vacuum, and over enthusiatic surface finding, picking 
up skin surface from parent volume ? 

Need to scrub isur based on absorber status of imat ? 


For imat "Steel" (and other absorbers) the isur is pointless and confusing, 
because there wont be any photons heading out of the Steel. 

Could make that call based on::

    epsilon:stree blyth$ cat mtname_no_rindex_names.txt
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
    epsilon:stree blyth$ 

Issue apparent just from the bnd names, all the isur are confusing and pointless::

    epsilon:standard blyth$ cat bnd_names.txt | grep Steel | grep Vacuum  
    Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/HamamatsuR12860_PMT_20inch_outer_edge_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/HamamatsuR12860_PMT_20inch_inner_edge_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/HamamatsuR12860_PMT_20inch_inner_ring_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/HamamatsuR12860_PMT_20inch_grid_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/HamamatsuR12860_PMT_20inch_shield_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel
    epsilon:standard blyth$ 

After change to U4TreeBorder.h::

    epsilon:standard blyth$ cat bnd_names.txt | grep Steel | grep Vacuum  
    Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface//Steel
    Vacuum/HamamatsuR12860_PMT_20inch_outer_edge_opsurface//Steel
    Vacuum/HamamatsuR12860_PMT_20inch_inner_edge_opsurface//Steel
    Vacuum/HamamatsuR12860_PMT_20inch_inner_ring_opsurface//Steel
    Vacuum/HamamatsuR12860_PMT_20inch_dynode_tube_opsurface//Steel
    Vacuum/HamamatsuR12860_PMT_20inch_grid_opsurface//Steel
    Vacuum/HamamatsuR12860_PMT_20inch_shield_opsurface//Steel
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface//Steel
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface//Steel
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface//Steel
    Vacuum/NNVTMCPPMT_PMT_20inch_mcp_opsurface//Steel
    epsilon:standard blyth$ 

Conversely with Pyrex///Vacuum borders which both have RINDEX both osur and isur are relevant
as photons go both ways::

    epsilon:standard blyth$ cat bnd_names.txt | grep Pyrex | grep Vacuum  
    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    Pyrex/PMT_3inch_absorb_logsurf2/PMT_3inch_absorb_logsurf1/Vacuum
    Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum
    Pyrex/PMT_20inch_veto_mirror_logsurf2/PMT_20inch_veto_mirror_logsurf1/Vacuum
    epsilon:standard blyth$ 

After change to U4TreeBorder.h no difference for these::

    epsilon:standard blyth$ cat bnd_names.txt | grep Pyrex | grep Vacuum 
    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    Pyrex/PMT_3inch_absorb_logsurf2/PMT_3inch_absorb_logsurf1/Vacuum
    Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum
    Pyrex/PMT_20inch_veto_mirror_logsurf2/PMT_20inch_veto_mirror_logsurf1/Vacuum
    epsilon:standard blyth$ 



After change, get the expected::

    In [1]: np.where( f.standard.optical[:,:,1] == 4 )
    Out[1]: (array([30, 30, 39, 39]), array([1, 2, 1, 2]))

    In [2]: np.unique(np.where( f.standard.optical[:,:,1] == 4 ) [0] )
    Out[2]: array([30, 39])

    In [3]: f.standard.bnd_names[np.unique(np.where( f.standard.optical[:,:,1] == 4 )[0]))
    Out[3]: 
    array(['Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum',
           'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum'], dtype='<U122')

    In [5]: f.standard.optical[np.unique(np.where( f.standard.optical[:,:,1] == 4 )[0])]
    Out[5]: 
    array([[[18,  0,  0,  0],
            [36,  4,  1, 99],
            [36,  4,  1, 99],
            [17,  0,  0,  0]],

           [[18,  0,  0,  0],
            [38,  4,  1, 99],
            [38,  4,  1, 99],
            [17,  0,  0,  0]]], dtype=int32)

Also the correspondence between Payload X and Y is as expected::

    In [12]: np.all(f.standard.optical[np.where(f.standard.optical[:,:,0] == 0)][:,1] == 1)
    Out[12]: True

    In [13]: np.all(f.standard.optical[np.where(f.standard.optical[:,:,1] == 1)][:,0] == 0)
    Out[13]: True






WIP : Remove confusing (and pointless) isur where imat is absorber
---------------------------------------------------------------------

::

     580 inline int U4Tree::initNodes_r(
     581     const G4VPhysicalVolume* const pv,
     582     const G4VPhysicalVolume* const pv_p,
     583     int depth,
     584     int sibdex,
     585     int parent )
     586 {
     587     // preorder visit before recursive call 
     588 
     589     U4TreeBorder border(st, num_surfaces, pv, pv_p) ;
     590   
     591     int omat = stree::GetPointerIndex<G4Material>(      materials, border.omat_);
     592     int osur = stree::GetPointerIndex<G4LogicalSurface>(surfaces,  border.osur_);
     593     int isur = stree::GetPointerIndex<G4LogicalSurface>(surfaces,  border.isur_);
     594     int imat = stree::GetPointerIndex<G4Material>(      materials, border.imat_);
     595 


* U4TreeBorder already has i_rindex, so use that to disable isur 

U4TreeBorder.h::

    114     o_rindex(U4Mat::GetRINDEX( omat_ )),
    115     osur_(U4Surface::Find( pv_p, pv )),    // look for border or skin surface
    116     isur_( i_rindex == nullptr ? nullptr : U4Surface::Find( pv  , pv_p )), // disable isur from absorbers without RINDEX
    117     implicit_idx(-1),
    118     implicit_isur(i_rindex != nullptr && o_rindex == nullptr),   
    119     implicit_osur(o_rindex != nullptr && i_rindex == nullptr)
    120 {   
    121 }



