sur_oldsur_matching
======================


Issue 1 : oldsur has 6 more surfaces : 2:Implicits 4:perfects
-----------------------------------------------------------------

::

    st
    ./stree_mat_test.sh 

    In [3]: t.sur.shape
    Out[3]: (40, 2, 761, 4)

    In [4]: t.oldsur.shape
    Out[4]: (46, 2, 761, 4)

    In [7]: a = np.array(t.oldsur_names)

    In [8]: b = np.array(t.sur_names)  

    In [9]: a.shape 
    Out[9]: (46,)

    In [10]: b.shape
    Out[10]: (40,)

    In [13]: np.all( a[:20] == b[:20] )
    Out[13]: True

    In [14]: np.all( a[:30] == b[:30] )
    Out[14]: True

    In [15]: np.all( a[:40] == b[:40] )
    Out[15]: True

    In [16]: a.shape
    Out[16]: (46,)

    In [17]: b.shape
    Out[17]: (40,)

    In [18]: a[40:]
    Out[18]: 
    array([
         'Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock', 
         'Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox', 
         'perfectDetectSurface', 
         'perfectAbsorbSurface', 
         'perfectSpecularSurface',
         'perfectDiffuseSurface'], dtype='<U54')


WIP : where are Implicit and perfect surfaces added
-------------------------------------------------------

::

    epsilon:extg4 blyth$ opticks-f Implicit_RINDEX_NoRINDEX
    ./extg4/X4PhysicalVolume.cc:    static const char* IMPLICIT_PREFIX = "Implicit_RINDEX_NoRINDEX" ; 
    ./sysrap/stree.h:* ~/opticks/notes/issues/stree_bd_names_and_Implicit_RINDEX_NoRINDEX.rst
    ./sysrap/SBnd.h:    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air
    ./sysrap/SBnd.h:    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air
    ./ggeo/GSurfaceLib.cc:    ss << "Implicit_RINDEX_NoRINDEX_" << spv1 << "_" << spv2 ;  
    ./u4/U4Tree.h:* see ~/opticks/notes/issues/stree_bd_names_and_Implicit_RINDEX_NoRINDEX.rst
    epsilon:opticks blyth$ 




Issue 2 : unused 2nd payload group filled with -1 in oldsur, 0 in sur
------------------------------------------------------------------------


::

    In [19]: a = t.oldsur

    In [20]: b = t.sur

    In [21]: a.shape
    Out[21]: (46, 2, 761, 4)

    In [22]: b.shape
    Out[22]: (40, 2, 761, 4)

    In [24]: np.all( a[:,1,:] == -1 )
    Out[24]: True

    In [25]: np.all( b[:,1,:] == -1 )
    Out[25]: False

    In [26]: np.all( b[:,1,:] == 0 )
    Out[26]: True


Observation : no significant diff in payload group 0 of the 40 sur in common
-------------------------------------------------------------------------------

::

    In [35]: np.abs( a[:40,0,:,:]-b[:40,0,:,:] ).max()
    Out[35]: 6.100014653676045e-07

