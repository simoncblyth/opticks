old_workflow_finds_extra_lSteel_skin_surface
===============================================

TODO : work out why get one extra bnd in oldoptical
------------------------------------------------------

::

    Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/Steel

Note already have 2 very similar bnd, that are in agreement between old and new::

    Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel
    Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutSteel

::

    In [7]: np.c_[obn[20:30],bn[20:30]]       ## gets out of step at 25 
    Out[7]: 
    array([['Acrylic///LS', 'Acrylic///LS'],
           ['LS///Acrylic', 'LS///Acrylic'],
           ['LS///PE_PA', 'LS///PE_PA'],
           ['Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel', 'Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel'],
           ['Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutSteel', 'Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutSteel'],
           ['Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/Steel', 'Water///Steel'],
           ['Water///Steel', 'Water///Water'],
           ['Water///Water', 'Water///AcrylicMask'],
           ['Water///AcrylicMask', 'Water/HamamatsuMaskOpticalSurface/HamamatsuMaskOpticalSurface/CDReflectorSteel'],
           ['Water/HamamatsuMaskOpticalSurface/HamamatsuMaskOpticalSurface/CDReflectorSteel', 'Water///Pyrex']], dtype='<U122')

    In [8]:                   










::

    In [4]: op.shape
    Out[4]: (52, 4, 4)

    In [5]: oop.shape
    Out[5]: (53, 4, 4)


::

    epsilon:tests blyth$ jgr StrutAcrylicOpSurface
    ./Simulation/DetSimV2/CentralDetector/src/StrutAcrylicConstruction.cc:    new G4LogicalSkinSurface("StrutAcrylicOpSurface", logicStrut, strut_optical_surface);
    epsilon:junosw blyth$ 


::

    jcv StrutAcrylicConstruction



    166 void
    167 StrutAcrylicConstruction::initMaterials() {
    168     Steel = G4Material::GetMaterial("StrutSteel");
    169 }
    170 
    171 void
    172 StrutAcrylicConstruction::makeStrutLogical() {
    173         solidStrut = new G4Tubs(
    174                         "sStrut",
    175                         m_radStrut_in,
    176                         m_radStrut_out,
    177                         m_lengthStrut/2,
    178                         0*deg,
    179                         360*deg);
    180 
    181 
    182         logicStrut = new G4LogicalVolume(
    183                         solidStrut,
    184                         Steel,
    185                         "lSteel",
    186                         0,
    187                         0,
    188                         0);




    198 void
    199 StrutAcrylicConstruction::makeStrutOpSurface() {
    200     G4OpticalSurface *strut_optical_surface = new G4OpticalSurface("opStrutAcrylic");
    201     strut_optical_surface->SetMaterialPropertiesTable(Steel->GetMaterialPropertiesTable());
    202     strut_optical_surface->SetModel(unified);
    203     strut_optical_surface->SetType(dielectric_metal);
    204     strut_optical_surface->SetFinish(ground);
    205     strut_optical_surface->SetSigmaAlpha(0.2);
    206 
    207     new G4LogicalSkinSurface("StrutAcrylicOpSurface", logicStrut, strut_optical_surface);
    208 }


Note StrutAcrylicConstruction uses StrutSteel so does not correspond to the extra bnd which is just "Steel".

TODO :  add some debug that std::raise(SIGINT) on adding that bnd. 

* thats in X4PhysicalVolume::addBoundary


::

    In [18]: oop[25]
    Out[18]:
    array([[19,  0,  0,  0],
           [34,  0,  3, 20],
           [34,  0,  3, 20],
           [ 4,  0,  0,  0]], dtype=int32)

    In [19]: oop[:,0,0].min()
    Out[19]: 1

    In [20]: np.array(t.oldmat_names)[19-1]
    Out[20]: 'Water'

    In [21]: np.array(t.oldmat_names)[4-1]
    Out[21]: 'Steel'

    In [22]: np.array(t.oldsur_names)[34-1]
    Out[22]: 'StrutAcrylicOpSurface'




::

    (gdb) bt
    #0  0x00007ffff741e4fb in raise () from /lib64/libpthread.so.0
    #1  0x00007fffd1997b39 in GBndLib::addBoundary (this=0xd501440, omat=0x1787f770 "Water", 
        osur=0x107b3730 "StrutAcrylicOpSurface", isur=0x107b3730 "StrutAcrylicOpSurface", imat=0x1787f7b0 "Steel")
        at /data/blyth/junotop/opticks/ggeo/GBndLib.cc:508
    #2  0x00007fffd240cb62 in X4PhysicalVolume::addBoundary (this=0x7fffffff4090, pv=0x5b18270, pv_p=0x5a9bba0)
        at /data/blyth/junotop/opticks/extg4/X4PhysicalVolume.cc:1695
    #3  0x00007fffd240ce39 in X4PhysicalVolume::convertNode (this=0x7fffffff4090, pv=0x5b18270, parent=0x176475a0, depth=6, 
        pv_p=0x5a9bba0, recursive_select=@0x7fffffff378f: false) at /data/blyth/junotop/opticks/extg4/X4PhysicalVolume.cc:1771
    #4  0x00007fffd240bc17 in X4PhysicalVolume::convertStructure_r (this=0x7fffffff4090, pv=0x5b18270, parent=0x176475a0, 
        depth=6, sibdex=591, parent_nidx=67846, parent_pv=0x5a9bba0, recursive_select=@0x7fffffff378f: false)
        at /data/blyth/junotop/opticks/extg4/X4PhysicalVolume.cc:1452
    #5  0x00007fffd240c02d in X4PhysicalVolume::convertStructure_r (this=0x7fffffff4090, pv=0x5a9bba0, parent=0x17644a40, 
        depth=5, sibdex=0, parent_nidx=67845, parent_pv=0x5a9bd50, recursive_select=@0x7fffffff378f: false)
        at /data/blyth/junotop/opticks/extg4/X4PhysicalVolume.cc:1515
    #6  0x00007fffd240c02d in X4PhysicalVolume::convertStructure_r (this=0x7fffffff4090, pv=0x5a9bd50, parent=0x17083960, 
        depth=4, sibdex=2120, parent_nidx=65724, parent_pv=0x5a9a200, recursive_select=@0x7fffffff378f: false)
        at /data/blyth/junotop/opticks/extg4/X4PhysicalVolume.cc:1515
    #7  0x00007fffd240c02d in X4PhysicalVolume::convertStructure_r (this=0x7fffffff4090, pv=0x5a9a200, parent=0x17082a30, 
        depth=3, sibdex=0, parent_nidx=65723, parent_pv=0x5a04720, recursive_select=@0x7fffffff378f: false)
        at /data/blyth/junotop/opticks/extg4/X4PhysicalVolume.cc:1515
    #8  0x00007fffd240c02d in X4PhysicalVolume::convertStructure_r (this=0x7fffffff4090, pv=0x5a04720, parent=0x17081c50, 
        depth=2, sibdex=0, parent_nidx=65722, parent_pv=0x5a04780, recursive_select=@0x7fffffff378f: false)


::


    (gdb) p _pv_p
    $14 = 0x1787f730 "pInnerWater0x5a9bba0"

    (gdb) p _lv_p
    $15 = 0x1787f8d0 "lInnerWater0x5a9ae00"



    (gdb) p _pv
    $13 = 0x1787f710 "lSteel_phys0x5b18270"

    (gdb) p _lv
    $16 = 0x1787f8b0 "lSteel0x5b181c0"


    (gdb) p omat
    $17 = 0x1787f770 "Water"

    (gdb) p imat
    $18 = 0x1787f7b0 "Steel"



::

    X4PhysicalVolume::addBoundary IsDebugBoundary  omat Water osur StrutAcrylicOpSurface isur StrutAcrylicOpSurface imat Steel
    X4PhysicalVolume::addBoundary
     _pv        lSteel_phys0x5b18640
     _pv_p      pInnerWater0x5a9bf70
     _lv        lSteel0x5b18590
     _lv_p      lInnerWater0x5a9b1d0
     _so_name   sStrutBallhead
     _so_p_name sInnerWater
    [New Thread 0x7fff9ffff700 (LWP 114833)]

    Program received signal SIGINT, Interrupt.


    (gdb) c
    Continuing.
    X4PhysicalVolume::addBoundary IsDebugBoundary  omat Water osur StrutAcrylicOpSurface isur StrutAcrylicOpSurface imat Steel
    X4PhysicalVolume::addBoundary
     _pv        lSteel_phys0x5b186d0
     _pv_p      pInnerWater0x5a9bf70
     _lv        lSteel0x5b18590
     _lv_p      lInnerWater0x5a9b1d0
     _so_name   sStrutBallhead
     _so_p_name sInnerWater


::

    epsilon:opticks blyth$ jgr sStrutBallhead
    ./Simulation/DetSimV2/CentralDetector/src/StrutBallheadAcrylicConstruction.cc:        solidStrut  =new  G4Orb("sStrutBallhead",
    epsilon:junosw blyth$ 


HMM there is no optical surface here... its looking like 
the error is an incorrectly detected optical surface in old workflow ?.

Maybe there is some problem of different LV with the same name "lSteel" 
that causes skin surface confusion in X4/GGeo ?::

    113 void
    114 StrutBallheadAcrylicConstruction::initMaterials() {
    115     Steel = G4Material::GetMaterial("Steel");
    116 }
    117 
    118 void
    119 StrutBallheadAcrylicConstruction::makeStrutLogical() {
    120         solidStrut  =new  G4Orb("sStrutBallhead",
    121                                 m_rad);
    122 
    123 
    124         logicStrut = new G4LogicalVolume(
    125                         solidStrut,
    126                         Steel,
    127                         "lSteel",
    128                         0,
    129                         0,
    130                         0);



The new way bases material index on position of G4Material pointer in a vector::

    486 
    487     int omat = stree::GetPointerIndex<G4Material>(      materials, border.omat_);
    488     int osur = stree::GetPointerIndex<G4LogicalSurface>(surfaces,  border.osur_);
    489     int isur = stree::GetPointerIndex<G4LogicalSurface>(surfaces,  border.isur_);
    490     int imat = stree::GetPointerIndex<G4Material>(      materials, border.imat_);
    491 
    492     int4 bd = {omat, osur, isur, imat } ;
    493 


    0712 template<typename T>
     713 inline int stree::GetPointerIndex( const std::vector<const T*>& vec, const T* obj) // static
     714 {
     715     if( obj == nullptr || vec.size() == 0 ) return -1 ;
     716     size_t idx = std::distance( vec.begin(), std::find(vec.begin(), vec.end(), obj ));
     717     return idx < vec.size() ? int(idx) : -1 ;
     718 }

Old way might be based on the string name of the material ?

HMM : mat unlikely to go wrong as simpler. More likely a problem 
with surface assignment.




::

    U4TreeBorder::desc 
     omat Water
     imat Steel
     osolid sInnerWater
     isolid sStrutBallhead
     is_debug_border YES
     U4Tree::initNodes_r
     (omat,osur,isur,imat) (    18,    -1,    -1,     3) 





     _so_p_name sInnerWater
    X4PhysicalVolume::addBoundary IsDebugBoundary  omat Water osur StrutAcrylicOpSurface isur StrutAcrylicOpSurface imat Steel
    X4PhysicalVolume::addBoundary
     _pv        lSteel_phys0x5b19530
     _pv_p      pInnerWater0x5a9c6d0
     _lv        lSteel0x5b18cf0
     _lv_p      lInnerWater0x5a9b930
     _so_name   sStrutBallhead
     _so_p_name sInnerWater
    X4PhysicalVolume::addBoundary IsDebugBoundary  omat Water osur StrutAcrylicOpSurface isur StrutAcrylicOpSurface imat Steel
    X4PhysicalVolume::addBoundary
     _pv        lSteel_phys0x5b19610
     _pv_p      pInnerWater0x5a9c6d0
     _lv        lSteel0x5b18cf0
     _lv_p      lInnerWater0x5a9b930
     _so_name   sStrutBallhead
     _so_p_name sInnerWater




YEP : skin surface is found based ONLY on the name of the lv, so duplicate named LV 
may explain the issue:: 

    0885 GSkinSurface* X4PhysicalVolume::findSkinSurfaceOK( const G4LogicalVolume* const lv) const
     886 {
     887     const char* _lv = X4::Name( lv ) ;
     888     GSkinSurface* sk = _lv ? m_slib->findSkinSurface(_lv) : nullptr ;
     889     return sk ;
     890 }

    1469 GSkinSurface* GSurfaceLib::findSkinSurface(const char* lv) const
    1470 {
    1471     GSkinSurface* ss = NULL ;
    1472     for(unsigned int i=0 ; i < m_skin_surfaces.size() ; i++ )
    1473     {
    1474          GSkinSurface* s = m_skin_surfaces[i];
    1475          if(s->matches(lv))   
    1476          {
    1477             ss = s ;
    1478             break ; 
    1479          }  
    1480     }    
    1481     return ss ;
    1482 }   

    076 bool GSkinSurface::matches(const char* lv) const
     77 {
     78     return strcmp(m_skinsurface_vol, lv) == 0;
     79 }

    057 void GSkinSurface::setSkinSurface(const char* vol)
     58 {
     59     m_skinsurface_vol = strdup(vol);
     60 }
     61 


::

    epsilon:tests blyth$ opticks-f setSkinSurface
    ./extg4/X4LogicalSkinSurface.cc:    dst->setSkinSurface(  X4::BaseNameAsis(lv) ) ; 
    ./ggeo/GSkinSurface.cc:void GSkinSurface::setSkinSurface(const char* vol)
    ./ggeo/GSurfaceLib.cc:    surf->setSkinSurface();
    ./ggeo/GSkinSurface.hh:      void setSkinSurface(const char* vol);
    ./ggeo/GPropertyMap.cc:void GPropertyMap<T>::setSkinSurface()  
    ./ggeo/GPropertyMap.hh:      void setSkinSurface(); 
    epsilon:opticks blyth$ 


::

    inner2_phys bpv2 PMT_3inch_cntr_phys .
     index : 32 is_sensor : N type :        bordersurface name :                           UpperChimneyTyvekSurface bpv1 pUpperChimneyLS bpv2 pUpperChimneyTyvek .
     index : 33 is_sensor : N type :          skinsurface name :                              StrutAcrylicOpSurface sslv lSteel .
     index : 34 is_sensor : N type :          skinsurface name :                             Strut2AcrylicOpSurface sslv lSteel2 .
     index : 35 is_sensor : N type :          skinsurface name : HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf sslv HamamatsuR12860_PMT_20inch_inner_log .
     index : 36 is_sensor : N type :          skinsurface name :                        HamamatsuMaskOpticalSurface sslv HamamatsuR12860lMaskTai

