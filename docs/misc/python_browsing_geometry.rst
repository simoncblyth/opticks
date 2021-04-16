python_browsing_geometry
==========================



GNodeLib.py : Find the CE of some Chimney volumes 
---------------------------------------------------

Want to find the current nidx corresponding to some old logging::

    2019-04-21 00:27:12.438 FATAL [107202] [OpticksAim::setTarget@121] OpticksAim::setTarget  based on CenterExtent from m_mesh0  target 352851 aim 1 ce 0.0000,0.0000,19785.0000,1965.0000


With the below find that the nidx is now : 304632::


    CE
    array([    0.,     0., 19785.,  1965.], dtype=float32)


::

    epsilon:ana blyth$ GNodeLib.py --ulv --sli 0:None 
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.ulv found 131 unique LV names
    GLb1.bt02_HBeam0x34c1e00
    GLb1.bt05_HBeam0x34cf620
    GLb1.bt06_HBeam0x34d1e20
    GLb1.bt07_HBeam0x34d4620
    ..

    epsilon:ana blyth$ GNodeLib.py --ulv --sli 0:None  | grep Chimney 
    lLowerChimney0x4ee4270
    lLowerChimneyAcrylic0x4ee4490
    lLowerChimneyLS0x4ee46a0
    lLowerChimneySteel0x4ee48c0
    lUpperChimney0x4ee1f50
    lUpperChimneyLS0x4ee2050
    lUpperChimneySteel0x4ee2160
    lUpperChimneyTyvek0x4ee2270
    epsilon:ana blyth$ 

    epsilon:ana blyth$ GNodeLib.py --lv lLowerChimney0x4ee4270
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.lv:lLowerChimney0x4ee4270 matched 1 nodes 
    slice 0:10:1 
    [304632]
    ### Node idx:304632 

    TR
    array([[    1.,     0.,     0.,     0.],
           [    0.,     1.,     0.,     0.],
           [    0.,     0.,     1.,     0.],
           [    0.,     0., 19785.,     1.]], dtype=float32)

    BB
    array([[ -520.,  -520., 17820.,     1.],
           [  520.,   520., 21750.,     1.]], dtype=float32)

    ID
    array([ 304632,    3080, 7733270,       0], dtype=uint32)

    NI
    array([    96,     50, 304632,  67841], dtype=uint32)

    CE
    array([    0.,     0., 19785.,  1965.], dtype=float32)

    PV
    b'lLowerChimney_phys0x4ee5e60'

    LV
    b'lLowerChimney0x4ee4270'

    epsilon:ana blyth$ 




