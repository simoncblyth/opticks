
lvidx_65_69_56_tube_deltaphi
===============================


Are the top 3 biggies all tube deltaphi ?
--------------------------------------------


::

    2017-07-06 16:09:32.174 INFO  [3663887] [GScene::compareMeshes_GMeshBB@436] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 500
       3869.75               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    507 intersection cylinder 
       3407.72               SstBotCirRibBase0xc26e2d0 lvidx  65 nsp   1212 difference cylinder box3 
       2074.65               SstTopCirRibBase0xc264f78 lvidx  69 nsp   1728 intersection cylinder box3 



Looks like lvid 56 too
-------------------------

::

    simon:tmp blyth$ grep deltaphi g4_00.gdml | grep -v deltaphi=\"360\" 
        <tube aunit="deg" deltaphi="44.6352759021238" lunit="mm" name="BlackCylinder0xc1762e8" rmax="2262.15" rmin="2259.15" startphi="0" z="997"/>
        <tube aunit="deg" deltaphi="45" lunit="mm" name="SstBotCirRibPri0xc26d4e0" rmax="2000" rmin="1980" startphi="0" z="430"/>
        <tube aunit="deg" deltaphi="45" lunit="mm" name="SstTopCirRibPri0xc2648b8" rmax="1220" rmin="1200" startphi="0" z="231.89"/>
        <sphere aunit="deg" deltaphi="180" deltatheta="180" lunit="mm" name="UpperAcrylicHemisphere0xc0b2ac0" rmax="10.035" rmin="0" startphi="0" starttheta="0"/>
        <sphere aunit="deg" deltaphi="180" deltatheta="180" lunit="mm" name="LowerAcrylicHemisphere0xc0b2be8" rmax="10.035" rmin="0" startphi="0" starttheta="0"/>
    simon:tmp blyth$ 


::

  799     <subtraction name="RadialShieldUnit0xc3d7da8">
  800       <first ref="BlackCylinder-ChildForRadialShieldUnit0xc3d8628"/>
  801       <second ref="PmtHole60xc3d7cb8"/>
  802       <position name="RadialShieldUnit0xc3d7da8_pos" unit="mm" x="1797.86532031977" y="1370.48119742355" z="-250"/>
  803       <rotation name="RadialShieldUnit0xc3d7da8_rot" unit="deg" x="-37.3176379510619" y="90" z="0"/>
  804     </subtraction>


