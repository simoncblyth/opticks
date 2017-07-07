
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



NCylinder : how to do phi segment SDF ? think 2 cutting planes
-----------------------------------------------------------------

* brought NSlab up to scratch 
* tested slicing by slab intersects in tboolean-cyslab

::

    1385 tboolean-cyslab(){ TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- ; }
    1386 tboolean-cyslab-(){  $FUNCNAME- | python $* ; } 
    1387 tboolean-cyslab--(){ cat << EOP 
    1388 import numpy as np
    1389 from opticks.ana.base import opticks_main
    1390 from opticks.analytic.csg import CSG  
    1391 args = opticks_main(csgpath="$TMP/$FUNCNAME")
    1392 
    1393 CSG.boundary = args.testobject
    1394 CSG.kwa = dict(poly="IM", resolution="50")
    1395 
    1396 container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx="20" )
    1397   
    1398 ca = CSG("cylinder", param=[0,0,0,500], param1=[-100,100,0,0] )
    1399 cb = CSG("cylinder", param=[0,0,0,400], param1=[-101,101,0,0] )
    1400 cy = ca - cb 
    1401 
    1402 
    1403 sa = CSG("slab", param=[1,1,0,0],param1=[0,501,0,0] )  # normalization done in NSlab.hpp/init_slab
    1404 sb = CSG("slab", param=[-1,1,0,0],param1=[0,501,0,0] )  # normalization done in NSlab.hpp/init_slab
    1405 
    1406 cysa = cy*sa 
    1407 cysb = cy*sb 
    1408 cysasb = cy*sa*sb 
    1409 
    1410 obj = cysasb
    1411 
    1412 CSG.Serialize([container, obj], args.csgpath )
    1413 
    1414 EOP
    1415 }





gdml.py
---------------


::

     393 class Tube(Primitive):
     394     """
     395     """
     396     @classmethod
     397     def make_cylinder(cls, radius, z1, z2, name):
     398         cn = CSG("cylinder", name=name)
     399         cn.param[0] = 0
     400         cn.param[1] = 0
     401         cn.param[2] = 0
     402         cn.param[3] = radius
     403         cn.param1[0] = z1
     404         cn.param1[1] = z2
     405         return cn
     406 
     407     @classmethod
     408     def make_disc(cls, x, y, inner, radius, z1, z2, name):
     409         cn = CSG("disc", name=name)
     410         cn.param[0] = x
     411         cn.param[1] = y
     412         cn.param[2] = inner
     413         cn.param[3] = radius
     414         cn.param1[0] = z1
     415         cn.param1[1] = z2
     416         return cn
     417 
     418     def as_cylinder(self, nudge_inner=0.01):
     419         hz = self.z/2.
     420         has_inner = self.rmin > 0.
     421 
     422         if has_inner:
     423             dz = hz*nudge_inner
     424             inner = self.make_cylinder(self.rmin, -(hz+dz), (hz+dz), self.name + "_inner")
     425         else:
     426             inner = None
     427         pass
     428         outer = self.make_cylinder(self.rmax, -hz, hz, self.name + "_outer" )
     429         return  CSG("difference", left=outer, right=inner, name=self.name + "_difference" ) if has_inner else outer
     430 
     431     def as_disc(self):
     432         hz = self.z/2.
     433         return self.make_disc( self.x, self.y, self.rmin, self.rmax, -hz, hz, self.name + "_disc" )
     434 
     435     def as_ncsg(self, hz_disc_cylinder_cut=1.):
     436         hz = self.z/2.
     437         rmin = self.rmin
     438         rmax = self.rmax
     439         pick_disc = hz < hz_disc_cylinder_cut
     440         if pick_disc:
     441             log.debug("Tube.as_ncsg.CSG_DISC %s as hz < cut,  hz:%s cut:%s rmin:%s rmax:%s " % (self.name, hz, hz_disc_cylinder_cut, rmin, rmax))
     442         pass
     443         return self.as_disc() if pick_disc else self.as_cylinder()
     444 


