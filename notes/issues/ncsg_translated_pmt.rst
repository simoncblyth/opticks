NCSG Translated PMT
======================

Missing physvol placement transforms
---------------------------------------

Doing all 5 solids of pmt together with tboolean-pmt
shows missing transforms dfor BOTTOM and presumably DYNODE too.





::

    081   <!-- The PMT vacuum -->
     82   <logvol name="lvPmtHemiVacuum" material="Vacuum">
     83     <union name="pmt-hemi-vac">
     84       <intersection name="pmt-hemi-bulb-vac">
     85     <sphere name="pmt-hemi-face-vac"
     86         outerRadius="PmtHemiFaceROCvac"/>
     87 
     88     <sphere name="pmt-hemi-top-vac"
     89         outerRadius="PmtHemiBellyROCvac"/>
     90     <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
     91 
     92     <sphere name="pmt-hemi-bot-vac"
     93         outerRadius="PmtHemiBellyROCvac"/>
     94     <posXYZ z="PmtHemiFaceOff+PmtHemiBellyOff"/>
     95 
     96       </intersection>
     97       <tubs name="pmt-hemi-base-vac"
     98         sizeZ="PmtHemiGlassBaseLength-PmtHemiGlassThickness"
     99         outerRadius="PmtHemiGlassBaseRadius-PmtHemiGlassThickness"/>
    100       <posXYZ z="-0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness"/>
    101     </union>
    102 
    103     <physvol name="pvPmtHemiCathode" 
    104          logvol="/dd/Geometry/PMT/lvPmtHemiCathode"/>
    105 
    106     <physvol name="pvPmtHemiBottom"
    107          logvol="/dd/Geometry/PMT/lvPmtHemiBottom">
    108       <posXYZ z="PmtHemiFaceOff+PmtHemiBellyOff"/>
    109     </physvol>
    110 
    111     <physvol name="pvPmtHemiDynode"
    112          logvol="/dd/Geometry/PMT/lvPmtHemiDynode">
    113       <posXYZ z="-0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness"/>
    114     </physvol>
    115   </logvol>



ddbase.py passes posXYZ placement from pv to referenced lv (within children, is it being used by translator?)::

    142     def children(self):
    143         """
    144         Defines the nature of the tree. 
    145 
    146         * for Physvol returns single item list containing the referenced Logvol
    147         * for Logvol returns list of all contained Physvol
    148         * otherwise returns empty list 
    149 
    150         NB bits of geometry of a Logvol are not regarded as children, 
    151         but rather are constitutent to it.
    152         """
    153         if type(self) is Physvol:
    154             posXYZ = self.find_("./posXYZ")
    155             lvn = self.logvolref.split("/")[-1]
    156             lv = self.g.logvol_(lvn)
    157             lv.posXYZ = posXYZ    # propagating the 
    158             if posXYZ is not None:
    159                 log.info("children... %s passing pv posXYZ to lv %s  " % (self.name, repr(lv)))
    160             return [lv]
    161         
    162         elif type(self) is Logvol:
    163             pvs = self.findall_("./physvol")
    164             return pvs
    165         else:
    166             return []
    167         pass


After follow that up with setting of node transforms on the lvnodes, 
the bottom appears to be in correct place, but dynode is poking thru the cathode...
this is probably the lack of transform offsets::

     15 class NCSGConverter(object):
     16     """
     17     Translate single volume detdesc primitives and CSG operations
     18     into an NCSG style node tree
     19     """
     20     @classmethod
     21     def ConvertLV(cls, lv ):
     22         """
     23         :param lv: Elem
     24         :return cn: CSG node instance 
     25         """
     26         lvgeom = lv.geometry()
     27         assert len(lvgeom) == 1, "expecting single CSG operator or primitive Elem within LV"
     28 
     29         cn = cls.convert(lvgeom[0])
     30 
     31         if lv.posXYZ is not None:
     32             assert cn.transform is None
     33             translate  = "%s,%s,%s" % (lv.xyz[0], lv.xyz[1], lv.xyz[2])
     34             cn.translate = translate
     35             log.info("TranslateLV posXYZ:%r -> translate %s  " % (lv.posXYZ, translate) )
     36         pass
     37         return cn
     38 




Cathode Inner or Outer
---------------------------

* can see from front but disappearing from back 
* observe wierdness in t_min clipping, 

* testing with tboolean-zsphere see the same wierdness, 
  its the missing cap handling 

* intersecting with a zslab works, but then you get a cap 

* used a flag to switch off the cap, but now getting sliver artifact and 
  spurious intersects

* actually switching off the caps prevents slab intersection from working, 
  get nothing with tboolean-sphere-slab ... cannot selectively have the intersect work for doing 
  the intersection chop and not work for giving an open cap...

  * cannot use infinite slab intersection without enabling the caps

  * so cannot use slab intersection and have open caps 
  * hmm, means must implement cap handling similar to cylinder in zsphere


Testing with tboolean-pmt with a kludge to just 
return the inner or outer in ncsgtranslator.py::


    182         cn.param[0] = en.xyz[0]
    183         cn.param[1] = en.xyz[1]
    184         cn.param[2] = en.xyz[2]
    185         cn.param[3] = radius
    186 
    187         if has_inner:
    188             #ret = CSG("difference", left=cn, right=inner )
    189             ret = inner
    190         else:
    191             ret = cn
    192         pass
    193         return ret
    194 






::

    2017-04-18 18:43:57.920 INFO  [962828] [GParts::dump@857] GParts::dump ni 4
         0.0000      0.0000      0.0000   1000.0000 
         0.0000      0.0000     123 <-bnd        0 <-INDEX    bn Rock//perfectAbsorbSurface/Vacuum 
         0.0000      0.0000      0.0000           6 (box) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000     124 <-bnd        1 <-INDEX    bn Vacuum///GlassSchottF2 
         0.0000      0.0000      0.0000           1 (union) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

         0.0000      0.0000      0.0000    127.9500 
        97.2867    127.9500     124 <-bnd        2 <-INDEX    bn Vacuum///GlassSchottF2 
         0.0000      0.0000      0.0000           7 (zsphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

         0.0000      0.0000     43.0000     98.9500 
        12.9934     55.7343     124 <-bnd        3 <-INDEX    bn Vacuum///GlassSchottF2 
         0.0000      0.0000      0.0000           7 (zsphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

