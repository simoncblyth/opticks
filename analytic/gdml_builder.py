#!/usr/bin/env python
"""

http://lxml.de/tutorial.html#the-e-factory

"""

import os, re, logging, math, collections

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main

import numpy as np
import lxml.etree as ET
import lxml.html as HT
from lxml.builder import E


tostring_ = lambda _:ET.tostring(_,pretty_print=True)
exists_ = lambda _:os.path.exists(os.path.expandvars(_))
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])


materials=ET.fromstring(r"""
  <materials>
    <material name="/dd/Materials/Pyrex0xc1005e0" state="solid">
    </material>
    <material name="/dd/Materials/MineralOil0xbf5c830" state="solid">
    </material>
    <material name="/dd/Materials/UnstStainlessSteel0xc5c11e8" state="solid">
    </material>
  </materials>
""")


solids=ET.fromstring(r"""
  <solids>

    <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-face-glass0xc0fde80" rmax="131" rmin="0" startphi="0" starttheta="0"/>
    <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-top-glass0xc0fdef0" rmax="102" rmin="0" startphi="0" starttheta="0"/>
    <intersection name="pmt-hemi-face-glass*ChildForpmt-hemi-glass-bulb0xbf1f8d0">
      <first ref="pmt-hemi-face-glass0xc0fde80"/>
      <second ref="pmt-hemi-top-glass0xc0fdef0"/>
      <position name="pmt-hemi-face-glass*ChildForpmt-hemi-glass-bulb0xbf1f8d0_pos" unit="mm" x="0" y="0" z="43"/>
    </intersection>

     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-bot-glass0xc0feac8" rmax="102" rmin="0" startphi="0" starttheta="0"/>

    <intersection name="pmt-hemi-glass-bulb0xc0feb98">
      <first ref="pmt-hemi-face-glass*ChildForpmt-hemi-glass-bulb0xbf1f8d0"/>
      <second ref="pmt-hemi-bot-glass0xc0feac8"/>
      <position name="pmt-hemi-glass-bulb0xc0feb98_pos" unit="mm" x="0" y="0" z="69"/>
    </intersection>

    <tube aunit="deg" deltaphi="360" lunit="mm" name="pmt-hemi-base0xc0fecb0" rmax="42.25" rmin="0" startphi="0" z="169"/>

    <union name="pmt-hemi0xc0fed90">
      <first ref="pmt-hemi-glass-bulb0xc0feb98"/>
      <second ref="pmt-hemi-base0xc0fecb0"/>
      <position name="pmt-hemi0xc0fed90_pos" unit="mm" x="0" y="0" z="-84.5"/>
    </union>

    <tube aunit="deg" deltaphi="360" lunit="mm" name="AdPmtCollar0xc2c5260" rmax="106" rmin="105" startphi="0" z="12.7"/>

    <tube aunit="deg" deltaphi="360" lunit="mm" name="oil0xbf5ed48" rmax="2488" rmin="0" startphi="0" z="4955"/>


   </solids>
""")


"""
      <physvol name="/dd/Geometry/AD/lvOIL#pvOAV0xbf8f638">
        <volumeref ref="/dd/Geometry/AD/lvOAV0xbf1c760"/>
        <position name="/dd/Geometry/AD/lvOIL#pvOAV0xbf8f638_pos" unit="mm" x="0" y="0" z="-49"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvOAV0xbf8f638_rot" unit="deg" x="0" y="0" z="-180"/>
      </physvol>


"""

structure={}

structure["collar"]=ET.fromstring(r"""

   <structure>

    <volume name="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0">
      <materialref ref="/dd/Materials/UnstStainlessSteel0xc5c11e8"/>
      <solidref ref="AdPmtCollar0xc2c5260"/>
    </volume>
 
    <volume name="/dd/Geometry/AD/lvOIL0xbf5e0b8">
      <materialref ref="/dd/Materials/MineralOil0xbf5c830"/>
      <solidref ref="oil0xbf5ed48"/>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920_pos" unit="mm" x="-2249.09266802649" y="-296.098667051187" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920_rot" unit="deg" x="90" y="-82.5" z="90"/>
      </physvol>

 
    </volume>
  </structure>
""")


structure["collar2"]=ET.fromstring(r"""

   <structure>

    <volume name="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0">
      <materialref ref="/dd/Materials/UnstStainlessSteel0xc5c11e8"/>
      <solidref ref="AdPmtCollar0xc2c5260"/>
    </volume>
 
    <volume name="/dd/Geometry/AD/lvOIL0xbf5e0b8">
      <materialref ref="/dd/Materials/MineralOil0xbf5c830"/>
      <solidref ref="oil0xbf5ed48"/>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920_pos" unit="mm" x="-2249.09266802649" y="-296.098667051187" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920_rot" unit="deg" x="90" y="-82.5" z="90"/>
      </physvol>


      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmtCollar0xc25dce8">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmtCollar0xc25dce8_pos" unit="mm" x="-2095.82071950185" y="-868.117366320206" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmtCollar0xc25dce8_rot" unit="deg" x="90" y="-67.5" z="90"/>
      </physvol>



 
    </volume>
  </structure>
""")







structure["pmt1"]=ET.fromstring(r"""
   <structure>
    <volume name="/dd/Geometry/PMT/lvPmtHemi0xc133740">
      <materialref ref="/dd/Materials/Pyrex0xc1005e0"/>
      <solidref ref="pmt-hemi0xc0fed90"/>
    </volume>

    <volume name="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0">
      <materialref ref="/dd/Materials/UnstStainlessSteel0xc5c11e8"/>
      <solidref ref="AdPmtCollar0xc2c5260"/>
    </volume>
 
    <volume name="/dd/Geometry/AD/lvOIL0xbf5e0b8">
      <materialref ref="/dd/Materials/MineralOil0xbf5c830"/>
      <solidref ref="oil0xbf5ed48"/>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc133740"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40_pos" unit="mm" x="-2304.61358026342" y="-303.40813381551" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40_rot" unit="deg" x="90" y="-82.5" z="90"/>
      </physvol>

    </volume>
  </structure>
""")

r"""
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40_rot" unit="deg" x="90" y="0" z="90"/>

"""





structure["pmt2"]=ET.fromstring(r"""
   <structure>
    <volume name="/dd/Geometry/PMT/lvPmtHemi0xc133740">
      <materialref ref="/dd/Materials/Pyrex0xc1005e0"/>
      <solidref ref="pmt-hemi0xc0fed90"/>
    </volume>

    <volume name="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0">
      <materialref ref="/dd/Materials/UnstStainlessSteel0xc5c11e8"/>
      <solidref ref="AdPmtCollar0xc2c5260"/>
    </volume>
 
    <volume name="/dd/Geometry/AD/lvOIL0xbf5e0b8">
      <materialref ref="/dd/Materials/MineralOil0xbf5c830"/>
      <solidref ref="oil0xbf5ed48"/>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc133740"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40_pos" unit="mm" x="-2304.61358026342" y="-303.40813381551" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40_rot" unit="deg" x="90" y="-82.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920_pos" unit="mm" x="-2249.09266802649" y="-296.098667051187" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920_rot" unit="deg" x="90" y="-82.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt0xc25dc28">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc133740"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt0xc25dc28_pos" unit="mm" x="-2147.55797332249" y="-889.547638532651" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt0xc25dc28_rot" unit="deg" x="90" y="-67.5" z="90"/>
      </physvol>
 
    </volume>
  </structure>
""")



structure["pmt5"]=ET.fromstring(r"""
   <structure>
    <volume name="/dd/Geometry/PMT/lvPmtHemi0xc133740">
      <materialref ref="/dd/Materials/Pyrex0xc1005e0"/>
      <solidref ref="pmt-hemi0xc0fed90"/>
    </volume>

    <volume name="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0">
      <materialref ref="/dd/Materials/UnstStainlessSteel0xc5c11e8"/>
      <solidref ref="AdPmtCollar0xc2c5260"/>
    </volume>
 

    <volume name="/dd/Geometry/AD/lvOIL0xbf5e0b8">
      <materialref ref="/dd/Materials/MineralOil0xbf5c830"/>
      <solidref ref="oil0xbf5ed48"/>


      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc133740"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40_pos" unit="mm" x="-2304.61358026342" y="-303.40813381551" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt0xc2a6b40_rot" unit="deg" x="90" y="-82.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920_pos" unit="mm" x="-2249.09266802649" y="-296.098667051187" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar0xc569920_rot" unit="deg" x="90" y="-82.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt0xc25dc28">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc133740"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt0xc25dc28_pos" unit="mm" x="-2147.55797332249" y="-889.547638532651" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt0xc25dc28_rot" unit="deg" x="90" y="-67.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmtCollar0xc25dce8">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmtCollar0xc25dce8_pos" unit="mm" x="-2095.82071950185" y="-868.117366320206" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmtCollar0xc25dce8_rot" unit="deg" x="90" y="-67.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmt0xc25dda0">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc133740"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmt0xc25dda0_pos" unit="mm" x="-1844.14983950698" y="-1415.06594173077" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmt0xc25dda0_rot" unit="deg" x="90" y="-52.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmtCollar0xc25de60">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmtCollar0xc25de60_pos" unit="mm" x="-1799.72205245067" y="-1380.97530170628" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmtCollar0xc25de60_rot" unit="deg" x="90" y="-52.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:4#pvAdPmtUnit#pvAdPmt0xc25df68">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc133740"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:4#pvAdPmtUnit#pvAdPmt0xc25df68_pos" unit="mm" x="-1415.06594173077" y="-1844.14983950698" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:4#pvAdPmtUnit#pvAdPmt0xc25df68_rot" unit="deg" x="90" y="-37.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:4#pvAdPmtUnit#pvAdPmtCollar0xc25e078">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:4#pvAdPmtUnit#pvAdPmtCollar0xc25e078_pos" unit="mm" x="-1380.97530170628" y="-1799.72205245067" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:4#pvAdPmtUnit#pvAdPmtCollar0xc25e078_rot" unit="deg" x="90" y="-37.5" z="90"/>
      </physvol>

      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:5#pvAdPmtUnit#pvAdPmt0xc25e180">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc133740"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:5#pvAdPmtUnit#pvAdPmt0xc25e180_pos" unit="mm" x="-889.547638532652" y="-2147.55797332249" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:5#pvAdPmtUnit#pvAdPmt0xc25e180_rot" unit="deg" x="90" y="-22.5" z="90"/>
      </physvol>


      <physvol name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:5#pvAdPmtUnit#pvAdPmtCollar0xc25e290">
        <volumeref ref="/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0"/>
        <position name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:5#pvAdPmtUnit#pvAdPmtCollar0xc25e290_pos" unit="mm" x="-868.117366320207" y="-2095.82071950185" z="-1750"/>
        <rotation name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:5#pvAdPmtUnit#pvAdPmtCollar0xc25e290_rot" unit="deg" x="90" y="-22.5" z="90"/>
      </physvol>
 
    </volume>
  </structure>
""")


def make_gdml(worldref="/dd/Geometry/PMT/lvPmtHemi0xc133740", structure_key="pmt5"):
   doc = E.gdml(
             materials,
             solids,
             structure.get(structure_key),
             E.setup(E.world(ref=worldref), name="Default", version="1.0")
             )
   return doc





if __name__ == '__main__':
   gg = make_gdml()
   print tostring_(gg)


