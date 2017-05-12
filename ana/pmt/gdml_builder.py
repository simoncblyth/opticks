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

   </solids>
""")


structure=ET.fromstring(r"""
   <structure>
    <volume name="/dd/Geometry/PMT/lvPmtHemi0xc133740">
      <materialref ref="/dd/Materials/Pyrex0xc1005e0"/>
      <solidref ref="pmt-hemi0xc0fed90"/>
    </volume>
  </structure>
""")


def make_gdml(worldref="/dd/Geometry/PMT/lvPmtHemi0xc133740"):
   doc = E.gdml(
             materials,
             solids,
             structure,
             E.setup(E.world(ref=worldref), name="Default", version="1.0")
             )
   return doc





if __name__ == '__main__':
   gg = make_gdml()
   print tostring_(gg)


