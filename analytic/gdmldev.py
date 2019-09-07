#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""

::

    In [1]: run gdml.py
    ...

    In [13]: len(g.volumes)  # lv
    Out[13]: 249

    In [14]: len(g.solids)   # top level solids, ie root nodes of all the gdml solids
    Out[14]: 707             # comparing lv:249 so:707 -> avg  ~3 solids per lv 

    In [15]: len(g.materials)
    Out[15]: 36


Hmm in scene serialization ? Preserve all the solids, or flatten into lv ?


Or flatten yet more... eg for PMT there are 5 solids that natuarally go together
to make an instance...

* need to analyse the solid tree to look for such clusters, or defer that til later ?

::

    In [76]: g.volumes(47)  # eg lv:47 comprises so:131,129,127,125,126,128,130 
    Out[76]: 
    [47] Volume /dd/Geometry/PMT/lvPmtHemi0xc133740 /dd/Materials/Pyrex0xc1005e0 pmt-hemi0xc0fed90
       [131] Union pmt-hemi0xc0fed90  
         l:[129] Intersection pmt-hemi-glass-bulb0xc0feb98  
         l:[127] Intersection pmt-hemi-face-glass*ChildForpmt-hemi-glass-bulb0xbf1f8d0  
         l:[125] Sphere pmt-hemi-face-glass0xc0fde80 mm rmin 0.0 rmax 131.0  x 0.0 y 0.0 z 0.0  
         r:[126] Sphere pmt-hemi-top-glass0xc0fdef0 mm rmin 0.0 rmax 102.0  x 0.0 y 0.0 z 0.0  
         r:[128] Sphere pmt-hemi-bot-glass0xc0feac8 mm rmin 0.0 rmax 102.0  x 0.0 y 0.0 z 0.0  
         r:[130] Tube pmt-hemi-base0xc0fecb0 mm rmin 0.0 rmax 42.25  x 0.0 y 0.0 z 169.0  
       [14] Material /dd/Materials/Pyrex0xc1005e0 solid
       PhysVol /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e8
     None None 


::

    In [83]: g.volumes(2).physvol   ## pv are placements: so there are loads of em, thus no need for abaolute indexing 
    Out[83]: 
    [PhysVol /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xc311da0
     Position mm -910.0 0.0 0.0  Rotation deg 0.0 0.0 -90.0  ,
     PhysVol /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit0xc125cf8
     Position mm -650.0 0.0 0.0  Rotation deg 0.0 0.0 -90.0  ,
     PhysVol /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:3#pvStrip14Unit0xc125df0
     Position mm -390.0 0.0 0.0  Rotation deg 0.0 0.0 -90.0  ,
     PhysVol /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:4#pvStrip14Unit0xc125ee8
     Position mm -130.0 0.0 0.0  Rotation deg 0.0 0.0 -90.0  ,
     PhysVol /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:5#pvStrip14Unit0xc125fe0
     Position mm 130.0 0.0 0.0  Rotation deg 0.0 0.0 -90.0  ,
     PhysVol /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:6#pvStrip14Unit0xc1260d8
     Position mm 390.0 0.0 0.0  Rotation deg 0.0 0.0 -90.0  ,
     PhysVol /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:7#pvStrip14Unit0xc1261d0
     Position mm 650.0 0.0 0.0  Rotation deg 0.0 0.0 -90.0  ,
     PhysVol /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:8#pvStrip14Unit0xc1262c8
     Position mm 910.0 0.0 0.0  Rotation deg 0.0 0.0 -90.0  ]



Absolute gidx volume(lv), solid and material indexing done via odict:: 

    In [75]: g.volumes.keys()
    Out[75]: 
    ['/dd/Geometry/PoolDetails/lvNearTopCover0xc137060',
     '/dd/Geometry/RPC/lvRPCStrip0xc2213c0',
     '/dd/Geometry/RPC/lvRPCGasgap140xbf98ae0',
     ...



    In [59]: g.volumes(100)
    Out[59]: 
    [100] Volume /dd/Geometry/CalibrationSources/lvLedSourceShell0xc3066b0 /dd/Materials/Acrylic0xc02ab98 led-source-shell0xc3068f0
       [320] Union led-source-shell0xc3068f0  
         l:[318] Union led-acryliccylinder+ChildForled-source-shell0xc306188  
         l:[316] Tube led-acryliccylinder0xc306f40 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 29.73  
         r:[317] Sphere led-acrylicendtop0xc306fe8 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  
         r:[319] Sphere led-acrylicendbot0xc307120 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  
       [8] Material /dd/Materials/Acrylic0xc02ab98 solid
       PhysVol /dd/Geometry/CalibrationSources/lvLedSourceShell#pvDiffuserBall0xc0d3488
     None None 

    In [60]: g.solids(320)
    Out[60]: 
    [320] Union led-source-shell0xc3068f0  
         l:[318] Union led-acryliccylinder+ChildForled-source-shell0xc306188  
         l:[316] Tube led-acryliccylinder0xc306f40 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 29.73  
         r:[317] Sphere led-acrylicendtop0xc306fe8 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  
         r:[319] Sphere led-acrylicendbot0xc307120 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  

    In [61]: g.solids(319)
    Out[61]: [319] Sphere led-acrylicendbot0xc307120 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  

    In [62]: g.materials(8)
    Out[62]: [8] Material /dd/Materials/Acrylic0xc02ab98 solid



    In [72]: print g.solids(2).xml
    <subtraction xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="near_top_cover-ChildFornear_top_cover_box0xc241498">
          <first ref="near_top_cover0xc5843d8"/>
          <second ref="near_top_cover_sub00xc584418"/>
          <position name="near_top_cover-ChildFornear_top_cover_box0xc241498_pos" unit="mm" x="8000" y="5000" z="0"/>
          <rotation name="near_top_cover-ChildFornear_top_cover_box0xc241498_rot" unit="deg" x="0" y="0" z="45"/>
        </subtraction>
        



    In [55]: g.volumes(100).solid
    Out[55]: 
    Union led-source-shell0xc3068f0  
         l:Union led-acryliccylinder+ChildForled-source-shell0xc306188  
         l:Tube led-acryliccylinder0xc306f40 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 29.73  
         r:Sphere led-acrylicendtop0xc306fe8 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  
         r:Sphere led-acrylicendbot0xc307120 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  

    In [56]: g.volumes(100).solid.idx
    Out[56]: 320

    In [57]: g.solids(320)
    Out[57]: 
    Union led-source-shell0xc3068f0  
         l:Union led-acryliccylinder+ChildForled-source-shell0xc306188  
         l:Tube led-acryliccylinder0xc306f40 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 29.73  
         r:Sphere led-acrylicendtop0xc306fe8 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  
         r:Sphere led-acrylicendbot0xc307120 mm rmin 0.0 rmax 10.035  x 0.0 y 0.0 z 0.0  





    In [19]: len(gdml.elem.findall("solids/*"))
    Out[19]: 707

    In [20]: len(gdml.elem.findall("solids//*"))
    Out[20]: 1526

    In [21]: set([e.tag for e in gdml.elem.findall("solids//*")])
    Out[21]: 
    {
     'position',
     'rotation',

     'first',
     'second',
     'intersection',
     'subtraction',
     'union',

     'box',     
     'sphere',
     'tube',
     'cone',
     'polycone', 'zplane'
     'trd',                 # trapezoid

   }


    In [26]: for e in gdml.elem.findall("solids//tube"):print tostring_(e)

    In [26]: for e in gdml.elem.findall("solids//trd"):print tostring_(e)
    <trd xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" lunit="mm" name="SstTopRadiusRibBase0xc271078" x1="160" x2="691.02" y1="20" y2="20" z="2228.5"/>
    <trd xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" lunit="mm" name="SstInnVerRibCut0xbf31118" x1="100" x2="237.2" y1="27" y2="27" z="50.02"/>


"""


if __name__ == '__main__':
    pass


 
