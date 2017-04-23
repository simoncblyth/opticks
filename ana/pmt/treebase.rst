Treebase
===========

::

    In [1]: run treebase.py
    ...
    In [2]: tr.root
    Out[2]: Node  0 : dig 8c5f pig d41d : LV lvPmtHemi                           Pyrex None : None  : None 

    In [5]: tr.byindex
    Out[5]: 
    {0: Node  0 : dig f34b pig d41d : LV lvPmtHemi                           Pyrex None : None  : None ,
     1: Node  1 : dig fafa pig f34b : LV lvPmtHemiVacuum                    Vacuum None : None  : None ,
     2: Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None ,
     3: Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0    ,
     4: Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5    }

    In [6]: tr.registry
    Out[6]: 
    {'324d9022d803eae989b540bbb2375f38': Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None ,
     '5e291f5b9bbedf27f720e5dceb65ad56': Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5    ,
     '9e612c43301f13a23a5fd41e7ea59404': Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0    ,
     'f34ba27750136ebdc5bd9f3119f2c559': Node  0 : dig f34b pig d41d : LV lvPmtHemi                           Pyrex None : None  : None ,
     'fafaa4fcd3682ac3f89da9afb9680e9a': Node  1 : dig fafa pig f34b : LV lvPmtHemiVacuum                    Vacuum None : None  : None }

    In [7]: 


Recursive dumper::

    In [20]: tr.get(0).traverse()
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  0 : dig f34b pig d41d : LV lvPmtHemi                           Pyrex None : None  : None  
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 1 Node  1 : dig fafa pig f34b : LV lvPmtHemiVacuum                    Vacuum None : None  : None  
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 2 Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None  
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 2 Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 2 Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     

    In [21]: tr.get(1).traverse()
    [2017-04-14 19:52:01,124] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  1 : dig fafa pig f34b : LV lvPmtHemiVacuum                    Vacuum None : None  : None  
    [2017-04-14 19:52:01,124] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 1 Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None  
    [2017-04-14 19:52:01,125] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 1 Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     
    [2017-04-14 19:52:01,125] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 1 Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     

    In [22]: tr.get(2).traverse()
    [2017-04-14 19:52:17,660] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None  

    In [23]: tr.get(3).traverse()
    [2017-04-14 19:52:29,365] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     

    In [24]: tr.get(4).traverse()
    [2017-04-14 19:52:42,476] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     


::

     37   <!-- The PMT glass -->
     38   <logvol name="lvPmtHemi" material="Pyrex">
     39     <union name="pmt-hemi">
     40       <intersection name="pmt-hemi-glass-bulb">
     41     <sphere name="pmt-hemi-face-glass"
     42         outerRadius="PmtHemiFaceROC"/>
     43 
     44     <sphere name="pmt-hemi-top-glass"
     45         outerRadius="PmtHemiBellyROC"/>
     46     <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
     47 
     48     <sphere name="pmt-hemi-bot-glass"
     49         outerRadius="PmtHemiBellyROC"/>
     50     <posXYZ z="PmtHemiFaceOff+PmtHemiBellyOff"/>
     51 
     52       </intersection>
     53       <tubs name="pmt-hemi-base"
     54         sizeZ="PmtHemiGlassBaseLength"
     55         outerRadius="PmtHemiGlassBaseRadius"/>
     56       <posXYZ z="-0.5*PmtHemiGlassBaseLength"/>
     57     </union>
     58 
     59     <physvol name="pvPmtHemiVacuum"
     60          logvol="/dd/Geometry/PMT/lvPmtHemiVacuum"/>
     61 
     62   </logvol>

::

    In [48]: py = tr.get(0)

    In [54]: py.lv.name
    Out[54]: 'lvPmtHemi'

    In [55]: py.lv.material
    Out[55]: 'Pyrex'

    In [57]: py.lv.findall_("./*")
    Out[57]: 
    [Union             pmt-hemi  ,
     PV pvPmtHemiVacuum      /dd/Geometry/PMT/lvPmtHemiVacuum ]

    In [58]: un = py.lv.findall_("./*")[0]

    In [59]: un
    Out[59]: Union             pmt-hemi  

    In [60]: un.findall_("./*")
    Out[60]: 
    [Intersection  pmt-hemi-glass-bulb  ,
     Tubs        pmt-hemi-base : outerRadius PmtHemiGlassBaseRadius : 42.25   sizeZ PmtHemiGlassBaseLength : 169.0   :  None ,
     PosXYZ  -0.5*PmtHemiGlassBaseLength : -84.5   ]


Need to findall_ recurse on the lv, constructing NCSG node tree.
Unclear what level to do this at, probably simpler to operate at dd level


