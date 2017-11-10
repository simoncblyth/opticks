FIXED : material names wrong for test geometry python side
==============================================================


* note the related issue with full geometries, for material indices exceeding 15 
  as seqmat only uses a nibble for each step, is not fixed


New Way of handling test geometry 
---------------------------------------


::

    simon:cfg4 blyth$ tboolean-;tboolean-media --okg4

    simon:ana blyth$ tboolean-;tboolean-media-ip
    args: /opt/local/bin/ipython --profile=g4opticks -i -- tboolean.py --det tboolean-media --tag 1
    [2017-11-10 13:23:40,383] p12700 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-media c2max 2.0 ipython True 
    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171110-1235 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171110-1235 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Pyrex
    /tmp/blyth/opticks/tboolean-media--
    ...

    .                seqmat_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  LS Gd
    0001     289569    290680             2.13  LS LS
    0002        290       292             0.01  LS LS Gd
    0003         82        98             1.42  LS LS LS
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
                   /tmp/blyth/opticks/evt/tboolean-media/torch/1 2e356218300096067b2b665b354d6469 414c405507776baab2b9b0f30ae1e582  600000    -1.0000 INTEROP_MODE 
    {u'nx': u'20', u'emitconfig': u'photons=600000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'emit': -1, u'poly': u'MC'}

    In [1]: ab
    Out[1]: 
    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171110-1235 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171110-1235 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Pyrex
    /tmp/blyth/opticks/tboolean-media--

    In [2]: ab.mat
    Out[2]: 
    .                seqmat_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  LS Gd
    0001     289569    290680             2.13  LS LS
    0002        290       292             0.01  LS LS Gd
    0003         82        98             1.42  LS LS LS
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  

    In [3]: ab.mat.__class__
    Out[3]: opticks.ana.seq.SeqTable



    In [5]: ab.a.mat
    Out[5]: 
    .                            1:tboolean-media 
    .                             600000         1.00 
    0000         0.517      310059      LS Gd
    0001         0.483      289569      LS LS
    0002         0.000         290      LS LS Gd
    0003         0.000          82      LS LS LS
    .                             600000         1.00 

    In [6]: ab.a.his
    Out[6]: 
    .                            1:tboolean-media 
    .                             600000         1.00 
    0000         0.517      310059      TO SA
    0001         0.483      289569      TO AB
    0002         0.000         290      TO SC SA
    0003         0.000          82      TO SC AB
    .                             600000         1.00 

    In [7]: ab.a.mattype
    Out[7]: <opticks.ana.mattype.MatType at 0x10637ea90>



    073 class MatType(SeqType):
     74     """
     75     MatType specializes SeqType by providing it with 
     76     material codes and abbreviations.
     77 
     78     ::
     79 
     80         In [17]: flags.code2name
     81         Out[17]: 
     82         {1: 'GdDopedLS',
     83          2: 'LiquidScintillator',
     84          3: 'Acrylic',
     85          4: 'MineralOil',
     86        
     87         In [18]: abbrev.abbr2name
     88         Out[18]: 
     89         {'AS': 'ADTableStainlessSteel',
     90          'Ac': 'Acrylic',
     91          'Ai': 'Air',
     92          'Al': 'Aluminium',
     93          'Bk': 'Bialkali',
     94          'Dw': 'DeadWater',
     95 
     96     """
     97     def __init__(self):
     98         material_names = ItemList("GMaterialLib")
     99         material_abbrev = Abbrev("$OPTICKS_DETECTOR_DIR/GMaterialLib/abbrev.json")
    100         SeqType.__init__(self, material_names, material_abbrev)
    101 

    140 class SeqType(BaseType):
    141     def __init__(self, flags, abbrev):
    142          BaseType.__init__(self, flags, abbrev, delim=" ")
    143 

    014 class BaseType(object):
     15     hexstr = re.compile("^[0-9a-f]+$")
     16     def __init__(self, flags, abbrev, delim=" "):
     17         """
     18         When no abbreviation available, use first and last letter of name eg::
     19 
     20            MACHINERY -> MY
     21            FABRICATED -> FD
     22            G4GUN -> GN
     23 
     24         """
     25         abbrs = map(lambda name:abbrev.name2abbr.get(name,firstlast_(name)), flags.names )
     26         self.abbr2code = dict(zip(abbrs, flags.codes))
     27         self.code2abbr = dict(zip(flags.codes, abbrs))
     28         self.flags = flags
     29         self.abbrev = abbrev
     30         self.delim = delim


    In [7]: ab.a.mattype
    Out[7]: <opticks.ana.mattype.MatType at 0x10637ea90>

    In [8]: ab.a.mattype.flags
    Out[8]: ItemLists names     38 name2code     38 code2name     38 offset     1 npath $IDPATH/GItemList/GMaterialLib.txt 

    /// ... python grabbing matnames from geocache ...
    /// but that no longer correct for new way of handling test geometry 
    /// with test geometry the CSGList directory becomes a kinda proxy for the geocache  

    In [9]: ab.a.mattype.abbrev
    Out[9]: <opticks.ana.base.Abbrev at 0x10637eb50>

    In [10]: ab.a.mattype.abbrev.__class__
    Out[10]: opticks.ana.base.Abbrev

    In [11]: ab.a.histype.abbrev
    Out[11]: <opticks.ana.base.Abbrev at 0x10637ead0>

    In [12]: ab.a.histype.flags
    Out[12]: <opticks.ana.base.IniFlags at 0x10637ea50>



::

    423 void GPropertyLib::saveToCache()
    424 {
    425 
    426     LOG(trace) << "GPropertyLib::saveToCache" ;
    427 
    428 
    429     if(!isClosed()) close();
    430 
    431     if(m_buffer)
    432     {
    433         std::string dir = getCacheDir();
    434         std::string name = getBufferName();
    435         m_buffer->save(dir.c_str(), name.c_str());
    436 
    437         if(m_meta)
    438         {
    439             m_meta->save(dir.c_str(),  METANAME );
    440         }
    441     }
    442 
    443     if(m_names)
    444     {
    445         m_names->save(m_resource->getIdPath());
    446     }
    447 
    448 
    449     LOG(trace) << "GPropertyLib::saveToCache DONE" ;
    450 
    451 }




Fix in GGeoTest::init::


    simon:ana blyth$ ll /tmp/blyth/opticks/tboolean-media--/
    total 16
    drwxr-xr-x    8 blyth  wheel   272 Nov  4 17:56 0
    drwxr-xr-x  179 blyth  wheel  6086 Nov 10 12:27 ..
    -rw-r--r--    1 blyth  wheel   132 Nov 10 14:19 csgmeta.json
    -rw-r--r--    1 blyth  wheel    32 Nov 10 14:19 csg.txt
    drwxr-xr-x    4 blyth  wheel   136 Nov 10 14:19 GItemList
    drwxr-xr-x    6 blyth  wheel   204 Nov 10 14:19 .
    simon:ana blyth$ ll /tmp/blyth/opticks/tboolean-media--/GItemList/
    total 16
    -rw-r--r--  1 blyth  wheel   21 Nov 10 14:19 GSurfaceLib.txt
    -rw-r--r--  1 blyth  wheel   11 Nov 10 14:19 GMaterialLib.txt
    drwxr-xr-x  6 blyth  wheel  204 Nov 10 14:19 ..
    drwxr-xr-x  4 blyth  wheel  136 Nov 10 14:19 .
    simon:ana blyth$ cat /tmp/blyth/opticks/tboolean-media--/GItemList/GMaterialLib.txt 
    Rock
    Pyrex
    simon:ana blyth$ 
    simon:ana blyth$ 
    simon:ana blyth$ cat /tmp/blyth/opticks/tboolean-media--/GItemList/GSurfaceLib.txt
    perfectAbsorbSurface
    simon:ana blyth$ 



Fixed::

    [2017-11-10 14:52:01,755] p23894 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171110-1419 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171110-1419 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Pyrex
    /tmp/blyth/opticks/tboolean-media--
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  TO SA
    0001     289569    290680             2.13  TO AB
    0002        290       292             0.01  TO SC SA
    0003         82        98             1.42  TO SC AB
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    .                pflags_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  TO|SA
    0001     289569    290680             2.13  TO|AB
    0002        290       292             0.01  TO|SA|SC
    0003         82        98             1.42  TO|SC|AB
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    .                seqmat_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  Py Rk
    0001     289569    290680             2.13  Py Py
    0002        290       292             0.01  Py Py Rk
    0003         82        98             1.42  Py Py Py
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
                   /tmp/blyth/opticks/evt/tboolean-media/torch/1 2e356218300096067b2b665b354d6469 414c405507776baab2b9b0f30ae1e582  600000    -1.0000 INTEROP_MODE 
    {u'nx': u'20', u'emitconfig': u'photons=600000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'emit': -1, u'poly': u'MC'}

    In [1]: 

