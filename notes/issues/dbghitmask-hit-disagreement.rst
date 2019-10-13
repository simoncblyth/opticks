dbghitmask-hit-disagreement
=============================


::

    OpticksIdx=error OKTest --target 62590 --xanalytic --eye -0.9,0,0 --generateoverride -1 --save --dbghitmask TO,BT,SC,RE,SA


Issue : looks like contradictory logging messages re hit downloading
---------------------------------------------------------------------------


::

    2019-10-13 17:38:58.915 INFO  [322418] [OpEngine::downloadEvent@186] .
    2019-10-13 17:38:58.915 INFO  [322418] [OContext::download@922] PROCEED for sequence as OPTIX_NON_INTEROP
    2019-10-13 17:38:58.939 INFO  [322418] [OEvent::downloadHits@382]  nhit 21988 --dbghit N hitmask 0x18b0 RE|SC|SA|BT|TO BULK_REEMIT|BULK_SCATTER|SURFACE_ABSORB|BOUNDARY_TRANSMIT|TORCH
    2019-10-13 17:38:58.940 INFO  [322418] [OpticksEvent::saveHitData@1683] saveHitData zero hits 
    2019-10-13 17:38:58.942 WARN  [322418] [OpticksEvent::saveIndex@2366] SKIP as not indexed 
    2019-10-13 17:38:58.944 INFO  [322418] [OpticksEvent::makeReport@1760] tagdir /home/blyth/local/opticks/evtbase/OKTest/evt/g4live/torch/-1
    2019-10-13 17:38:59.838 INFO  [322418] [OpticksEvent::makeReport@1760] tagdir /home/blyth/local/opticks/evtbase/OKTest/evt/g4live/torch/1
    2019-10-13 17:38:59.844 WARN  [322418] [GGeo::anaEvent@1969] GGeo::anaEvent evt 0x28715af0
    2019-10-13 17:38:59.844 INFO  [322418] [OpticksAna::run@92]  anakey (null) enabled N



Just Misleading logging from empty G4 evt (tag -1) next to OK event (tag 1) 
-------------------------------------------------------------------------------

::

    2019-10-13 18:45:34.089 INFO  [428852] [OpEngine::downloadEvent@186] .
    2019-10-13 18:45:34.089 NONE  [428852] [OEvent::download@418] [ id 0
    2019-10-13 18:45:34.089 INFO  [428852] [OContext::download@922] PROCEED for sequence as OPTIX_NON_INTEROP
    2019-10-13 18:45:34.106 NONE  [428852] [OEvent::download@453] ]
    2019-10-13 18:45:34.110 NONE  [428852] [OEvent::downloadHitsInterop@525]  nhit 21988 hit 21988,4,4
    2019-10-13 18:45:34.110 INFO  [428852] [OEvent::downloadHits@382]  nhit 21988 --dbghit N hitmask 0x18b0 RE|SC|SA|BT|TO BULK_REEMIT|BULK_SCATTER|SURFACE_ABSORB|BOUNDARY_TRANSMIT|TORCH
    2019-10-13 18:45:34.110 NONE  [428852] [OEvent::download@407]  nhit 21988
    2019-10-13 18:45:34.111 INFO  [428852] [OpticksEvent::saveHitData@1683]  num_hit 0 ht 0,4,4 tag -1
    2019-10-13 18:45:34.111 WARN  [428852] [OpticksEvent::saveIndex@2373] SKIP as not indexed 
    2019-10-13 18:45:34.112 INFO  [428852] [OpticksEvent::makeReport@1767] tagdir /home/blyth/local/opticks/evtbase/OKTest/evt/g4live/torch/-1
    2019-10-13 18:45:34.120 INFO  [428852] [OpticksEvent::saveHitData@1683]  num_hit 21988 ht 21988,4,4 tag 1
    2019-10-13 18:45:34.985 INFO  [428852] [OpticksEvent::makeReport@1767] tagdir /home/blyth/local/opticks/evtbase/OKTest/evt/g4live/torch/1
    2019-10-13 18:45:34.991 WARN  [428852] [GGeo::anaEvent@1969] GGeo::anaEvent evt 0x27554ae0
    2019-10-13 18:45:34.991 INFO  [428852] [OpticksAna::run@92]  anakey (null) enabled N




See a zero hit logging message, but it aint true : EXPLAINED by the empty G4 evt 
------------------------------------------------------------------------------------------

::

    [blyth@localhost 1]$ np.py /home/blyth/local/opticks/evtbase/OKTest/evt/g4live/torch/1/ht.npy
    a : /home/blyth/local/opticks/evtbase/OKTest/evt/g4live/torch/1/ht.npy :        (21988, 4, 4) : 697edbbe58ee293028f03cbecc3de23d : 20191013-1738 

    [blyth@localhost 1]$ np.py ht.npy
    a :                                                       ht.npy :        (21988, 4, 4) : 697edbbe58ee293028f03cbecc3de23d : 20191013-1826


Huh::

    1677 void OpticksEvent::saveHitData(NPY<float>* ht) const
    1678 {
    1679     if(ht)
    1680     {
    1681         unsigned num_hit = ht->getNumItems();
    1682         ht->save(m_pfx, "ht", m_typ,  m_tag, m_udet);  // even when zero hits
    1683         if(num_hit == 0) LOG(info) << "saveHitData zero hits " ;
    1684     }
    1685 }
    1686 



