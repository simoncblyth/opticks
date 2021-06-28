tds3ip_OK_pflags_mismatch_warning
====================================

* this is because pflags2 lacks "EC|EX"


* avoided warning with 

::

     848         # see notes/issues/tds3ip_OK_pflags_mismatch_warning.rst
     849         exclude_ecex = True
     850         if exclude_ecex:
     851             ecex = self.hismask.code("EC|EX")
     852             pflags_without_ecex = self.pflags & ~ecex
     853             pflags = pflags_without_ecex
     854         else:
     855             pflags = self.pflags
     856         pass
     857 
     858         self.msk_mismatch = pflags != self.pflags2
     859         self.num_msk_mismatch = np.count_nonzero(self.msk_mismatch)
     860 
     861         if self.num_msk_mismatch == 0:
     862             log.debug("pflags2(=seq2msk(seqhis)) and pflags  match")
     863         else:
     864             log.info("pflags2(=seq2msk(seqhis)) and pflags  MISMATCH    num_msk_mismatch: %d " % self.num_msk_mismatch )
     865         pass
     866 




::

    [{__init__            :evt.py    :296} INFO     - ] A 
    [{init_sequence       :evt.py    :853} INFO     - pflags2(=seq2msk(seqhis)) and pflags  MISMATCH    num_msk_mismatch: 20252 
    [{__init__            :evt.py    :296} INFO     - ] B 
    [{load                :ab.py     :411} INFO     - ] 



    In [3]: a.pflags                                                                                                                                                                                  
    Out[3]: A([ 6168,  7192,  4104, ...,  6168,  4104, 22608], dtype=uint32)

    In [4]: a.pflags2                                                                                                                                                                                 
    Out[4]: A([6168, 7192, 4104, ..., 6168, 4104, 6224], dtype=uint64)

    In [5]: w = np.where( a.pflags != a.pflags2 )                                                                                                                                                     

    In [6]: w                                                                                                                                                                                         
    Out[6]: (array([   13,    21,    24, ..., 79989, 79990, 79999]),)

    In [7]: a.pflags[w]                                                                                                                                                                               
    Out[7]: A([22608, 39024, 38992, ..., 22640, 38992, 22608], dtype=uint32)

    In [8]: a.pflags2[w]                                                                                                                                                                              
    Out[8]: A([6224, 6256, 6224, ..., 6256, 6224, 6224], dtype=uint64)

    In [9]: np.unique(a.pflags[w])                                                                                                                                                                    
    Out[9]: A([22608, 22640, 23632, 23664, 38992, 39024, 40016, 40048], dtype=uint32)

    In [10]: np.unique(a.pflags2[w])                                                                                                                                                                  
    Out[10]: A([6224, 6256, 7248, 7280], dtype=uint64)



    In [4]: a.hismask.label(np.unique(a.pflags[w]))                                                                                                                                                   
    Out[4]: 
    ['EX|TO|BT|SD|RE',
     'EX|TO|BT|SD|SC|RE',
     'EX|TO|BT|BR|SD|RE',
     'EX|TO|BT|BR|SD|SC|RE',
     'EC|TO|BT|SD|RE',
     'EC|TO|BT|SD|SC|RE',
     'EC|TO|BT|BR|SD|RE',
     'EC|TO|BT|BR|SD|SC|RE']

    In [5]: a.hismask.label(np.unique(a.pflags2[w]))                                                                                                                                                  
    Out[5]: ['TO|BT|SD|RE', 'TO|BT|SD|SC|RE', 'TO|BT|BR|SD|RE', 'TO|BT|BR|SD|SC|RE']



