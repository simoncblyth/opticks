#!/usr/bin/env python

import numpy as np

class ArrayReprMixin(object):
    """
    ArrayReprMixin
    ================

    This mixin base class is added with::

        class PhotonDV(ArrayReprMixin, object):
         
    And used from the repr with::
    
        def __repr__(self):
            return self.MakeRepr(self.dv, symbol="pdv.dv")

    It simply presents the array provided with rows and columns annotated, providing eg::

              pdv.dv :         1e-06 1e-05  0.0001 0.001  0.01   0.1    1      10     100    1000    

                 pos : array([[   47,   117,  1732,  4412,  2710,   965,    16,     1,     0,     0],
                time :        [ 2746,  5430,  1724,    96,     4,     0,     0,     0,     0,     0],
                 mom :        [ 6404,  2937,   647,    11,     1,     0,     0,     0,     0,     0],
                 pol :        [ 9995,     1,     1,     3,     0,     0,     0,     0,     0,     0],
                  wl :        [10000,     0,     0,     0,     0,     0,     0,     0,     0,     0]], dtype=uint32)

    The adopting class is required to provide two class level lists containing the annotation::

        COLUMN_LABEL
        ROW_LABEL  

    """
    @classmethod
    def FindColumnPos(cls, s):
        """
        :param s: first line of np.array repr string
        :return pos: array of string position indices of first spaces in each column

        HMM: for smaller arrays often first column has no space
        'array([[1., 0., 0., 0.],'  

        """
        if s is None: s = 'array([[   47,   117,  1732,  4412,  2710,   965,    16,     1,     0,     0],'
        pos = []
        for i in range(len(s)):
            if len(pos) == 0:
                if i > 0 and s[i-1] == "[" and s[i] != "[": 
                    pos.append(i) 
                pass
            else:
                if i > 0 and s[i-1] != " " and s[i] == " ": 
                    pos.append(i) 
                pass
            pass
        pass
        return pos 

    @classmethod
    def ShowColumnPos(cls, s):
        """
        :param s: first line of np.array repr string
        :return s2: string with the first spaces in each column replaced with a "+"

        """
        if s is None: s = 'array([[   47,   117,  1732,  4412,  2710,   965,    16,     1,     0,     0],'
        pos = cls.FindColumnPos(s)
        c = np.zeros( len(s), dtype=np.int8 )     
        for i in range(len(s)):c[i] = ord(s[i])  
        for p in pos: c[p] = ord("+")
        s2 = "".join(list(map(chr,c)))   
        return s2

    @classmethod
    def MakeHdr(cls, s):
        """
        :param s: first line of np.array repr string
        :return hdr: 
        """
        pos = cls.FindColumnPos(s)
        assert len(cls.COLUMN_LABEL) == len(pos) 

        h = np.zeros(len(s), dtype=np.int8 )  
        h.fill(ord(" ")) 
        for i,p in enumerate(pos):
            label = cls.COLUMN_LABEL[i]
            for j in range(len(label)):
                h[p+j] = ord(label[j])
            pass
        pass
        hdr = "".join(list(map(chr,h)))   
        return hdr 

    @classmethod
    def MakeRepr(cls, arr, symbol="arr"):  
        srep = repr(arr).split("\n")
        hdr = cls.MakeHdr(srep[0])
        fmt = "%20s : %s" 
        lines = []
        lines.append("")
        lines.append(fmt % (symbol, hdr)) 
        lines.append("")
        for i, line in enumerate(srep):
            item = cls.ROW_LABEL[i] 
            lines.append(fmt % (item, line)) 
        pass
        return "\n".join(lines)



class Example(ArrayReprMixin, object):
    """
    In [1]: run array_repr_mixin.py

                    eg.a :         x  y   z   w    

                       a : array([[1., 0., 0., 0.],
                       b :        [0., 1., 0., 0.],
                       c :        [0., 0., 1., 0.],
                       d :        [0., 0., 0., 1.]])

    """
    ROW_LABEL = ["a","b","c","d" ]
    COLUMN_LABEL = ["x","y","z","w" ]
 
    def __init__(self):
        self.a = np.eye(4)
    def __repr__(self):
        return self.MakeRepr(self.a, "eg.a")

if __name__ == '__main__':
    eg = Example()
    print(eg)




