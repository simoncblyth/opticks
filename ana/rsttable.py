#!/usr/bin/env python

import logging
import numpy as np
log = logging.getLogger(__name__)

class RSTTable(object):
     def divider(self, widths, char="-"):
         """+----+----------------+----------------+"""
         return "+".join([""]+list(map(lambda j:char*widths[j], range(len(widths))))+[""]) 

     @classmethod
     def Render(cls, t, labels, wids, hfmt, rfmt, pre, post):
         """
         :param t:  2D array of np.object that are the content of the table 
         :param labels: list of string labels for the header
         :param wids: list of integer widths 
         :param hfmt: list of header format strings
         :param rfmt: list of row format strings
         :param pre:  list of strings that prepend the header and row formats
         :param post: list of strings that postpend the header and row formats 
         """
         tab = RSTTable(t)  
         tab.labels = labels
         tab.pre  = np.array( list(map(len,pre)),  dtype=np.int32 )
         tab.post = np.array( list(map(len,post)), dtype=np.int32 )
         tab.wids = np.array( list(map(int,wids)), dtype=np.int32 )
         tab.hfmt = [ pre[i]+hfmt[i]+post[i] for i in range(len(hfmt)) ]
         tab.rfmt = [ pre[i]+rfmt[i]+post[i] for i in range(len(rfmt)) ]
         tab.wids += tab.pre 
         tab.wids += tab.post 
         return str(tab)

     @classmethod
     def Rdr(cls, t, labels, wid=10, hfm="%10s", rfm="%10.4f", pre_="", post_=""  ):
         """
         :param t: 2D array "table" of np.object items to populate the RST table
         :param labels: list of labels
         :param wid: int width that is repeated across all columns
         :param hfm: string header format string
         :param rfm: string row format string
         :param pre_: string that prepends the header and row formats 
         :param post_: string that postpends the header and row formats 

         *Rdr* provides a simpler interface to creating RST tables than *Render*, which it uses
         via *np.repeat* repetitions to generate the detailed input arrays needed by *Render*  
         """
         nlab = len(labels)

         wids = np.repeat( wid, nlab ) 
         hfmt = np.repeat( hfm, nlab )
         rfmt = np.repeat( rfm, nlab )
         pre  = np.repeat( pre_, nlab )
         post  = np.repeat( post_, nlab )

         rst = cls.Render(t, labels, wids, hfmt, rfmt, pre, post )
         return rst 

     def __init__(self, t):
         self.t = t  

     def __str__(self):
         """
         Builds the RST table line by line 
         """
         nrow = self.t.shape[0]
         ncol = self.t.shape[1]    

         assert len(self.hfmt) == ncol
         assert len(self.rfmt) == ncol
         assert len(self.labels) == ncol
         assert len(self.wids) == ncol
         assert len(self.pre) == ncol
         
         hfmt = "|".join( [""]+self.hfmt+[""])
         rfmt = "|".join( [""]+self.rfmt+[""])

         lines = []
         lines.append(self.divider(self.wids, "-")) 
         lines.append(hfmt % tuple(self.labels))
         lines.append(self.divider(self.wids, "="))
         for i in range(nrow):
             lines.append(rfmt % tuple(self.t[i]))
             lines.append(self.divider(self.wids,"-"))   
         pass
         return "\n".join(lines)    



def test_Render():
     log.info("test_Render")
     t = np.empty( [2,2], dtype=np.object )
     t[0] = ["a", "b" ]
     t[1] = ["c", "d" ]

     labels = ["A", "B"] 
     wids = [ 10, 10]
     hfmt = [ "%10s", "%10s" ]   # header format    
     rfmt = [ "%10s", "%10s" ]   # row format
     pre  = [ "" ,    "   " ]
     post = [ "" ,    "   " ]

     rst = RSTTable.Render(t, labels, wids, hfmt, rfmt, pre, post )
     print(rst)


def test_Rdr2x2():
     log.info("test_Rdr2x2")
     t = np.zeros( [2,2], dtype=np.object )
     t[0] = [ 1.1 , 1.2 ]
     t[1] = [ 2.2 , 3.1 ]
     labels = ["A", "B"] 
     rst = RSTTable.Rdr(t, labels )
     print(rst)


def test_Rdr3x3():
     log.info("test_Rdr3x3")
     t = np.zeros( [3,3], dtype=np.object )
     t[0] = [ 1.1 , 1.2 , 1.3 ]
     t[1] = [ 2.2 , 3.1 , 4.1 ]
     t[2] = [ 3.2 , 4.1 , 5.1 ]
     labels = ["A", "B", "C" ] 
     rst = RSTTable.Rdr(t, labels )
     print(rst)


if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     test_Render()
     test_Rdr2x2()
     test_Rdr3x3()


