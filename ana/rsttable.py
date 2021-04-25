#!/usr/bin/env python

import numpy as np

class RSTTable(object):
     def divider(self, widths, char="-"):
         """+----+----------------+----------------+"""
         return "+".join([""]+list(map(lambda j:char*widths[j], range(len(widths))))+[""]) 

     @classmethod
     def Render(cls, t, labels, wids, hfmt, rfmt, pre):
         tab = RSTTable(t)  
         tab.labels = labels
         tab.pre  = np.array( list(map(len,pre)), dtype=np.int32 )
         tab.wids = np.array( list(map(int,wids)), dtype=np.int32 )
         tab.hfmt = [ pre[i]+hfmt[i] for i in range(len(hfmt)) ]
         tab.rfmt = [ pre[i]+rfmt[i] for i in range(len(rfmt)) ]
         tab.wids += tab.pre 
         return str(tab)

     def __init__(self, t):
         self.t = t  

     def __str__(self):
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


if __name__ == '__main__':

     t = np.empty( [2,2], dtype=np.object )
     t[0] = ["a", "b" ]
     t[1] = ["c", "d" ]

     labels = ["A", "B"] 
     wids = [ 10, 10]
     hfmt = [ "%10s", "%10s" ]
     rfmt = [ "%10s", "%10s" ]
     pre  = [ "" ,    "   " ]

     rst = RSTTable.Render(t, labels, wids, hfmt, rfmt, pre )

     print(rst)

