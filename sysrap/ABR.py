#!/usr/bin/env python

class ABR(object):
    """
    As this takes a general side by side repr approach it belongs in its own module. 
    """
    def __init__(self, A, B, symbol="AB"):
        self.A = A
        self.B = B
        self.symbol = symbol

    def _get_idx(self):
        return self.A.idx 
    def _set_idx(self, idx):
        self.A.idx = idx 
        self.B.idx = idx 
    idx = property(_get_idx, _set_idx)

    def _get_flavor(self):
        return self.A.flavor
    def _set_flavor(self, flavor):
        self.A.flavor = flavor 
        self.B.flavor = flavor 
    flavor = property(_get_flavor, _set_flavor)


    def __call__(self, idx):
        self.idx = idx
        self.flavor = "call"
        return self  
    def __getitem__(self, idx):
        self.idx = idx
        self.flavor = "getitem"
        return self  

   
    @classmethod
    def MaxLen(cls, lines):
        mx = 0 
        for i in range(len(lines)):
            if len(lines[i]) > mx: mx = len(lines[i]) 
        pass
        return mx 

    @classmethod
    def LeftRightPaddedSplit(cls, lhs, rhs):
        """
        :param lhs: text delimited by newlines
        :param rhs: text delimited by newlines
        :return l,r: lhs and rhs lines padded to be the same length 
        """
        l = lhs.split("\n")
        r = rhs.split("\n") 
        nl = len(l)
        nr = len(r)
        if nl == nr:
             pass
        elif nl > nr:
            r.extend( ["" for _ in range(nl-nr)] )  
            nr = len(r)
        elif nr > nl:
            l.extend( ["" for _ in range(nr-nl)] )  
            nl = len(l)
        else:
            pass
        pass
        assert nl == nr 
        return l, r 
 
    @classmethod
    def SideBySide(cls, lhs, rhs, margin=10):
        l, r = cls.LeftRightPaddedSplit(lhs, rhs)
        nl = len(l)
        nr = len(r)
        assert nl == nr

        lx_ = cls.MaxLen(l)
        rx_ = cls.MaxLen(r)
        lim = 100 

        lx = min(lim,lx_) 
        rx = min(lim,rx_)

        pass
        fmt_ = "{:%d}{:%d}"   
        fmt = fmt_ % (lx+margin, rx+margin)

        lines = []
        for i in range(nl):
            line = fmt.format(l[i], r[i])    
            lines.append(line.rstrip())
        pass
        return "\n".join(lines)

    def identification(self):
        ia  = self.A.identification() if hasattr(self.A, "identification") else ""
        ib  = self.B.identification() if hasattr(self.B, "identification") else ""
        return "\n".join(filter(None,[ia,ib]))

    def __repr__(self):
        return "\n".join([self.identification(), self.SideBySide(repr(self.A), repr(self.B))])


if __name__ == '__main__':
    pass

    


