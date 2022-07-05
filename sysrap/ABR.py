#!/usr/bin/env python

class ABR(object):
    """
    As this takes a general side by side repr approach it belongs in its own module. 
    """
    def __init__(self, A, B, symbol="AB"):
        self.A = A
        self.B = B
        self.symbol = symbol
        self.idx = 0 

    def __call__(self, idx):
        self.idx = idx
        self.A.idx = idx 
        self.B.idx = idx 
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

        lx = cls.MaxLen(l)
        rx = cls.MaxLen(r)
        pass
        fmt_ = "{:%d}{:%d}"   
        fmt = fmt_ % (lx+margin, rx+margin)

        lines = []
        for i in range(nl):
            line = fmt.format(l[i], r[i])    
            lines.append(line)
        pass
        return "\n".join(lines)

    def __repr__(self):
        return self.SideBySide(repr(self.A), repr(self.B)) 


if __name__ == '__main__':
    pass

    


