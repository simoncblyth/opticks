#!/usr/bin/env python

import numpy as np

ENTER = 0
EXIT = 1
MISS = 2 

STATE = { ENTER:"ENTER", EXIT:"EXIT", MISS:"MISS" }


def dump(pending):
    print(" pending : %s " % bin(pending))
    for i in range(8):
        if (pending >> i) & 0x1 == 1:
            print("bit %i is set " % i)
        pass
    pass

"""

    |-----|
        |-------|
              |------|
             
"""


class MockSolid(object):
    def __init__(self):
        n = 8 
        a = np.zeros( [n,2], dtype=np.float32 )
        c = np.linspace(0,70,8)  
        a[:,0] = c - 6     
        a[:,1] = c + 6  

        a[-1] = (100,110)  # make a disjoint 

        self.a = a 
        self.n = len(a) 

    def intersect_pending(self, tmin ):
        """
        pending approach has to keep repeating getting entrance distances
        better to collect the enters and sort to avoid that 
        """

        a = self.a 
        pending = np.uint8( (0x1 << 8) - 1 )
        farthest_exit = 0

        stage = "init"   
        print("  %20s :   pending  : %s  farthest_exit : %s    " % (stage, bin(pending), farthest_exit) )

        for i in range(len(a)):
             t, state = self.intersect_sub(i, tmin)
             if state == EXIT:
                 if t > farthest_exit: 
                     farthest_exit = t   
                 pass
             pass   
             mask = np.uint8(0x1 << i)
             if state == EXIT or state == MISS: pending &= ~mask 
        pass

        stage = "after pass1"
        print("  %20s :   pending  : %s  farthest_exit : %s    " % (stage, bin(pending), farthest_exit) )

        loop = 0 

        while pending:
            loop += 1 
            initial_farthest_exit = farthest_exit
            for i in range(len(a)): 
                mask = np.uint8(0x1 << i)
                if (pending >> i) & 0x1 == 1:
                    t_enter, state = self.intersect_sub(i, tmin)
                    assert state == ENTER 
                    print("t_enter %s  farthest_exit %s " % (t_enter,farthest_exit) )
                    if t_enter < farthest_exit:
                        t_advanced = t_enter+0.0001
                        t_exit, state = self.intersect_sub(i, t_advanced)
                        assert state == EXIT
                        print("t_exit %s  farthest_exit %s  " % (t_exit,farthest_exit) )
                        if t_exit > farthest_exit: 
                            farthest_exit = t_exit         
                        pass
                        pending &= ~mask
                    pass
                pass
            pass
            stage = "loop %d " % loop
            print("  %20s :   pending  : %s  farthest_exit : %s    " % (stage, bin(pending), farthest_exit) )
            if farthest_exit == initial_farthest_exit: break 
        pass
        return farthest_exit

    def intersect_sorting(self, tmin ):
        a = self.a 
        farthest_exit = 0

        stage = "init"   
        print("  %20s :  farthest_exit : %s    " % (stage, farthest_exit) )

        enter = np.zeros(8, dtype=np.float32) 

        # 1st pass : find farthest_exit and collect enter distances 
        for i in range(len(a)):
             enter[i] = np.nan    # argsort treats nan and inf as large values
             t, state = self.intersect_sub(i, tmin)
             if state == EXIT:
                 if t > farthest_exit: 
                     farthest_exit = t   
                 pass
             elif state == ENTER:
                 enter[i] = t 
             pass   
        pass

        stage = "after pass1"
        print("  %20s :   s  farthest_exit : %s    " % (stage, farthest_exit) )

        # only need to sort the enter, but need to maintain state on all 
        idx = np.argsort(enter)   

        for i in range(len(a)):
            j = idx[i]
            t_enter = enter[j]
            if not np.isnan(t_enter):
                if t_enter < farthest_exit:
                    t_advanced = t_enter+0.0001
                    t_exit, state = self.intersect_sub(j, t_advanced)
                    assert state == EXIT
                    if t_exit > farthest_exit: 
                        farthest_exit = t_exit         
                    pass
                pass
            pass
        pass
        stage = "fin"
        print("  %20s :   s  farthest_exit : %s    " % (stage, farthest_exit) )
        return farthest_exit

    def intersect_sub(self, i, t):
        """

        t < t0 
            Enter 
        t0 <= t < t1 
            Exit
        t > t1
            Miss

        ::
               
                   +------------+ 
                   |            | 
                   |            | 
                   |            | 
                   |            | 
                   +------------+ 
                   t0           t1 

        """
        a = self.a
        n = self.n
        assert i < n  

        tbeg = t 

        t0, t1 = a[i]

        if t < t0:
            t = t0 
            state = ENTER 
        elif t < t1:
            t = t1
            state = EXIT
        else:
            t = np.inf
            state = MISS
        pass
        print("intersect_sub i %2d  tbeg %10.4f  t0 %10.4f t1 %10.4f   state %s " % (i, tbeg, t0, t1, STATE[state] ))
        return t, state


def test_pending():
    pending = np.uint8( (0x1 << 8) - 1 )
    dump(pending)
    i = 0 
    while pending:
        pending &= ~(0x1 << i)
        dump(pending)
        i += 1 
    pass


if __name__ == '__main__':
    ms = MockSolid()
    #ms.intersect_pending(0)
    ms.intersect_sorting(0)

    #test_pending()

