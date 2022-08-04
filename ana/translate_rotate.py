#!/usr/bin/env python
"""
translate_rotate.py
=====================

Insights from this sympy checking used in::

   U4Transform::Convert_RotateThenTranslate
   U4Transform::Convert_TranslateThenRotate


"""
import os, sympy as sp, numpy as np
from sympy import pprint as pp

def pr(title):
    print("\n\n%s\n" % title)


def translate_rotate():
    """
    P1
     x  y  z  1
    TR
     rxx  ryx  rzx  0 
                      
     rxy  ryy  rzy  0 
                      
     rxz  ryz  rzz  0 
                      
     tx   ty   tz   tw 
    P*TR
     rxx x + rxy y + rxz z + tx w  ryx x + ryy y + ryz z + ty w  rzx x + rzy y + rzz z + tz w  tw w
    P*TR.subs(v_rid)
     tx w + x  ty w + y  tz w + z  tw w

    """

    pr("R")
    pp(R)

    pr("T")
    pp(T)

    pr("T*R : row3 has translation and rotation mixed up : ie translation first and then rotation")
    pp(T*R)

    pr("R*T : familiar row3 as translation : that means rotate then translate ")
    pp(R*T)

    pr("RT")
    pp(RT)

    assert RT == R*T

    pr("P1")
    pp(P1)


    pr("P1*RT : notice that the translation just gets added to rotated coordinates : ie rotation first and then translation")
    pp(P1*RT)


    P_RT = P*RT
    P1_RT = P1*RT

    pr("P1*RT.subs(v_rid) : setting rotation to identity ")
    pp(P1_RT.subs(v_rid))



if __name__ == '__main__':

    row0 = rxx,ryx,rzx,rwx = sp.symbols("rxx,ryx,rzx,rwx")
    row1 = rxy,ryy,rzy,rwy = sp.symbols("rxy,ryy,rzy,rwy")
    row2 = rxz,ryz,rzz,rwz = sp.symbols("rxz,ryz,rzz,rwz")
    row3 = tx,ty,tz,tw     = sp.symbols("tx,ty,tz,tw")

    RTxyz = sp.Matrix([row0,row1,row2,row3])

    v_rid = [ 
       (rxx,1),(ryx,0),(rzx,0),
       (rxy,0),(ryy,1),(rzy,0),
       (rxz,0),(ryz,0),(rzz,1) ]    # identity rotation 

    v_rw = [(rwx,0),(rwy,0),(rwz,0)]
    v_t0 = [(tx,0),(ty,0),(tz,0),(tw,1)] # identity translation
    v_tw = [(tw,1),]    


    RT = RTxyz.subs(v_rw+v_tw)

    R = RTxyz.subs(v_rw+v_t0)

    T = RTxyz.subs(v_rid+v_rw+v_tw)


    x,y,z,w = sp.symbols("x,y,z,w")
    P = sp.Matrix([[x,y,z,w]])

    assert P.shape == (1,4)
    P1 = P.subs([(w,1)])    # position
    P0 = P.subs([(w,0)])    # direction vector

    translate_rotate()


