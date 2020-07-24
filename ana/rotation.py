#!/usr/bin/env python
"""
rotate.py 
===========

Make sense of GDML physvol/rotation and global to local transforms

::

     71423       <physvol copynumber="11336" name="pLPMT_Hamamatsu_R128600x353fc90">
     71424         <volumeref ref="HamamatsuR12860lMaskVirtual0x3290b70"/>
     71425         <position name="pLPMT_Hamamatsu_R128600x353fc90_pos" unit="mm" x="-7148.9484" y="17311.741" z="-5184.2567"/>
     71426         <rotation name="pLPMT_Hamamatsu_R128600x353fc90_rot" unit="deg" x="-73.3288783033161" y="-21.5835981926051" z="-96.2863976680901"/>
     71427       </physvol>


::

    epsilon:src blyth$ grep \"rotation\" *.cc
    G4GDMLReadDefine.cc:      if (tag=="rotation") { RotationRead(child); } else
    G4GDMLReadParamvol.cc:      if (tag=="rotation") { VectorRead(child,rotation); } else
    G4GDMLReadSolids.cc:      if (tag=="rotation") { VectorRead(child,rotation); } else
    G4GDMLReadSolids.cc:      if (tag=="rotation") { VectorRead(child,rotation); } else
    G4GDMLReadStructure.cc:     else if (tag=="rotation")
    G4GDMLReadStructure.cc:      if (tag=="rotation")
    epsilon:src blyth$ pwd
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/src


::

    g4-cls Rotation

    336   HepRotation & rotateX(double delta);
    337   // Rotation around the x-axis; equivalent to R = RotationX(delta) * R
    338 
    339   HepRotation & rotateY(double delta);
    340   // Rotation around the y-axis; equivalent to R = RotationY(delta) * R
    341 
    342   HepRotation & rotateZ(double delta);
    343   // Rotation around the z-axis; equivalent to R = RotationZ(delta) * R
    344 


"""

import os, sympy as sp, numpy as np
from sympy import pprint as pp


def tr_inverse(tt):
    """
    decompose translate rotate matrix by inspection, 
    negate the translation and transpose the rotation

        In [167]: tr
        Out[167]: 
        array([[   -0.1018,    -0.9243,     0.3679,     0.    ],
               [    0.2466,    -0.3817,    -0.8908,     0.    ],
               [    0.9638,     0.    ,     0.2668,     0.    ],
               [   -0.003 ,     0.0127, 19433.9994,     1.    ]])

        In [168]: np.dot( hit, tr )
        Out[168]: array([-112.6704,  165.9216,  109.6381,    1.    ])


    """
    it = np.eye(4)
    it[3,:3] = -tt[3,:3]

    ir = np.eye(4)
    ir[:3,:3] = tt[:3,:3].T

    tr = np.dot(it,ir)
    return tr 


class Instance(object):
    def __init__(self, ridx):
        """
        epsilon:1 blyth$ inp GMergedMesh/?/iidentity.npy GMergedMesh/?/itransforms.npy
        a :                                  GMergedMesh/1/iidentity.npy :        (25600, 5, 4) : a4a7deb934cae243b9181c80ddc1066b : 20200719-2129 
        b :                                GMergedMesh/1/itransforms.npy :        (25600, 4, 4) : 29a7bf21dabfd4a6f9228fadb7edabca : 20200719-2129 
        c :                                  GMergedMesh/2/iidentity.npy :        (12612, 6, 4) : 4423ba6434c39aff488e6784df468ae1 : 20200719-2129 
        d :                                GMergedMesh/2/itransforms.npy :        (12612, 4, 4) : 766b1e274449b0d9f2ecc35d58b52a71 : 20200719-2129 
        e :                                  GMergedMesh/3/iidentity.npy :         (5000, 6, 4) : 52c59e1bb3179c404722c2df4c26ac81 : 20200719-2129 
        f :                                GMergedMesh/3/itransforms.npy :         (5000, 4, 4) : 1ff4e96acee67137c4740b05e6684c93 : 20200719-2129 
        g :                                  GMergedMesh/4/iidentity.npy :         (2400, 6, 4) : 08846aa446e53c50c1a7cea89674a398 : 20200719-2129 
        h :                                GMergedMesh/4/itransforms.npy :         (2400, 4, 4) : aafe0245a283080c130d8575b7a83e67 : 20200719-2129 

        #. iidentity is now reshaped shortly after creation to have same item count as itransforms
        """
        tt = np.load(os.path.expandvars("$GC/GMergedMesh/%d/itransforms.npy" % ridx))
        ii = np.load(os.path.expandvars("$GC/GMergedMesh/%d/iidentity.npy" % ridx))

        assert tt.shape[1:] == (4,4)
        assert len(ii.shape) == 3 and ii.shape[2] == 4 
        assert len(tt) == len(ii)
        nvol = ii.shape(1)   # physvol per instance

        self.ii = ii
        self.tt = tt 
        self.nvol = nvol 

    def find_instance_index(self, pmtid):
        """
        Using vol 0 corresponds to the outer volume of the instance :
        but the copyNo is duplicated for all the volumes of the instance
        so all 0:nvol are the same.
        """ 
        return np.where( self.ii[:,0,3] == pmtid )[0][0]
    def find_instance_transform(self, pmtid):
        ix = self.find_instance_index(pmtid)
        return self.tt[ix]
    def find_instance_transform_inverse(self, pmtid):
        tt = self.find_instance_transform(pmtid)
        return tr_inverse(tt)

    def find_local_pos(self, pmtid, global_pos):
        assert global_pos.shape == (4,)
        tr = self.find_instance_transform_inverse(pmtid)  
        local_pos = np.dot( global_pos, tr )
        return local_pos



def three_to_four(M3):
    assert M3.shape == (3,3)
    M4 = sp.zeros(4)
    for i in range(3):
        for j in range(3):
            M4[i*4+j] = M3.row(i)[j]
        pass
    M4[15] = 1
    return M4
pass


alpha,beta,gamma = sp.symbols("alpha beta gamma")

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



deg = np.pi/180.
v_rot = [(alpha,-73.3288783033161*deg),(beta,-21.5835981926051*deg),(gamma, -96.2863976680901*deg)]
v_pos = [(tx, -7148.9484),(ty,17311.741), (tz,-5184.2567)]

lhit0 = np.array([-112.67072395684227,165.92175413608675,109.63878699927591,1])  # from debug session in ProcessHits
v_lhit0 = [(x,lhit0[0]), (y,lhit0[1]), (z,lhit0[2])]

hit = np.array([-7250.504552589168,17122.963751776308,-5263.596996014085, 1])  # global hit position 
v_hit = [(x,hit[0]),(y,hit[1]),(z,hit[2]),(w,hit[3])]



pmtid = 11336     # "BP=junoSD_PMT_v2::ProcessHits tds" debug session, see jnu/opticks-junoenv-hit.rst
it = Instance(3)  # ridx 3 is Hamamatsu PMTs
tt = it.find_instance_transform(pmtid)
tr = it.find_instance_transform_inverse(pmtid)
lhit = it.find_local_pos(pmtid, hit )

assert np.allclose(lhit0, lhit), lhit0-lhit 


zhit = lhit[2]
rhit = np.sqrt(lhit[0]*lhit[0]+lhit[1]*lhit[1])
# 249 185  axes of cathode ellipsoid for Hamamatsu (see ana/gpmt.py)
one = ((zhit*zhit)/(185.*185.))+((rhit*rhit)/(249.*249.))   ## hmm generic way to get major axes of cathode ellipsoid ?




Rx = three_to_four(sp.rot_axis1(alpha))
IRx = three_to_four(sp.rot_axis1(-alpha))

Ry = three_to_four(sp.rot_axis2(beta))
IRy = three_to_four(sp.rot_axis2(-beta))

Rz = three_to_four(sp.rot_axis3(gamma))
IRz = three_to_four(sp.rot_axis3(-gamma))

R0 = Rz*Ry*Rx
IR0 = IRx*IRy*IRz    # NB : reversed order 
assert IR0.transpose() == R0       # the inverse of a rotation matrix is its transpose
assert R0.transpose() == IR0 

R1 = Rx*Ry*Rz
IR1 = IRz*IRy*IRx
assert IR1.transpose() == R1 
assert R1.transpose() == IR1 







# row3 as translation is used for simpler matching with glm/OpenGL standard practice when serializing transforms
# (col3 as translation is the other possibility) 
R1T = sp.Matrix([R1.row(0), R1.row(1), R1.row(2), T.row(3)]) 
R1T_check = R1*T 

assert R1T == R1T_check 
assert P*R1*T == P*R1T  
# so that is rotate then translate 
#    as rotations are around the origin that is appropriate for orienting   
#    a PMT before translating it into place 

"""
IR1T 
Matrix([
[cos(beta)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma) - sin(gamma)*cos(alpha),  sin(alpha)*sin(gamma) + sin(beta)*cos(alpha)*cos(gamma), 0],
[sin(gamma)*cos(beta), sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma), -sin(alpha)*cos(gamma) + sin(beta)*sin(gamma)*cos(alpha), 0],
[          -sin(beta),                                    sin(alpha)*cos(beta),                                     cos(alpha)*cos(beta), 0],
[                  tx,                                                      ty,                                                       tz, 1]])

"""



IR1T = sp.Matrix([IR1.row(0), IR1.row(1), IR1.row(2), T.row(3)])  # matrix in terms of alpha,beta,gamma,tx,ty,tz

M = IR1T.subs(v_rot+v_pos)



match = np.allclose( np.array(M, dtype=np.float32 ), tt )
assert match


MI = np.array(M.inv(), dtype=np.float64)

amx = np.abs(MI-tr).max()    # not close enough for np.allclose
assert amx < 1e-3




T2 = sp.Matrix([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [tx,ty,tz,1]])
 
assert T == T2, (T, T2)


IT = sp.Matrix([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [-tx,-ty,-tz,1]])
 







"""
     71423       <physvol copynumber="11336" name="pLPMT_Hamamatsu_R128600x353fc90">
     71424         <volumeref ref="HamamatsuR12860lMaskVirtual0x3290b70"/>
     71425         <position name="pLPMT_Hamamatsu_R128600x353fc90_pos" unit="mm" x="-7148.9484" y="17311.741" z="-5184.2567"/>
     71426         <rotation name="pLPMT_Hamamatsu_R128600x353fc90_rot" unit="deg" x="-73.3288783033161" y="-21.5835981926051" z="-96.2863976680901"/>
     71427       </physvol>
"""


if 0:
    print("\nRx")
    pp(Rx)
    print("\nRy")
    pp(Ry)
    print("\nRz")
    pp(Rz)


def rotateX():
    """
    This demonstrates that HepRotation::rotateX is multiplying to the rhs::

        Rxyz*Rx

    66 HepRotation & HepRotation::rotateX(double a) {    // g4-cls Rotation
    67   double c1 = std::cos(a);
    68   double s1 = std::sin(a);
    69   double x1 = ryx, y1 = ryy, z1 = ryz;
    70   ryx = c1*x1 - s1*rzx;
    71   ryy = c1*y1 - s1*rzy;
    72   ryz = c1*z1 - s1*rzz;
    73   rzx = s1*x1 + c1*rzx;
    74   rzy = s1*y1 + c1*rzy;
    75   rzz = s1*z1 + c1*rzz;
    76   return *this;
    77 }
    """
    pass

    x1,y1,z1 = sp.symbols("x1,y1,z1")
    v_ry = [(ryx,x1),(ryy,y1),(ryz,z1)]
       
    Xlhs = (Rx*R).subs(v_ry)
    Xrhs = (R*Rx).subs(v_ry)   # HepRotation::rotateX is multiplying on the rhs 


    print(rotateX.__doc__)

    print("\nR")
    pp(R)  

    print("\nv_ry")
    pp(v_ry)

    print("\nXrhs = (R*Rx).subs(v_ry) ")
    pp(Xrhs)  

    #print("\nXlhs = (Rx*R).subs(v_ry) ")  clearly rotateX not doing this
    #pp(Xlhs)  

def rotateY():
    """
    This demonstrates that HepRotation::rotateY is multiplying to the rhs::

        Rxyz*Ry

    079 HepRotation & HepRotation::rotateY(double a){
     80   double c1 = std::cos(a);
     81   double s1 = std::sin(a);
     82   double x1 = rzx, y1 = rzy, z1 = rzz;
     83   rzx = c1*x1 - s1*rxx;
     84   rzy = c1*y1 - s1*rxy;
     85   rzz = c1*z1 - s1*rxz;
     86   rxx = s1*x1 + c1*rxx;
     87   rxy = s1*y1 + c1*rxy;
     88   rxz = s1*z1 + c1*rxz;
     89   return *this;
     90 }
    """
    x1,y1,z1 = sp.symbols("x1,y1,z1")
    v_rz = [(rzx,x1),(rzy,y1),(rzz,z1)]
 
    Ylhs = (Ry*R).subs(v_rz)
    Yrhs = (R*Ry).subs(v_rz) 

    print(rotateY.__doc__)

    print("\nR")
    pp(R)  

    print("\nv_rz")
    pp(v_rz)

    print("\nYrhs = (R*Ry).subs(v_rz) ")
    pp(Yrhs)  



def rotateZ():
    """
    This demonstrates that HepRotation::rotateZ is multiplying to the rhs::

        Rxyz*Rz


    092 HepRotation & HepRotation::rotateZ(double a) {
     93   double c1 = std::cos(a);
     94   double s1 = std::sin(a);
     95   double x1 = rxx, y1 = rxy, z1 = rxz;
     96   rxx = c1*x1 - s1*ryx;
     97   rxy = c1*y1 - s1*ryy;
     98   rxz = c1*z1 - s1*ryz;
     99   ryx = s1*x1 + c1*ryx;
    100   ryy = s1*y1 + c1*ryy;
    101   ryz = s1*z1 + c1*ryz;
    102   return *this;
    103 }
    """

    x1,y1,z1 = sp.symbols("x1,y1,z1")
    v_rx = [(rxx,x1),(rxy,y1),(rxz,z1)]
 
    Zlhs = (Rz*R).subs(v_rx)
    Zrhs = (R*Rz).subs(v_rx) 

    print(rotateZ.__doc__)

    print("\nR")
    pp(R)  

    print("\nv_rx")
    pp(v_rx)

    print("\nZrhs = (R*Rz).subs(v_rx) ")
    pp(Zrhs)  

  


def G4GDMLReadStructure():
    """

    289 void G4GDMLReadStructure::
    290 PhysvolRead(const xercesc::DOMElement* const physvolElement,
    291             G4AssemblyVolume* pAssembly)
    292 {
    ...
    372    G4Transform3D transform(GetRotationMatrix(rotation).inverse(),position);
    373    transform = transform*G4Scale3D(scale.x(),scale.y(),scale.z());


    132 G4RotationMatrix
    133 G4GDMLReadDefine::GetRotationMatrix(const G4ThreeVector& angles)
    134 {
    135    G4RotationMatrix rot;
    136 
    137    rot.rotateX(angles.x());
    138    rot.rotateY(angles.y());
    139    rot.rotateZ(angles.z());
    140    rot.rectify();  // Rectify matrix from possible roundoff errors
    141 
    142    return rot;
    143 }


    g4-cls Transform3D (icc)

    029 inline
     30 Transform3D::Transform3D(const CLHEP::HepRotation & m, const CLHEP::Hep3Vector & v) {
     31   xx_= m.xx(); xy_= m.xy(); xz_= m.xz();
     32   yx_= m.yx(); yy_= m.yy(); yz_= m.yz();
     33   zx_= m.zx(); zy_= m.zy(); zz_= m.zz();
     34   dx_= v.x();  dy_= v.y();  dz_= v.z();
     35 }


    NB the order  (rotateX, rotateY, rotateZ).inverse()


    In [18]: t[3218]        # use the instance index to give the instance transform : rot matrix and tlate look familiar
    Out[18]: 
    array([[   -0.1018,     0.2466,     0.9638,     0.    ],
           [   -0.9243,    -0.3817,     0.    ,     0.    ],
           [    0.3679,    -0.8908,     0.2668,     0.    ],
           [-7148.948 , 17311.74  , -5184.257 ,     1.    ]], dtype=float32)


    In [70]: pp((Rx*Ry*Rz).transpose().subs(v_rot)*T.subs(v_pos))
     -0.101820513179743   0.24656591428434    0.963762332221457    0
     -0.924290430171623  -0.381689927419044  8.32667268468867e-17  0
     0.367858374634817   -0.890796300632177   0.266762379264878    0
         -7148.9484          17311.741            -5184.2567       1


    In [52]: (P*IT*R1).subs(v_hit+v_pos+v_rot)    ### << MATCHES <<
    Out[52]: Matrix([[-112.670723956843, 165.921754136086, 109.638786999275, 1.0]])

    In [100]: (P*IT*Rx*Ry*Rz).subs(v_hit+v_pos+v_rot)
    Out[100]: Matrix([[-112.670723956843, 165.921754136086, 109.638786999275, 1]])



    In [101]: (P*Rx*Ry*Rz*IT).subs(v_hit+v_pos+v_rot)
    Out[101]: Matrix([[7036.28119031864, -17145.8318197405, -14140.1045440872, 1]])









    (gdb) p local_pos
    $7 = {dx = -112.67072395684227, dy = 165.92175413608675, dz = 109.63878699927591, static tolerance = 2.22045e-14}
    (gdb) p trans



    ## 4th row matches the GDML 

     71423       <physvol copynumber="11336" name="pLPMT_Hamamatsu_R128600x353fc90">
     71424         <volumeref ref="HamamatsuR12860lMaskVirtual0x3290b70"/>
     71425         <position name="pLPMT_Hamamatsu_R128600x353fc90_pos" unit="mm" x="-7148.9484" y="17311.741" z="-5184.2567"/>
     71426         <rotation name="pLPMT_Hamamatsu_R128600x353fc90_rot" unit="deg" x="-73.3288783033161" y="-21.5835981926051" z="-96.2863976680901"/>
     71427       </physvol>


             { rxx = -0.10182051317974285,   rxy = -0.92429043017162327,   rxz =  0.36785837463481702, 
               ryx =  0.24656591428433955,   ryy = -0.38168992741904467,   ryz = -0.89079630063217707, 
               rzx =  0.96376233222145669,   rzy =  0,                     rzz =  0.26676237926487772, 
               tx =  -0.0035142754759363015, ty =   0.012573876562782971,  tz  =  19434.000031086449}   

             THIS IS THE INVERSE TRANSFORM 


    From examples/UseGeant4/UseGeant.cc

    dbg_affine_trans   Transformation: 
    rx/x,y,z: -0.101821 -0.92429 0.367858
    ry/x,y,z: 0.246566 -0.38169 -0.890796
    rz/x,y,z: 0.963762 0 0.266762
    tr/x,y,z: -0.00351428 0.0125739 19434


    """
    R0 = Rx*Ry*Rz
    R0T = (Rx*Ry*Rz).T

    R1 = Rz*Ry*Rx
    R1T = (Rz*Ry*Rx).T
     

    print(G4GDMLReadStructure.__doc__)

    if 0:
        print("\nR0 = Rx*Ry*Rz \n")
        pp(R0)
        pp(R)
        pp(R0.subs(va))
    pass

    print("\nR0T = (Rx*Ry*Rz).T   THIS MATCHES THE ROTATION PART OF THE ITRANSFORM \n")
    pp(R0T)
    pp(R)
    pp(R0T.subs(v_rot))


    if 0:
        print("\nR1 = Rz*Ry*Rx\n")
        pp(R1)
        pp(R)
        pp(R1.subs(va))
    pass



def translate():
    """
    Using col3 for the translation as opposed to glm/OpenGL approach 
    of row3 for the translation is a transpose of the translation matrix, 
    which means need to transpose the point for shape consistency
    and multiply in other order.
    """  

    print("P1")
    pp(P1)
    print("T")
    pp(T)
    P1_T = P1*T
    print("P1*T")
    pp(P1_T)
    print("P1*T*T")
    pp(P1*T*T)
    P1_T_reverse_transpose_check = (T.T*P1.T).T
    print("(T.T*P1.T).T")
    pp(P1_T_reverse_transpose_check)
    assert P1_T_reverse_transpose_check == P1_T


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

    print("R")
    pp(R)
    print("T")
    pp(T)
    print("T*R : row3 has translation and rotation mixed up : ie translation first and then rotation which depends  ")
    pp(T*R)
    print("R*T : familiar row3 as translation : that means rotate then translate ")
    pp(R*T)

    print("RT")
    pp(RT)
    assert RT == R*T


    print("P1")
    pp(P1)
    print("P*RT : notice that the translation just gets added to rotated coordinates : ie rotation first and then translation")
    pp(P*RT)

    P_RT = P*RT
    print("P*RT.subs(v_rid) : setting rotation to identity ")
    pp(P_RT.subs(v_rid))




if __name__ == '__main__':
 
    #rotateX()
    #rotateY()
    #rotateZ()

    #translate()
    #translate_rotate()

    G4GDMLReadStructure()



