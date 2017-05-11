#!/usr/bin/env python
"""

* http://nghiaho.com/?page_id=846



NB Opposite Sign Convention to GLM
--------------------------------------

* http://planning.cs.uiuc.edu/node102.html
* http://planning.cs.uiuc.edu/node103.html


  | r11 r12 r13 |
  | r21 r22 r23 |
  | r31 r32 r33 |


    Rz (yaw)  conterclockwise  "alpha"

         cosZ  -sinZ  0
         sinZ   cosZ  0
            0      0  1

    Ry (pitch) counterclockwise  "beta"

         cosY    0   sinY
            0    1     0
         -sinY   0   cosY

    Rx (roll) counterclockwise  "gamma"
      
        
         1   0   0
         0 cosX -sinX
         0 sinX cosX



     yawPitchRoll
      Rz  Ry  Rx


     Rzyx = Rz(alpha).Ry(beta).Rx(gamma)  
                               ^^^^^^ roll first

     First roll Rx, then pitch Ry then finally yaw Rz



       11: cosZ cosY     12: cosZ sinY sinX - sinZ cosX      13: cosZ sinY cosX + sinZ sinX
     
       21: sinZ cosY     22: sinZ sinY sinX + cosZ cosX      23: sinZ sinY cosX - cosZ sinX
  
       31: -sinY         32: cosY sinX                       33: cosY cosX




       r32/r33 = cosY sinX / cosY cosX = tanX 

       r32^2 + r33^2 =   cosY^2 sinX^2 + cosY^2 cosX^2 = cosY^2 

       -r31/sqrt(r32^2 + r33^2) =  sinY / cosY = tanY 

       r21/r11 = tanZ



        r11^2 + r21^2  = cosZ^2 cosY^2 + sinZ^2 cosY^2 = cosY^2    "cosb"^2

        -r31/sqrt(r11^2 + r21^2) =  sinY / cosY = tanY 


              cosY->0 => sinY=>1   
                      ... DONT FOLLOW THE LEAP TO sinZ = 0, cosZ = 1   

        -r23/r22 =  -(sinZ sinY cosX - cosZ sinX) / (sinZ sinY sinX + cosZ cosX )
         
                    how is this meant to yield tanY ??? ... perhaps a mal-assumption made here that sinY->0 ???

                       cosZ sinX / cosZ cosX ->  tanX        (if sinY->0, BUT IT DOESNT ???)



* https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf
* ~/opticks_refs/Extracting_Euler_Angles.pdf

"""
import numpy as np


def GetAngles(m):
    """

     51 G4ThreeVector G4GDMLWriteDefine::GetAngles(const G4RotationMatrix& mtx)
     52 {
     53    G4double x,y,z;
     54    G4RotationMatrix mat = mtx;
     55    mat.rectify();   // Rectify matrix from possible roundoff errors
     56 
     57    // Direction of rotation given by left-hand rule; clockwise rotation
     58 
     59    static const G4double kMatrixPrecision = 10E-10;
     60    const G4double cosb = std::sqrt(mtx.xx()*mtx.xx()+mtx.yx()*mtx.yx());
     ..                                       r11^2 + r21^2
     61 
     62    if (cosb > kMatrixPrecision)
     63    {
     64       x = std::atan2(mtx.zy(),mtx.zz());   
     ..                         r32      r33   
     65       y = std::atan2(-mtx.zx(),cosb);
     ..                        -r31 
     66       z = std::atan2(mtx.yx(),mtx.xx());
     ..                         r21     r11
     67    }
     68    else
     69    {
     70       x = std::atan2(-mtx.yz(),mtx.yy());
     ..                        -r23      r22           
     71       y = std::atan2(-mtx.zx(),cosb);   
     ..                                 huh division by smth very small... unhealthy                                             
     ..                        -r31     sqrt(r11^2 + r21^2)
     72       z = 0.0;
     73    }
     74 
     75    return G4ThreeVector(x,y,z);
     76 }

    """
    pass




def extractEulerAnglesXYZ(M, unit=np.pi/180., dtype=np.float32):
    """
    https://github.com/jzrake/glm/commit/d3313421c664db5bd1b672d39ba3faec0d430117
    https://github.com/g-truc/glm/blob/master/glm/gtx/euler_angles.inl
    https://gamedev.stackexchange.com/questions/50963/how-to-extract-euler-angles-from-transformation-matrix

    ~/opticks_refs/Extracting_Euler_Angles.pdf 

    ::

        template<typename T>
        GLM_FUNC_DECL void extractEulerAngleXYZ(mat<4, 4, T, defaultp> const & M,
                                                T & t1,
                                                T & t2,
                                                T & t3)
        {
            float T1 = glm::atan2<T, defaultp>(M[2][1], M[2][2]);

            float C2 = glm::sqrt(M[0][0]*M[0][0] + M[1][0]*M[1][0]);



            float T2 = glm::atan2<T, defaultp>(-M[2][0], C2);
            float S1 = glm::sin(T1);
            float C1 = glm::cos(T1);

            float T3 = glm::atan2<T, defaultp>(S1*M[0][2] - C1*M[0][1], C1*M[1][1] - S1*M[1][2  ]);
            t1 = -T1;
            t2 = -T2;
            t3 = -T3;
        }



    """

    T1 = np.arctan2(M[2][1], M[2][2]);
    C2 = np.sqrt(M[0][0]*M[0][0] + M[1][0]*M[1][0]);
    T2 = np.arctan2(-M[2][0], C2);
    S1 = np.sin(T1);
    C1 = np.cos(T1);

    T3 = np.arctan2(S1*M[0][2] - C1*M[0][1], C1*M[1][1] - S1*M[1][2  ]);
    t1 = -T1;
    t2 = -T2;
    t3 = -T3;

    return np.array([t1/unit,t2/unit,t3/unit], dtype=dtype)






def yawPitchRoll(yaw, pitch, roll, dtype=np.float32):
    """

    yaw: Z
    


    https://github.com/g-truc/glm/blob/master/glm/gtx/euler_angles.inl
    ::

        template<typename T>
            GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> yawPitchRoll
            (
                T const & yaw,
                T const & pitch,
                T const & roll
            )
            {
                T tmp_ch = glm::cos(yaw);
                T tmp_sh = glm::sin(yaw);
                T tmp_cp = glm::cos(pitch);
                T tmp_sp = glm::sin(pitch);
                T tmp_cb = glm::cos(roll);
                T tmp_sb = glm::sin(roll);

                mat<4, 4, T, defaultp> Result;
                Result[0][0] = tmp_ch * tmp_cb + tmp_sh * tmp_sp * tmp_sb;
                Result[0][1] = tmp_sb * tmp_cp;
                Result[0][2] = -tmp_sh * tmp_cb + tmp_ch * tmp_sp * tmp_sb;
                Result[0][3] = static_cast<T>(0);
                Result[1][0] = -tmp_ch * tmp_sb + tmp_sh * tmp_sp * tmp_cb;
                Result[1][1] = tmp_cb * tmp_cp;
                Result[1][2] = tmp_sb * tmp_sh + tmp_ch * tmp_sp * tmp_cb;
                Result[1][3] = static_cast<T>(0);
                Result[2][0] = tmp_sh * tmp_cp;
                Result[2][1] = -tmp_sp;
                Result[2][2] = tmp_ch * tmp_cp;
                Result[2][3] = static_cast<T>(0);
                Result[3][0] = static_cast<T>(0);
                Result[3][1] = static_cast<T>(0);
                Result[3][2] = static_cast<T>(0);
                Result[3][3] = static_cast<T>(1);
                return Result;
            }

    """

    tmp_ch = np.cos(yaw);
    tmp_sh = np.sin(yaw);
    tmp_cp = np.cos(pitch);
    tmp_sp = np.sin(pitch);
    tmp_cb = np.cos(roll);
    tmp_sb = np.sin(roll);

    Result = np.eye(4, dtype=dtype)

    Result[0][0] = tmp_ch * tmp_cb + tmp_sh * tmp_sp * tmp_sb;
    Result[0][1] = tmp_sb * tmp_cp;
    Result[0][2] = -tmp_sh * tmp_cb + tmp_ch * tmp_sp * tmp_sb;
    Result[0][3] = 0;
    Result[1][0] = -tmp_ch * tmp_sb + tmp_sh * tmp_sp * tmp_cb;
    Result[1][1] = tmp_cb * tmp_cp;
    Result[1][2] = tmp_sb * tmp_sh + tmp_ch * tmp_sp * tmp_cb;
    Result[1][3] = 0;
    Result[2][0] = tmp_sh * tmp_cp;
    Result[2][1] = -tmp_sp;
    Result[2][2] = tmp_ch * tmp_cp;
    Result[2][3] = 0;
    Result[3][0] = 0;
    Result[3][1] = 0;
    Result[3][2] = 0;
    Result[3][3] = 1;
    return Result;



def eulerAngleX(angleX, dtype=np.float32):
    """

    * opposite sign to *roll* of http://planning.cs.uiuc.edu/node102.html

    /usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtx/euler_angles.inl::

         35     template <typename T>
         36     GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> eulerAngleX
         37     (
         38         T const & angleX
         39     )
         40     {
         41         T cosX = glm::cos(angleX);
         42         T sinX = glm::sin(angleX);
         43    
         44         return tmat4x4<T, defaultp>(
         45             T(1), T(0), T(0), T(0),
         46             T(0), cosX, sinX, T(0),
         47             T(0),-sinX, cosX, T(0),
         48             T(0), T(0), T(0), T(1));
         49     }
         50 

    """
    m = np.eye(4, dtype=dtype)
    cosX = np.cos(angleX);
    sinX = np.sin(angleX);

    m[0] = [1.,    0.,   0., 0.]
    m[1] = [0.,  cosX, sinX, 0.]
    m[2] = [0., -sinX, cosX, 0.]
    m[3] = [0.,    0.,   0., 1.]

    return m


def eulerAngleY(angleY, dtype=np.float32):
    """
    * opposite sign to *pitch* of http://planning.cs.uiuc.edu/node102.html

    /usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtx/euler_angles.inl

    ::

         51     template <typename T>
         52     GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> eulerAngleY
         53     (
         54         T const & angleY
         55     )
         56     {
         57         T cosY = glm::cos(angleY);
         58         T sinY = glm::sin(angleY);
         59 
         60         return tmat4x4<T, defaultp>(
         61             cosY,   T(0),   -sinY,  T(0),
         62             T(0),   T(1),   T(0),   T(0),
         63             sinY,   T(0),   cosY,   T(0),
         64             T(0),   T(0),   T(0),   T(1));
         65     }


    """
    m = np.eye(4, dtype=dtype)
    cosY = np.cos(angleY);
    sinY = np.sin(angleY);

    m[0] = [cosY,  0., -sinY, 0.]
    m[1] = [0.,    1.,    0., 0.]
    m[2] = [sinY,  0.,  cosY, 0.]
    m[3] = [0.,    0.,    0., 1.]

    return m


def eulerAngleZ(angleZ, dtype=np.float32):
    """

    * opposite sign to *yaw* of http://planning.cs.uiuc.edu/node102.html

    /usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtx/euler_angles.inl

    ::

         67     template <typename T>
         68     GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> eulerAngleZ
         69     (
         70         T const & angleZ
         71     )
         72     {
         73         T cosZ = glm::cos(angleZ);
         74         T sinZ = glm::sin(angleZ);
         75 
         76         return tmat4x4<T, defaultp>(
         77             cosZ,   sinZ,   T(0), T(0),
         78             -sinZ,  cosZ,   T(0), T(0),
         79             T(0),   T(0),   T(1), T(0),
         80             T(0),   T(0),   T(0), T(1));
         81     }
             

    """
    m = np.eye(4, dtype=dtype)
    cosZ = np.cos(angleZ);
    sinZ = np.sin(angleZ);

    m[0] = [ cosZ,  sinZ,  0., 0.]
    m[1] = [-sinZ,  cosZ,  0., 0.]
    m[2] = [    0.,    0., 1., 0.]
    m[3] = [    0.,    0., 0., 1.]

    return m


def eulerAngleXYZ(t123, unit=np.pi/180., dtype=np.float32):
    """
    ::

        In [14]: eulerAngleXYZ([45,0,0])
        Out[14]: 
        array([[ 1.    ,  0.    ,  0.    ,  0.    ],
               [-0.    ,  0.7071,  0.7071,  0.    ],
               [ 0.    , -0.7071,  0.7071,  0.    ],
               [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=float32)

        In [15]: eulerAngleXYZ([0,45,0])
        Out[15]: 
        array([[ 0.7071,  0.    , -0.7071,  0.    ],
               [-0.    ,  1.    ,  0.    ,  0.    ],
               [ 0.7071, -0.    ,  0.7071,  0.    ],
               [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=float32)

        In [16]: eulerAngleXYZ([0,0,45])
        Out[16]: 
        array([[ 0.7071,  0.7071,  0.    ,  0.    ],
               [-0.7071,  0.7071,  0.    ,  0.    ],
               [ 0.    , -0.    ,  1.    ,  0.    ],
               [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=float32)



        In [11]: extractEulerAnglesXYZ(eulerAngleXYZ([45,0,0]))
        Out[11]: array([ 45.,   0.,   0.], dtype=float32)

        In [12]: extractEulerAnglesXYZ(eulerAngleXYZ([0,45,0]))
        Out[12]: array([  0.,  45.,  -0.], dtype=float32)

        In [13]: extractEulerAnglesXYZ(eulerAngleXYZ([0,0,45]))
        Out[13]: array([  0.,   0.,  45.], dtype=float32)



    https://github.com/g-truc/glm/blob/master/glm/gtx/euler_angles.inl

    ::

        template<typename T>
            GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleXYZ
            (
             T const & t1,
             T const & t2,
             T const & t3
             )
            {
                T c1 = glm::cos(-t1);
                T c2 = glm::cos(-t2);
                T c3 = glm::cos(-t3);
                T s1 = glm::sin(-t1);
                T s2 = glm::sin(-t2);
                T s3 = glm::sin(-t3);
                
                mat<4, 4, T, defaultp> Result;
                Result[0][0] = c2 * c3;
                Result[0][1] =-c1 * s3 + s1 * s2 * c3;
                Result[0][2] = s1 * s3 + c1 * s2 * c3;
                Result[0][3] = static_cast<T>(0);
                Result[1][0] = c2 * s3;
                Result[1][1] = c1 * c3 + s1 * s2 * s3;
                Result[1][2] =-s1 * c3 + c1 * s2 * s3;
                Result[1][3] = static_cast<T>(0);
                Result[2][0] =-s2;
                Result[2][1] = s1 * c2;
                Result[2][2] = c1 * c2;
                Result[2][3] = static_cast<T>(0);
                Result[3][0] = static_cast<T>(0);
                Result[3][1] = static_cast<T>(0);
                Result[3][2] = static_cast<T>(0);
                Result[3][3] = static_cast<T>(1);
                return Result;
            }

    """

    a = np.asarray(t123, dtype=dtype)
    a *= unit 

    t1 = a[0]
    t2 = a[1]
    t3 = a[2]

    c1 = np.cos(-t1);
    c2 = np.cos(-t2);
    c3 = np.cos(-t3);
    s1 = np.sin(-t1);
    s2 = np.sin(-t2);
    s3 = np.sin(-t3);
                
    Result = np.eye(4, dtype=dtype);
    Result[0][0] = c2 * c3;
    Result[0][1] =-c1 * s3 + s1 * s2 * c3;
    Result[0][2] = s1 * s3 + c1 * s2 * c3;
    Result[0][3] = 0;
    Result[1][0] = c2 * s3;
    Result[1][1] = c1 * c3 + s1 * s2 * s3;
    Result[1][2] =-s1 * c3 + c1 * s2 * s3;
    Result[1][3] = 0;
    Result[2][0] =-s2;
    Result[2][1] = s1 * c2;
    Result[2][2] = c1 * c2;
    Result[2][3] = 0;
    Result[3][0] = 0;
    Result[3][1] = 0;
    Result[3][2] = 0;
    Result[3][3] = 1;
    return Result;



if __name__ == '__main__':
    pass

    # YXZ
    #m = yawPitchRoll( ) 


    t1 = 10.
    t2 = 20.
    t3 = 30.

    a0 = np.array([t1,t2,t3])

    m = eulerAngleXYZ(a0, unit=np.pi/180.  )

    a1 = extractEulerAnglesXYZ( m, unit=np.pi/180. )

    





