/*

transforms do the intended when use : vec*matrix 
presumable because optix uses row-major but the matrices
come in as column-major
when only scaling with no translation or rotation pre or post makes no difference

*/

__device__ void transform_test()
{
    int tranIdx = 0 ; 
    if(3*tranIdx+2 >= tranBuffer.size()) return ;  

    Matrix4x4 t = tranBuffer[3*tranIdx+0] ; 
    Matrix4x4 v = tranBuffer[3*tranIdx+1] ; 
    Matrix4x4 q = tranBuffer[3*tranIdx+2] ; 

    const Matrix4x4& T = t ;    // transform
    const Matrix4x4& V = v ;    // inverse
    const Matrix4x4& Q = q ;    // transpose of the inverse

    Matrix4x4 Q2 = V.transpose() ; 


    Matrix4x4 TV = T*V ; 
    Matrix4x4 VT = V*T ; 

    rtPrintf("##test_tranBuffer T(transform)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          T[0], T[1], T[2], T[3],  
          T[4], T[5], T[6], T[7],  
          T[8], T[9], T[10], T[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", T[12], T[13], T[14], T[15] );

    rtPrintf("##test_tranBuffer V(inverse)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          V[0], V[1], V[2], V[3],  
          V[4], V[5], V[6], V[7],  
          V[8], V[9], V[10], V[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", V[12], V[13], V[14], V[15] );

    rtPrintf("##test_tranBuffer Q(inverse.T)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          Q[0], Q[1], Q[2], Q[3],  
          Q[4], Q[5], Q[6], Q[7],  
          Q[8], Q[9], Q[10], Q[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", Q[12], Q[13], Q[14], Q[15] );

    rtPrintf("##test_tranBuffer Q2(inverse.T)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          Q2[0], Q2[1], Q2[2], Q2[3],  
          Q2[4], Q2[5], Q2[6], Q2[7],  
          Q2[8], Q2[9], Q2[10], Q2[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", Q2[12], Q2[13], Q2[14], Q2[15] );






    rtPrintf("##test_tranBuffer TV(~identity)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          TV[0], TV[1], TV[2], TV[3],  
          TV[4], TV[5], TV[6], TV[7],  
          TV[8], TV[9], TV[10], TV[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", TV[12], TV[13], TV[14], TV[15] );

    rtPrintf("##test_tranBuffer VT(~identity)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          VT[0], VT[1], VT[2], VT[3],  
          VT[4], VT[5], VT[6], VT[7],  
          VT[8], VT[9], VT[10], VT[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", VT[12], VT[13], VT[14], VT[15] );


    // test points and vectors
    float4 O = make_float4(  0.f, 0.f, 0.f,   1.f );
    float4 P = make_float4(  1.f, 1.f, 1.f,   1.f );
    float4 N = make_float4( -1.f, -1.f, -1.f, 1.f );
    float4 X = make_float4(  1.f, 0.f, 0.f,   0.f );
    float4 Y = make_float4(  0.f, 1.f, 0.f,   0.f );
    float4 Z = make_float4(  0.f, 0.f, 1.f,   0.f );

    // using the transform

    float4 OT = O * T ;   
    float4 TO = T * O ; 

    float4 PT = P * T ;   // <-ok 
    float4 TP = T * P ; 

    float4 NT = N * T ;   // <-ok
    float4 TN = T * N ; 

    float4 XT = X * T ; 
    float4 TX = T * X ; 

    float4 YT = Y * T ; 
    float4 TY = T * Y ; 

    float4 ZT = Z * T ; 
    float4 TZ = T * Z ; 


    rtPrintf(" O %8.3f %8.3f %8.3f %8.3f  O*T %8.3f %8.3f %8.3f %8.3f    T*O %8.3f %8.3f %8.3f %8.3f  \n",
          O.x,   O.y,  O.z,  O.w,  
          OT.x, OT.y, OT.z, OT.w,
          TO.x, TO.y, TO.z, TO.w);

    rtPrintf(" P %8.3f %8.3f %8.3f %8.3f  P*T %8.3f %8.3f %8.3f %8.3f    T*P %8.3f %8.3f %8.3f %8.3f  \n",
          P.x,   P.y,  P.z,  P.w,  
          PT.x, PT.y, PT.z, PT.w,
          TP.x, TP.y, TP.z, TP.w);

    rtPrintf(" N %8.3f %8.3f %8.3f %8.3f  N*T %8.3f %8.3f %8.3f %8.3f    T*N %8.3f %8.3f %8.3f %8.3f  \n",
          N.x,   N.y,  N.z,  N.w,  
          NT.x, NT.y, NT.z, NT.w,
          TN.x, TN.y, TN.z, TN.w);

    rtPrintf(" X %8.3f %8.3f %8.3f %8.3f  X*T %8.3f %8.3f %8.3f %8.3f    T*X %8.3f %8.3f %8.3f %8.3f  \n",
          X.x,   X.y,  X.z,  X.w,  
          XT.x, XT.y, XT.z, XT.w,
          TX.x, TX.y, TX.z, TX.w);

    rtPrintf(" Y %8.3f %8.3f %8.3f %8.3f  Y*T %8.3f %8.3f %8.3f %8.3f    T*Y %8.3f %8.3f %8.3f %8.3f  \n",
          Y.x,   Y.y,  Y.z,  Y.w,  
          YT.x, YT.y, YT.z, YT.w,
          TY.x, TY.y, TY.z, TY.w);

    rtPrintf(" Z %8.3f %8.3f %8.3f %8.3f  Z*T %8.3f %8.3f %8.3f %8.3f    T*Z %8.3f %8.3f %8.3f %8.3f  \n",
          Z.x,   Z.y,  Z.z,  Z.w,  
          ZT.x, ZT.y, ZT.z, ZT.w,
          TZ.x, TZ.y, TZ.z, TZ.w);


    // using the inverse transform

    float4 OV = O * V ;  // <-ok
    float4 VO = V * O ; 

    float4 PV = P * V ;   // <-ok 
    float4 VP = V * P ; 

    float4 NV = N * V ;   // <-ok
    float4 VN = V * N ; 

    float4 XV = X * V ; 
    float4 VX = V * X ; 

    float4 YV = Y * V ; 
    float4 VY = V * Y ; 

    float4 ZV = Z * V ; 
    float4 VZ = V * Z ; 

    // using the transpose of the inverse transform


    float4 OQ = O * Q ;  // <-ok
    float4 QO = Q * O ; 


    float4 PQ = P * Q ;   // <-ok 
    float4 QP = Q * P ; 

    float4 NQ = N * Q ;   // <-ok
    float4 QN = Q * N ; 

    float4 XQ = X * Q ; 
    float4 QX = Q * X ; 

    float4 YQ = Y * Q ; 
    float4 QY = Q * Y ; 

    float4 ZQ = Z * Q ; 
    float4 QZ = Q * Z ; 




    rtPrintf(" O %8.3f %8.3f %8.3f %8.3f  O*V %8.3f %8.3f %8.3f %8.3f    V*O %8.3f %8.3f %8.3f %8.3f  \n",
          O.x,   O.y,  O.z,  O.w,  
          OV.x, OV.y, OV.z, OV.w,
          VO.x, VO.y, VO.z, VO.w);

    rtPrintf(" P %8.3f %8.3f %8.3f %8.3f  P*V %8.3f %8.3f %8.3f %8.3f    V*P %8.3f %8.3f %8.3f %8.3f  \n",
          P.x,   P.y,  P.z,  P.w,  
          PV.x, PV.y, PV.z, PV.w,
          VP.x, VP.y, VP.z, VP.w);

    rtPrintf(" N %8.3f %8.3f %8.3f %8.3f  N*V %8.3f %8.3f %8.3f %8.3f    V*N %8.3f %8.3f %8.3f %8.3f  \n",
          N.x,   N.y,  N.z,  N.w,  
          NV.x, NV.y, NV.z, NV.w,
          VN.x, VN.y, VN.z, VN.w);

    rtPrintf(" X %8.3f %8.3f %8.3f %8.3f  X*V %8.3f %8.3f %8.3f %8.3f    V*X %8.3f %8.3f %8.3f %8.3f  \n",
          X.x,   X.y,  X.z,  X.w,  
          XV.x, XV.y, XV.z, XV.w,
          VX.x, VX.y, VX.z, VX.w);

    rtPrintf(" Y %8.3f %8.3f %8.3f %8.3f  Y*V %8.3f %8.3f %8.3f %8.3f    V*Y %8.3f %8.3f %8.3f %8.3f  \n",
          Y.x,   Y.y,  Y.z,  Y.w,  
          YV.x, YV.y, YV.z, YV.w,
          VY.x, VY.y, VY.z, VY.w);

    rtPrintf(" Z %8.3f %8.3f %8.3f %8.3f  Z*V %8.3f %8.3f %8.3f %8.3f    V*Z %8.3f %8.3f %8.3f %8.3f  \n",
          Z.x,   Z.y,  Z.z,  Z.w,  
          ZV.x, ZV.y, ZV.z, ZV.w,
          VZ.x, VZ.y, VZ.z, VZ.w);




    rtPrintf(" O %8.3f %8.3f %8.3f %8.3f  O*Q %8.3f %8.3f %8.3f %8.3f    Q*O %8.3f %8.3f %8.3f %8.3f  \n",
          O.x,   O.y,  O.z,  O.w,  
          OQ.x, OQ.y, OQ.z, OQ.w,
          QO.x, QO.y, QO.z, QO.w);

    rtPrintf(" O %8.3f %8.3f %8.3f %8.3f  V*O %8.3f %8.3f %8.3f %8.3f   O*V %8.3f %8.3f %8.3f %8.3f  #\n",
          O.x,   O.y,  O.z,  O.w,  
          VO.x, VO.y, VO.z, VO.w,
          OV.x, OV.y, OV.z, OV.w);


    rtPrintf(" P %8.3f %8.3f %8.3f %8.3f  P*Q %8.3f %8.3f %8.3f %8.3f    Q*P %8.3f %8.3f %8.3f %8.3f  \n",
          P.x,   P.y,  P.z,  P.w,  
          PQ.x, PQ.y, PQ.z, PQ.w,
          QP.x, QP.y, QP.z, QP.w);

    rtPrintf(" P %8.3f %8.3f %8.3f %8.3f  V*P %8.3f %8.3f %8.3f %8.3f    P*V %8.3f %8.3f %8.3f %8.3f  #\n",
          P.x,   P.y,  P.z,  P.w,  
          VP.x, VP.y, VP.z, VP.w,
          PV.x, PV.y, PV.z, PV.w);



    rtPrintf(" N %8.3f %8.3f %8.3f %8.3f  N*Q %8.3f %8.3f %8.3f %8.3f    Q*N %8.3f %8.3f %8.3f %8.3f  \n",
          N.x,   N.y,  N.z,  N.w,  
          NQ.x, NQ.y, NQ.z, NQ.w,
          QN.x, QN.y, QN.z, QN.w);

    rtPrintf(" X %8.3f %8.3f %8.3f %8.3f  X*Q %8.3f %8.3f %8.3f %8.3f    Q*X %8.3f %8.3f %8.3f %8.3f  \n",
          X.x,   X.y,  X.z,  X.w,  
          XQ.x, XQ.y, XQ.z, XQ.w,
          QX.x, QX.y, QX.z, QX.w);

    rtPrintf(" Y %8.3f %8.3f %8.3f %8.3f  Y*Q %8.3f %8.3f %8.3f %8.3f    Q*Y %8.3f %8.3f %8.3f %8.3f  \n",
          Y.x,   Y.y,  Y.z,  Y.w,  
          YQ.x, YQ.y, YQ.z, YQ.w,
          QY.x, QY.y, QY.z, QY.w);

    rtPrintf(" Z %8.3f %8.3f %8.3f %8.3f  Z*Q %8.3f %8.3f %8.3f %8.3f    Q*Z %8.3f %8.3f %8.3f %8.3f  \n",
          Z.x,   Z.y,  Z.z,  Z.w,  
          ZQ.x, ZQ.y, ZQ.z, ZQ.w,
          QZ.x, QZ.y, QZ.z, QZ.w);


}


