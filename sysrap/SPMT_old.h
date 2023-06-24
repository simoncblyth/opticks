// SPMT_old.h 

    // HMM: annoying shuffling : allows C4MultiLayerStack.h to stay very independent
    /*
    ss.ls[0].nr = spec.q0.f.x ; 

    ss.ls[1].nr = spec.q1.f.x ; 
    ss.ls[1].ni = spec.q1.f.y ; 
    ss.ls[1].d  = spec.q1.f.z ;
 
    ss.ls[2].nr = spec.q2.f.x ; 
    ss.ls[2].ni = spec.q2.f.y ; 
    ss.ls[2].d  = spec.q2.f.z ;

    ss.ls[3].nr = spec.q3.f.x ; 
    */

    /*
    NOW INCORPORATED THE BELOW INTO stack.calc

    const float _si = stack.ll[0].st.real() ; // sqrt(one - minus_cos_theta*minus_cos_theta) 
    float E_s2 = _si > 0.f ? dot_pol_cross_mom_nrm/_si : 0.f ;
    E_s2 *= E_s2;

    // E_s2 : S-vs-P power fraction : signs make no difference as squared
    // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 

    const float one = 1.f ;
    const float S = E_s2 ;
    const float P = one - S ;

    const float T = S*stack.art.T_s + P*stack.art.T_p ;  // matched with TransCoeff see sysrap/tests/stmm_vs_sboundary_test.cc
    const float R = S*stack.art.R_s + P*stack.art.R_p ;
    const float A = S*stack.art.A_s + P*stack.art.A_p ;
    //const float A1 = one - (T+R);  // note that A1 matches A 

    stack.art.xx = A ; 
    stack.art.yy = R ; 
    stack.art.zz = T ; 
    stack.art.ww = S ; 
    */ 


