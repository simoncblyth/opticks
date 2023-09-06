leap_to_new_workflow_insitu_comparison
=========================================


Insitu Comparison Workflow (from j/issues/opticksMode3-contents-comparison.rst)
----------------------------------------------------------------------------------


Rebuild offline after Opticks rebuild::

    jo
    ./build_Debug.sh 

After C4 updates usually need nuclear rebuild 

* HMM: think I fixed that using fixed locations for the "next" C4 release  


Updating the bash function on workstation, after preparing on laptop::

    jxv               # laptop, for example change "ntds" ipho stats to 10k 
    jxscp             # laptop, scp jx.bash to remote 
    jxf               # workstation, pick up updated jx.bash functions 


Running the bi-simulation::

    ntds3_noxjsjfa    # workstation, run opticksMode:3 doing both optical simulations in one invokation

Pull back from workstation to laptop::

    GEOM                                # laptop/workstation : check GEOM setting is eg V1J011 for current full geom
    GEOM tmpget                         # laptop, pullback the paired SEvt 
    MODE=3 PICK=AB ~/j/ntds/ntds3.sh    # laptop, run analysis ntds3.py loading two SEvt into ipython for comparison, plotting



Initial using new workflow with V1J010 : which does U4Tree__DISABLE_OSUR_IMPLICIT=1 SO 3inch PMT surface is discrepant
------------------------------------------------------------------------------------------------------------------------

::

    QCF qcf :  
    a.q 100000 b.q 100000 lim slice(None, None, None) 
    c2sum :  2063.1819 c2n :   114.0000 c2per:    18.0981  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  2063.18/114:18.098 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT BT BT SA                                                                              ' ' 0' ' 37494  37425' ' 0.0635' '     8      3']
     [' 1' 'TO BT BT BT BT SD                                                                              ' ' 1' ' 30866  30874' ' 0.0010' '     4      4']
     [' 2' 'TO BT BT BT BT BT SA                                                                           ' ' 2' ' 12382  12477' ' 0.3630' '  9412   9096']
     [' 3' 'TO BT BT BT BT BT SR SA                                                                        ' ' 3' '  3810   3794' ' 0.0337' ' 11059  10892']
     [' 4' 'TO BT BT BT BT BT SR SR SA                                                                     ' ' 4' '  1998   1996' ' 0.0010' ' 10899  10879']
     [' 5' 'TO BT BT AB                                                                                    ' ' 5' '   884    893' ' 0.0456' '    26     28']
     [' 6' 'TO BT BT BT BT BT SR SR SR SA                                                                  ' ' 6' '   572    563' ' 0.0714' ' 14725  14727']
     [' 7' 'TO BT BT BT BT BR BT BT BT BT BT BT AB                                                         ' ' 7' '   473    440' ' 1.1928' '  3182   4895']
     [' 8' 'TO BT BT BT BT AB                                                                              ' ' 8' '   319    352' ' 1.6230' '   651     46']
     [' 9' 'TO BT BT BT BT BR BT BT BT BT BT BT SD                                                         ' ' 9' '   326    342' ' 0.3832' '  5262   5279']
     ['10' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                                   ' '10' '   326    332' ' 0.0547' '  7444   7463']
     ['11' 'TO BT BT BT BT BT SR BR SA                                                                     ' '11' '   309    328' ' 0.5667' ' 33584  33575']
     ['12' 'TO BT BT BT BT BR BT BT BT BT BT AB                                                            ' '12' '   321     52' '193.9973' '  1021  17293']
     ['13' 'TO BT BT BT BT BR BT BT BT BT BT SA                                                            ' '13' '    24    318' '252.7368' '  4471   1017']
     ['14' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                                   ' '14' '   312    263' ' 4.1757' '  8147   8138']
     ['15' 'TO BT BT BT BT BR BT BT BT BT AB                                                               ' '15' '   279    264' ' 0.4144' '   646    940']
     ['16' 'TO BT BT BT BT BT SR SR SR BR SA                                                               ' '16' '   212    240' ' 1.7345' ' 14749  14746']
     ['17' 'TO BT BT BR BT BT BT SA                                                                        ' '17' '    10    238' '209.6129' '  2991     17']
     ['18' 'TO BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT SA                                             ' '18' '     0    197' '197.0000' '    -1  15508']
     ['19' 'TO BT BT BT BR BT BT BT BT SA                                                                  ' '19' '     9    194' '168.5961' '  3510    194']
     ['20' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT BT BT BT BT SD                                       ' '20' '   190    171' ' 1.0000' ' 16931  17569']
     ['21' 'TO BT BT BT BR BT BT BT BT AB                                                                  ' '21' '   187      4' '175.3351' '   206  22156']
     ['22' 'TO BT BT BR BT BT BT AB                                                                        ' '22' '   183      3' '174.1935' '     2  39342']
     ['23' 'TO BT BT BT BT BT SR SR SR BR BR SR SA                                                         ' '23' '   168    166' ' 0.0120' ' 15414  15495']
     ['24' 'TO BT BT BT BT BT BR SR SA                                                                     ' '24' '   148    164' ' 0.8205' '  9351   9255']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## bzero: A histories not in B 
    [['38' 'TO BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT BT AB                                          ' '38' '    91      0' '91.0000' ' 16654     -1']
     ['44' 'TO BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT AB                                             ' '44' '    83      0' '83.0000' ' 15529     -1']
     ['56' 'TO BT BT BT BT BT SR SR BT BT BT BT BT BT BT BT SD                                             ' '56' '    56      0' '56.0000' ' 26920     -1']
     ['63' 'TO BT BT BT SA                                                                                 ' '63' '    42      0' '42.0000' ' 49820     -1']
     ['76' 'TO BT BT BT SD                                                                                 ' '76' '    34      0' '34.0000' ' 49823     -1']
     ['81' 'TO BT BT BT BT BT BT BT BT BT BT BT AB                                                         ' '81' '    31      0' '31.0000' '  9297     -1']
     ['92' 'TO BT BT BT BT BT SR SR BT BT BT BT BT BT BT BT SA                                             ' '92' '    26      0' ' 0.0000' ' 27573     -1']
     ['105' 'TO BT BT BT BT BT SR SR BT BT BT BT BT BT BT SD                                                ' '105' '    22      0' ' 0.0000' ' 26717     -1']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## azero: B histories not in A 
    [['18' 'TO BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT SA                                             ' '18' '     0    197' '197.0000' '    -1  15508']
     ['26' 'TO BT BT BT BT BT SR SR BT BT BT BT BT BT SA                                                   ' '26' '     0    161' '161.0000' '    -1  26558']
     ['77' 'TO BT BT BT BT BT BT BT BT BT BT BT SA                                                         ' '77' '     0     33' '33.0000' '    -1   9210']]
    PICK=CF MODE=3 SEL=0 ~/j/ntds/ntds3.sh 



Pick single photons in A and B to highlight that have "only" histories::

    APID=16654 BPID=15508 MODE=3 PICK=AB ~/j/ntds/ntds3.sh 



Try V1J011 : with OSUR implicits not disabled : Still get 3inch and PMT virtual apex issues
---------------------------------------------------------------------------------------------

That avoids B histories not in A::

    MODE=3 PICK=AB ~/j/ntds/ntds3.sh 

    QCF qcf :  
    a.q 100000 b.q 100000 lim slice(None, None, None) 
    c2sum :  1229.4878 c2n :   112.0000 c2per:    10.9776  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  1229.49/112:10.978 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT BT BT SA                                                                              ' ' 0' ' 37494  37425' ' 0.0635' '     8      3']
     [' 1' 'TO BT BT BT BT SD                                                                              ' ' 1' ' 30866  30874' ' 0.0010' '     4      4']
     [' 2' 'TO BT BT BT BT BT SA                                                                           ' ' 2' ' 12382  12477' ' 0.3630' '  9412   9096']
     [' 3' 'TO BT BT BT BT BT SR SA                                                                        ' ' 3' '  3810   3794' ' 0.0337' ' 11059  10892']
     [' 4' 'TO BT BT BT BT BT SR SR SA                                                                     ' ' 4' '  1998   1996' ' 0.0010' ' 10899  10879']
     [' 5' 'TO BT BT AB                                                                                    ' ' 5' '   884    893' ' 0.0456' '    26     28']
     [' 6' 'TO BT BT BT BT BT SR SR SR SA                                                                  ' ' 6' '   572    563' ' 0.0714' ' 14725  14727']
     [' 7' 'TO BT BT BT BT BR BT BT BT BT BT BT AB                                                         ' ' 7' '   411    440' ' 0.9882' ' 11875   4895']
     [' 8' 'TO BT BT BT BT AB                                                                              ' ' 8' '   319    352' ' 1.6230' '   651     46']
     [' 9' 'TO BT BT BT BT BR BT BT BT BT BT BT SD                                                         ' ' 9' '   326    342' ' 0.3832' '  5262   5279']
     ['10' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                                   ' '10' '   326    332' ' 0.0547' '  7444   7463']
     ['11' 'TO BT BT BT BT BT SR BR SA                                                                     ' '11' '   309    328' ' 0.5667' ' 33584  33575']
     ['12' 'TO BT BT BT BT BR BT BT BT BT BT AB                                                            ' '12' '   321     52' '193.9973' '  1021  17293']
     ['13' 'TO BT BT BT BT BR BT BT BT BT BT SA                                                            ' '13' '    90    318' '127.4118' '  3182   1017']
     ['14' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                                   ' '14' '   312    263' ' 4.1757' '  8147   8138']
     ['15' 'TO BT BT BT BT BR BT BT BT BT AB                                                               ' '15' '   279    264' ' 0.4144' '   646    940']
     ['16' 'TO BT BT BT BT BT SR SR SR BR SA                                                               ' '16' '   212    240' ' 1.7345' ' 14749  14746']
     ['17' 'TO BT BT BR BT BT BT SA                                                                        ' '17' '    74    238' '86.2051' '  1691     17']
     ['18' 'TO BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT SA                                             ' '18' '    93    197' '37.2966' ' 16654  15508']
     ['19' 'TO BT BT BT BR BT BT BT BT SA                                                                  ' '19' '    41    194' '99.6128' '  2286    194']
     ['20' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT BT BT BT BT SD                                       ' '20' '   190    171' ' 1.0000' ' 16931  17569']
     ['21' 'TO BT BT BT BR BT BT BT BT AB                                                                  ' '21' '   187      4' '175.3351' '   206  22156']
     ['22' 'TO BT BT BR BT BT BT AB                                                                        ' '22' '   183      3' '174.1935' '     2  39342']
     ['23' 'TO BT BT BT BT BT SR SR SR BR BR SR SA                                                         ' '23' '   168    166' ' 0.0120' ' 15414  15495']
     ['24' 'TO BT BT BT BT BT BR SR SA                                                                     ' '24' '   148    164' ' 0.8205' '  9351   9255']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## bzero: A histories not in B 
    [['42' 'TO BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT AB                                             ' '42' '    83      0' '83.0000' ' 15529     -1']
     ['60' 'TO BT BT BT SA                                                                                 ' '60' '    42      0' '42.0000' ' 49820     -1']
     ['74' 'TO BT BT BT SD                                                                                 ' '74' '    34      0' '34.0000' ' 49823     -1']
     ['78' 'TO BT BT BT BT BT BT BT BT BT BT BT AB                                                         ' '78' '    31      0' '31.0000' '  9297     -1']
     ['101' 'TO BT BT BT BT BT SR SR BT BT BT BT BT BT BT SD                                                ' '101' '    22      0' ' 0.0000' ' 26717     -1']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## azero: B histories not in A 
    []
    PICK=AB MODE=3 SEL=0 ~/j/ntds/ntds3.sh 



Again V1J011 : with OSUR implicits not disabled AND export Tub3inchPMTV3Manager__VIRTUAL_DELTA_MM=1
-----------------------------------------------------------------------------------------------------


::

    269 double Tub3inchPMTV3Manager::VIRTUAL_DELTA_MM = EGet::Get<double>(__VIRTUAL_DELTA_MM, 1.e-3 );
    270 
    271 void
    272 Tub3inchPMTV3Manager::helper_make_solid()
    273 {
    274     std::cerr
    275         << "Tub3inchPMTV3Manager::helper_make_solid"
    276         << " " << desc() << std::endl
    277         ;
    278 
    279     pmt_solid = m_pmtsolid_maker->GetUnionSolid(GetName() + "_pmt_solid", VIRTUAL_DELTA_MM*mm);
    280     body_solid = m_pmtsolid_maker->GetEllipsoidSolid(GetName() + "_body_solid", 0.);
    281     inner1_solid = m_pmtsolid_maker->GetEllipsoidSolid(GetName()+"_inner1_solid", m_pmt_H, m_photocathode_Z, -1.*m_glass_thickness);
    282     inner2_solid = m_pmtsolid_maker->GetEllipsoidSolid(GetName()+"_inner2_solid", m_photocathode_Z, m_cntr_Z1, -1.*m_glass_thickness);
    283     cntr_solid = m_pmtsolid_maker->GetContainerSolid(GetName()+"_cntr_solid", -1.e-3*mm);
    284 }




::

    QCF qcf :  
    a.q 100000 b.q 100000 lim slice(None, None, None) 
    c2sum :   162.4843 c2n :   108.0000 c2per:     1.5045  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  162.48/108:1.504 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT BT BT SA                                                                              ' ' 0' ' 37494  37620' ' 0.2114' '     8      1']
     [' 1' 'TO BT BT BT BT SD                                                                              ' ' 1' ' 30866  30749' ' 0.2222' '     4     13']
     [' 2' 'TO BT BT BT BT BT SA                                                                           ' ' 2' ' 12382  12416' ' 0.0466' '  9412   8882']
     [' 3' 'TO BT BT BT BT BT SR SA                                                                        ' ' 3' '  3810   3831' ' 0.0577' ' 11059  11054']
     [' 4' 'TO BT BT BT BT BT SR SR SA                                                                     ' ' 4' '  1998   1969' ' 0.2120' ' 10899  10889']
     [' 5' 'TO BT BT AB                                                                                    ' ' 5' '   884    895' ' 0.0680' '    26     20']
     [' 6' 'TO BT BT BT BT BT SR SR SR SA                                                                  ' ' 6' '   572    604' ' 0.8707' ' 14725  14758']
     [' 7' 'TO BT BT BT BT BR BT BT BT BT BT BT AB                                                         ' ' 7' '   411    451' ' 1.8561' ' 11875   5071']
     [' 8' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                                   ' ' 8' '   337    346' ' 0.1186' '  7444   7444']
     [' 9' 'TO BT BT BT BT AB                                                                              ' ' 9' '   319    345' ' 1.0181' '   651     14']
     ['10' 'TO BT BT BT BT BR BT BT BT BT BT BT SD                                                         ' '10' '   314    335' ' 0.6795' '  5262   5252']
     ['11' 'TO BT BT BT BT BR BT BT BT BT BT SA                                                            ' '11' '   332    312' ' 0.6211' '  1021   1025']
     ['12' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                                   ' '12' '   320    289' ' 1.5780' '  8147   8170']
     ['13' 'TO BT BT BT BT BT SR BR SA                                                                     ' '13' '   309    319' ' 0.1592' ' 33584  33568']
     ['14' 'TO BT BT BT BT BR BT BT BT BT AB                                                               ' '14' '   279    248' ' 1.8235' '   646   9164']
     ['15' 'TO BT BT BR BT BT BT SA                                                                        ' '15' '   243    211' ' 2.2555' '     2      2']
     ['16' 'TO BT BT BT BT BT SR SR SR BR SA                                                               ' '16' '   212    239' ' 1.6164' ' 14749  14761']
     ['17' 'TO BT BT BT BR BT BT BT BT SA                                                                  ' '17' '   216    214' ' 0.0093' '   206    226']
     ['18' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT BT BT BT BT SD                                       ' '18' '   190    166' ' 1.6180' ' 16931  11835']
     ['19' 'TO BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT SA                                             ' '19' '   176    187' ' 0.3333' ' 15529  15388']
     ['20' 'TO BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT BT BT SA                                    ' '20' '   148    171' ' 1.6583' ' 17266  16930']
     ['21' 'TO BT BT BT BT BT SR SR SR BR BR SR SA                                                         ' '21' '   168    155' ' 0.5232' ' 15414  15512']
     ['22' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT BT BT BT BT SA                                       ' '22' '   163    167' ' 0.0485' ' 11832  17198']
     ['23' 'TO BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT BT BT SD                                    ' '23' '   149    159' ' 0.3247' ' 16930  16725']
     ['24' 'TO BT BT BT BT BT SR SR BT BT BT BT BT BT SA                                                   ' '24' '   143    151' ' 0.2177' ' 26577  26568']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## bzero: A histories not in B 
    [['55' 'TO BT BT BT SA                                                                                 ' '55' '    42      0' '42.0000' ' 49820     -1']
     ['68' 'TO BT BT BT SD                                                                                 ' '68' '    34      0' '34.0000' ' 49823     -1']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## azero: B histories not in A 
    []


