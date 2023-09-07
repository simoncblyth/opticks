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

Split that for capture::

    MODE=2 APID=16654 PICK=A ~/j/ntds/ntds3.sh
    MODE=2 BPID=15508 PICK=B ~/j/ntds/ntds3.sh



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




Check 3inch G4CXTest.sh with degenerate default 1e-3
--------------------------------------------------------

~/opticks/u4/tests/FewPMT.sh::

     delta=1e-3   # DEGENERATE DEFAULT IN C++
     #delta=1e-1
     #delta=1
     export Tub3inchPMTV3Manager__VIRTUAL_DELTA_MM=$delta
     


With delta 1e-3 : YUCK::

    ~/opticks/g4cx/tests/G4CXTest.sh 

    a.CHECK : circle_inwards_100 
    b.CHECK : circle_inwards_100 
    QCF qcf :  
    a.q 10000 b.q 10000 lim slice(None, None, None) 
    c2sum :  2034.1321 c2n :     5.0000 c2per:   406.8264  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  2034.13/5:406.826 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT SD   ' ' 0' '  7003   4354' '617.8745' '   310    310']
     [' 1' 'TO BT SA      ' ' 1' '   985   3449' '1369.2594' '     0   5775']
     [' 2' 'TO BT BT SA   ' ' 2' '  1907   2162' '15.9806' '    60      0']
     [' 3' 'TO BT BT AB   ' ' 3' '    48      9' '26.6842' '   431   1226']
     [' 4' 'TO BT BR BT SA' ' 4' '    26     13' ' 4.3333' '   107   1143']
     [' 5' 'TO AB         ' ' 5' '    11      7' ' 0.0000' '   336    615']
     [' 6' 'TO BT BR AB   ' ' 6' '     9      0' ' 0.0000' '  6190     -1']
     [' 7' 'TO BT AB      ' ' 7' '     7      2' ' 0.0000' '  2413   1246']
     [' 8' 'TO SC SA      ' ' 8' '     2      3' ' 0.0000' '  1338   4018']
     [' 9' 'TO SC BT BT SD' ' 9' '     1      0' ' 0.0000' '  1859     -1']
     ['10' 'TO SC AB      ' '10' '     0      1' ' 0.0000' '    -1   5925']
     ['11' 'TO BT SC AB   ' '11' '     1      0' ' 0.0000' '  6977     -1']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## in A but not B 
    []

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## in B but not A 
    []
    PICK=AB MODE=0 SEL=0 POI=-1 ./G4CXAppTest.sh ana 
    not plotting as MODE 0 in environ

With delta 1e-2 : ALSO YUCK::

    QCF qcf :  
    a.q 10000 b.q 10000 lim slice(None, None, None) 
    c2sum :  2065.5898 c2n :     5.0000 c2per:   413.1180  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  2065.59/5:413.118 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT SD   ' ' 0' '  7003   4354' '617.8745' '   310    310']
     [' 1' 'TO BT SA      ' ' 1' '   961   3449' '1403.6608' '     0   5775']
     [' 2' 'TO BT BT SA   ' ' 2' '  1931   2162' '13.0371' '    54      0']
     [' 3' 'TO BT BT AB   ' ' 3' '    48      9' '26.6842' '   431   1226']
     [' 4' 'TO BT BR BT SA' ' 4' '    26     13' ' 4.3333' '   107   1143']
     [' 5' 'TO AB         ' ' 5' '    11      7' ' 0.0000' '   336    615']
     [' 6' 'TO BT BR AB   ' ' 6' '     9      0' ' 0.0000' '  6190     -1']
     [' 7' 'TO BT AB      ' ' 7' '     7      2' ' 0.0000' '  2413   1246']
     [' 8' 'TO SC SA      ' ' 8' '     2      3' ' 0.0000' '  1338   4018']
     [' 9' 'TO SC BT BT SD' ' 9' '     1      0' ' 0.0000' '  1859     -1']
     ['10' 'TO SC AB      ' '10' '     0      1' ' 0.0000' '    -1   5925']
     ['11' 'TO BT SC AB   ' '11' '     1      0' ' 0.0000' '  6977     -1']]


With delta 5e-2 : GOOD AGREEMENT (PropagateEpsilon which is used to set tmin is 0.05)::

    QCF qcf :  
    a.q 10000 b.q 10000 lim slice(None, None, None) 
    c2sum :     4.9946 c2n :     4.0000 c2per:     1.2486  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)   4.99/4:1.249 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT SD   ' ' 0' '  4351   4354' ' 0.0010' '   310    310']
     [' 1' 'TO BT SA      ' ' 1' '  3451   3449' ' 0.0006' '     0   5775']
     [' 2' 'TO BT BT SA   ' ' 2' '  2142   2162' ' 0.0929' '     1      0']
     [' 3' 'TO BT BR BT SA' ' 3' '    27     13' ' 4.9000' '   107   1143']
     [' 4' 'TO BT BT AB   ' ' 4' '    12      9' ' 0.0000' '   431   1226']
     [' 5' 'TO AB         ' ' 5' '    11      7' ' 0.0000' '   336    615']
     [' 6' 'TO SC SA      ' ' 6' '     2      3' ' 0.0000' '  1338   4018']
     [' 7' 'TO BT AB      ' ' 7' '     3      2' ' 0.0000' '  2413   1246']
     [' 8' 'TO SC BT BT SD' ' 8' '     1      0' ' 0.0000' '  1859     -1']
     [' 9' 'TO SC AB      ' ' 9' '     0      1' ' 0.0000' '    -1   5925']]

With delta 1e-1 : GOOD AGREEMENT : ALMOST NO DIFFERENCE FROM 1::

    a.q 10000 b.q 10000 lim slice(None, None, None) 
    c2sum :     4.9763 c2n :     4.0000 c2per:     1.2441  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)   4.98/4:1.244 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT SD   ' ' 0' '  4351   4354' ' 0.0010' '   310    310']
     [' 1' 'TO BT SA      ' ' 1' '  3449   3449' ' 0.0000' '  5775   5775']
     [' 2' 'TO BT BT SA   ' ' 2' '  2144   2162' ' 0.0752' '     0      0']
     [' 3' 'TO BT BR BT SA' ' 3' '    27     13' ' 4.9000' '   107   1143']
     [' 4' 'TO BT BT AB   ' ' 4' '    12      9' ' 0.0000' '   431   1226']
     [' 5' 'TO AB         ' ' 5' '    11      7' ' 0.0000' '   336    615']
     [' 6' 'TO SC SA      ' ' 6' '     2      3' ' 0.0000' '  1338   4018']
     [' 7' 'TO BT AB      ' ' 7' '     3      2' ' 0.0000' '  2413   1246']
     [' 8' 'TO SC BT BT SD' ' 8' '     1      0' ' 0.0000' '  1859     -1']
     [' 9' 'TO SC AB      ' ' 9' '     0      1' ' 0.0000' '    -1   5925']]

With delta 1 : GOOD AGREEMENT::

    QCF qcf :  
    a.q 10000 b.q 10000 lim slice(None, None, None) 
    c2sum :     4.9681 c2n :     4.0000 c2per:     1.2420  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)   4.97/4:1.242 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT SD   ' ' 0' '  4351   4354' ' 0.0010' '   310    310']
     [' 1' 'TO BT SA      ' ' 1' '  3449   3449' ' 0.0000' '  5775   5775']
     [' 2' 'TO BT BT SA   ' ' 2' '  2145   2162' ' 0.0671' '     0      0']
     [' 3' 'TO BT BR BT SA' ' 3' '    27     13' ' 4.9000' '   107   1143']
     [' 4' 'TO BT BT AB   ' ' 4' '    12      9' ' 0.0000' '   431   1226']
     [' 5' 'TO AB         ' ' 5' '    10      7' ' 0.0000' '   336    615']
     [' 6' 'TO SC SA      ' ' 6' '     2      3' ' 0.0000' '  1338   4018']
     [' 7' 'TO BT AB      ' ' 7' '     3      2' ' 0.0000' '  2413   1246']
     [' 8' 'TO SC BT BT SD' ' 8' '     1      0' ' 0.0000' '  1859     -1']
     [' 9' 'TO SC AB      ' ' 9' '     0      1' ' 0.0000' '    -1   5925']]



HMM : IS THAT JUST THE SETTING OF PROPAGATE_EPSILON ? 
--------------------------------------------------------

::

    epsilon:opticks blyth$ opticks-f propagate_epsilon
    ./ana/simtrace_plot.py:        epsilon = self.frame.propagate_epsilon
    ./ana/debug/genstep_sequence_material_mismatch.py:     331     m_context[ "propagate_epsilon"]->setFloat(pe);  // TODO: check impact of changing propagate_epsilon
    ./ana/debug/genstep_sequence_material_mismatch.py:    171         rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );
    ./CSG/csg_intersect_tree.h:    float propagate_epsilon = 0.0001f ;  // ? 
    ./CSG/csg_intersect_tree.h:                    float tminAdvanced = fabsf(csg.data[loopside].w) + propagate_epsilon ;
    ./CSG/csg_intersect_tree.h:                        printf("// %3d : looping one side tminAdvanced %10.4f with eps %10.4f \n", nodeIdx, tminAdvanced, propagate_epsilon );  
    ./CSG/csg_intersect_node.h:    const float propagate_epsilon = 0.0001f ; 
    ./CSG/csg_intersect_node.h:        //float tminAdvanced = enter[i] + propagate_epsilon ;    // <= without ordered enters get internal spurious 
    ./CSG/csg_intersect_node.h:        float tminAdvanced = enter[idx[i]] + propagate_epsilon ; 
    ./CSG/csg_intersect_node.h:        if(tminAdvanced < farthest_exit.w)  // enter[idx[i]]+propagate_epsilon < "farthest_contiguous_exit" so far    
    ./CSG/csg_intersect_node.h:    float propagate_epsilon = 0.0001f ; 
    ./CSG/csg_intersect_node.h:                float tminAdvanced = sub_isect.w + propagate_epsilon ; 
    ./CSG/CSGFoundry.cc:    fr.set_propagate_epsilon( SEventConfig::PropagateEpsilon() ); 
    ./sysrap/sframe.h:    void set_propagate_epsilon(float eps); 
    ./sysrap/sframe.h:    float propagate_epsilon() const ; 
    ./sysrap/sframe.h:       << " propagate_epsilon " << std::setw(10) << std::fixed << std::setprecision(5) << propagate_epsilon()
    ./sysrap/sframe.h:inline void sframe::set_propagate_epsilon(float eps){     aux.q1.f.x = eps ; }
    ./sysrap/sframe.h:inline float sframe::propagate_epsilon() const   { return aux.q1.f.x ; }
    ./sysrap/sframe.py:        propagate_epsilon = a[3,1,0]   # aux.q1.f.x 
    ./sysrap/sframe.py:        self.propagate_epsilon = propagate_epsilon 
    ./dev/csg/slavish.py:propagate_epsilon = 1e-3
    ./dev/csg/slavish.py:            tX_min[side] = isect[THIS][0][W] + propagate_epsilon 
    ./dev/csg/slavish.py:                    tminAdvanced = abs(csg.data[loopside,0,W]) + propagate_epsilon 
    ./dev/csg/slavish.py:                    tX_min[side] = isect[THIS][0][W] + propagate_epsilon 
    ./dev/csg/slavish.py:                        tX_min[side] = isect[THIS][0][W] + propagate_epsilon 
    ./optixrap/cu/csg_intersect_boolean.h:    float tA_min = propagate_epsilon ;  
    ./optixrap/cu/csg_intersect_boolean.h:            x_tmin[side] = isect[side].w + propagate_epsilon ; 
    ./optixrap/cu/csg_intersect_boolean.h:                    float tminAdvanced = fabsf(csg.data[loopside].w) + propagate_epsilon ;
    ./optixrap/cu/csg_intersect_boolean.h:                 tX_min[side] = _side.w + propagate_epsilon ;  // classification as well as intersect needs the advance
    ./optixrap/cu/csg_intersect_boolean.h:                     tX_min[side] = isect[side+LEFT].w + propagate_epsilon ; 
    ./optixrap/cu/csg_intersect_boolean.h:        tX_min[side] = _side.w + propagate_epsilon ;
    ./optixrap/cu/generate.cu:rtDeclareVariable(float,         propagate_epsilon, , );
    ./optixrap/cu/generate.cu:    rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );
    ./optixrap/cu/generate.cu:        rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );
    ./optixrap/cu/intersect_analytic.cu:rtDeclareVariable(float, propagate_epsilon, , );
    ./optixrap/OPropagator.cc:    m_context[ "propagate_epsilon"]->setFloat( m_ok->getEpsilon() );       // TODO: check impact of changing propagate_epsilon
    epsilon:opticks blyth$ 


::

    epsilon:opticks blyth$ opticks-f SEventConfig::PropagateEpsilon
    ./CSGOptiX/CSGOptiX.cc:    params->tmin = SEventConfig::PropagateEpsilon() ;  // eg 0.1 0.05 to avoid self-intersection off boundaries
    ./CSG/CSGFoundry.cc:    fr.set_propagate_epsilon( SEventConfig::PropagateEpsilon() ); 
    ./sysrap/SEventConfig.cc:float SEventConfig::PropagateEpsilon(){ return _PropagateEpsilon ; }
    epsilon:opticks blyth$ 

    epsilon:opticks blyth$ SEventConfigTest | grep PropagateEpsilon
    OPTICKS_PROPAGATE_EPSILON   PropagateEpsilon  :     0.0500
    OPTICKS_PROPAGATE_EPSILON   PropagateEpsilon  :     0.0500
    epsilon:opticks blyth$ 

::

     60 float SEventConfig::_PropagateEpsilonDefault = 0.05f ;





Try to repoduce LPMT apex degen issue standalone
---------------------------------------------------

::

   Change u4/tests/FewPMT.sh geomlist:hmskLogicMaskVirtual
   Change g4cx/tests/G4CXTest.sh check:rain_line_205 

   PICK=B D=2 ~/opticks/g4cx/tests/G4CXTest.sh ana 
   PICK=B D=2 ~/opticks/g4cx/tests/G4CXTest.sh

   ~/opticks/g4cx/tests/G4CXTest.sh    
   ~/opticks/g4cx/tests/G4CXTest.sh grab 

   PICK=AB D=2 ~/opticks/g4cx/tests/G4CXTest.sh ana 






GEOM:FewPMT u4/tests/FewPMT.sh geomlist:hmskLogicMaskVirtual g4cx/tests/G4CXTest.sh check:rain_line_205  
----------------------------------------------------------------------------------------------------------

HMM : DO NOT SEE APEX DEGEN ISSUE STANDALONE WITH 10k, 100k 

::

    PICK=AB D=2 ~/opticks/g4cx/tests/G4CXTest.sh ana


    In [18]: a.f.record[50000-5:50000+5,0:5,0,2]
    Out[18]: 
    array([[ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   ,  141.013],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225]], dtype=float32)

    In [19]: b.f.record[50000-5:50000+5,0:5,0,2]
    Out[19]: 
    array([[ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225],
           [ 205.   ,  200.05 ,  200.   ,  192.   , -175.225]], dtype=float32)


GEOM:V1J011 : Investigate the LPMT apex degeneracy 
----------------------------------------------------

Below PMT-local-frame z-positions for z-points of LPMT apex photons
show geometry delta of 0.05, which happens to 
exactly match the default PropagateEpsilon of 0.05 

jcv HamamatsuMaskManager::

     60     MAGIC_virtual_thickness = 0.05*mm;
     61 }
      


* that could explain why the standalone check not showing the issue currently 

* DONE : ADDED ENVVAR CONTROL FOR THE 0.05 DELTA : export HamamatsuMaskManager__MAGIC_virtual_thickness_MM=0.05 
* TODO : CHECK THAT DECREASING THAT MAKES STANDALONE EXHIBIT THE ISSUE 
* TODO : CHECK THAT INCREASING THAT MAKES INSITU AVOID THE ISSUE 

::

    PICK=AB D=2 ~/j/ntds/ntds3.sh ana 
     
    In [10]: a.lpos[50000-5:50000+5,0:5,2]    
    Out[10]: 
    array([[229.999, 200.049, 192.   , 190.001, 184.999],
           [229.999, 200.048, 192.   , 189.999, 185.   ],
           [229.999, 200.049, 199.998, 191.999, 189.999],
           [229.999, 200.049, 192.   , 189.999, 185.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 192.   , 189.999, 185.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 192.   , 190.   , 185.   ],
           [230.   , 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 192.001, 190.   , 185.   ]])

    In [11]: b.lpos[50000-5:50000+5,0:5,2]
    Out[11]: 
    array([[229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 199.999, 191.999, 190.   ],
           [230.   , 200.049, 199.999, 191.999, 190.   ],
           [229.999, 200.049, 199.999, 192.   , 190.   ]])

    In [12]: a.q[50000-5:50000+5]
    Out[12]: 
    array([[b'TO BT BT BT BT SA                                                                               '],
           [b'TO BT BT BT SA                                                                                  '],
           [b'TO BT BT BT BT BT SA                                                                            '],
           [b'TO BT BT BT BT SA                                                                               '],
           [b'TO BT BT BT BT BT SA                                                                            '],
           [b'TO BT BT BT BT SA                                                                               '],
           [b'TO BT BT BT BT BT SA                                                                            '],
           [b'TO BT BT BT BT SA                                                                               '],
           [b'TO BT BT BT BT SD                                                                               '],
           [b'TO BT BT BT SD                                                                                  ']], dtype='|S96')

    In [13]: b.q[50000-5:50000+5]
    Out[13]: 
    array([[b'TO BT BT BT BT BT SA                                                                            '],
           [b'TO BT BT BT BT BT SA                                                                            '],
           [b'TO BT BT BT BT SD                                                                               '],
           [b'TO BT BT BT BT SD                                                                               '],
           [b'TO BT BT BT BT BT SA                                                                            '],
           [b'TO BT BT BT BT SA                                                                               '],
           [b'TO BT BT BT BT SA                                                                               '],
           [b'TO BT BT BT BT BR BT BT BT BT BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA             '],
           [b'TO BT BT BT BT SA                                                                               '],
           [b'TO BT BT BT BT SA                                                                               ']], dtype='|S96')






Try with decreased MAGIC to 0.01 gives apex issues, increase back to 0.05 back to no issue 
--------------------------------------------------------------------------------------------

::

   ~/opticks/g4cx/tests/G4CXTest.sh 

::


    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## in A but not B 
    [[' 4' 'TO BT BT BT DR BT BT SA                                                                        ' ' 4' '  4302      0' '4302.0000' '  1438     -1']
     [' 9' 'TO BT BT SA                                                                                    ' ' 9' '   943      0' '943.0000' ' 48985     -1']
     ['12' 'TO BT BT DR BT BT BT SA                                                                        ' '12' '   700      0' '700.0000' ' 48986     -1']
     ['13' 'TO BT BT BT DR DR BT BT SA                                                                     ' '13' '   609      0' '609.0000' '  2119     -1']
     ['24' 'TO BT BT DR SA                                                                                 ' '24' '   105      0' '105.0000' ' 49008     -1']
     ['26' 'TO BT BT BT DR DR DR BT BT SA                                                                  ' '26' '    94      0' '94.0000' '  4048     -1']
     ['27' 'TO BT BT DR BT BT SA                                                                           ' '27' '    88      0' '88.0000' ' 48995     -1']
     ['32' 'TO BT BT DR DR BT BT BT SA                                                                     ' '32' '    65      0' '65.0000' ' 48990     -1']
     ['33' 'TO BT BT BT BR DR BT BT SA                                                                     ' '33' '    61      0' '61.0000' '   897     -1']
     ['56' 'TO BT BT DR DR SA                                                                              ' '56' '    17      0' ' 0.0000' ' 48992     -1']
     ['57' 'TO BT BT DR DR BT BT SA                                                                        ' '57' '    16      0' ' 0.0000' ' 49185     -1']
     ['66' 'TO BT BT BT DR DR DR DR BT BT SA                                                               ' '66' '    12      0' ' 0.0000' '  3531     -1']
     ['69' 'TO BT BT DR BT AB                                                                              ' '69' '    11      0' ' 0.0000' ' 49002     -1']
     ['70' 'TO BT BT BT DR BR BT BT SA                                                                     ' '70' '    11      0' ' 0.0000' '  2174     -1']]


Back to 0.05 no problem::

    a.q 100000 b.q 100000 lim slice(None, None, None) 
    c2sum :    59.2115 c2n :    47.0000 c2per:     1.2598  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  59.21/47:1.260 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT BT SA                                                                                 ' ' 0' ' 43249  43434' ' 0.3948' '  1434   1434']
     [' 1' 'TO BT BT BT DR BT BT BT SA                                                                     ' ' 1' ' 34571  34395' ' 0.4491' '  1435   1440']
     [' 2' 'TO BT BT BT DR SA                                                                              ' ' 2' '  6513   6436' ' 0.4579' '  1433   1437']
     [' 3' 'TO BT BT BT DR DR BT BT BT SA                                                                  ' ' 3' '  4745   4628' ' 1.4605' '  1457   2038']
     [' 4' 'TO BT BT BR BT BT SA                                                                           ' ' 4' '  2377   2342' ' 0.2596' '     2      0']
     [' 5' 'TO BT BT BT DR DR SA                                                                           ' ' 5' '  1188   1276' ' 3.1429' '  1449   2201']
     [' 6' 'TO BT BT AB                                                                                    ' ' 6' '  1174   1272' ' 3.9264' '    26     44']
     [' 7' 'TO BT BR BT SA                                                                                 ' ' 7' '   925    956' ' 0.5109' '     1      3']
     [' 8' 'TO BT BT BT DR DR DR BT BT BT SA                                                               ' ' 8' '   828    865' ' 0.8086' '  2041   2249']
     [' 9' 'TO BT BT BT AB                                                                                 ' ' 9' '   797    788' ' 0.0511' '   985   1433']
     ['10' 'TO BT BT BT DR AB                                                                              ' '10' '   436    387' ' 2.9174' '  1899   2353']
     ['11' 'TO BT BT BT DR BT AB                                                                           ' '11' '   426    394' ' 1.2488' '  1834   1847']
     ['12' 'TO BT BT BT DR DR DR SA                                                                        ' '12' '   202    232' ' 2.0737' '  3079   2440']
     ['13' 'TO BT BT BT BT BT BT SA                                                                        ' '13' '   210    202' ' 0.1553' '   778    778']
     ['14' 'TO BT BT BT BR SA                                                                              ' '14' '   155    170' ' 0.6923' '   844    837']
     ['15' 'TO BT BT BT DR BR BT BT BT SA                                                                  ' '15' '   169    168' ' 0.0030' '  1495   1436']
     ['16' 'TO BT BT BT DR DR DR DR BT BT BT SA                                                            ' '16' '   148    165' ' 0.9233' '  3531   2162']
     ['17' 'TO BT BT BT BR DR BT BT BT SA                                                                  ' '17' '   162    134' ' 2.6486' '   830    850']
     ['18' 'TO BT BT BT DR BT BT BT AB                                                                     ' '18' '    82    124' ' 8.5631' '  1605   2569']
     ['19' 'TO BT BT BT DR BT BR BT BT BT BT SA                                                            ' '19' '   109     85' ' 2.9691' '  1464   1473']
     ['20' 'TO BT BT BT SC BT BT BT SA                                                                     ' '20' '   100    106' ' 0.1748' '  2079   1858']
     ['21' 'TO AB                                                                                          ' '21' '    79     80' ' 0.0063' '   336    846']
     ['22' 'TO BT BT BR AB                                                                                 ' '22' '    67     45' ' 4.3214' '     4     15']
     ['23' 'TO BT BT BT DR DR AB                                                                           ' '23' '    55     59' ' 0.1404' '  7999   2756']
     ['24' 'TO BT BT BT DR DR BT AB                                                                        ' '24' '    56     58' ' 0.0351' '  2946   2140']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## in A but not B 
    []

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## in B but not A 
    []






V1J011 : Insitu with larger MAGIC
-----------------------------------

::

     694 ntds_noxjsjfa()
     695 {
     696    #local gpfx=R           # R:Release builds of junosw+custom4   
     697    local gpfx=V          # V:Debug builds of junosw+custom4  
     698    GPFX=${GPFX:-$gpfx}    # need to match with j/ntds/ntds.sh  AGEOM, BGEOM
     699 
     700    export EVTMAX=1
     701 
     702 
     703    ## export U4Tree__DISABLE_OSUR_IMPLICIT=1   
     704    unset U4Tree__DISABLE_OSUR_IMPLICIT
     705    ## WHEN REMOVING AN ENVVAR MUST REMEMBER TO unset
     706    ## disabling OSUR implicit was needed previously to avoid scrambling CSGNode border 
     707    ## but the move to the new workflow should avoid that issue 
     708 
     709    export Tub3inchPMTV3Manager__VIRTUAL_DELTA_MM=1
     710    export HamamatsuMaskManager__MAGIC_virtual_thickness_MM=0.1  # 0.05 C++ default
     711 
     712 
     713    NOXJ=1 NOSJ=1 NOFA=1 GEOM=${GPFX}1J011 OPTICKS_INTEGRATION_MODE=${OPTICKS_INTEGRATION_MODE:-0} ntds
     714 
     715    # this will fail for lack of input photon for OPTICKS_INTEGRATION_MODE=0 
     716 }
     717 



    ntds3_noxjsjfa    # workstation, run opticksMode:3 doing both optical simulations in one invokation

    GEOM # check its V1J011
    GEOM get 
    GEOM tmpget 

    PICK=AB D=2 ~/j/ntds/ntds3.sh ana 



Zooming in on simtrace : the clearance is 0.05 mm
----------------------------------------------------

::

    MODE=2 ~/opticks/u4/tests/U4SimtraceTest.sh
    MODE=2 ~/opticks/u4/tests/U4SimtraceTest.sh ana
    MODE=2 VERBOSE=1 ~/opticks/u4/tests/U4SimtraceTest.sh     
  
Verified the envvar control works with the simtrace plot, clearance increased to 1 mm
via setting in u4/tests/FewPMT.sh::

    108 #magic=0.05    # initial default in original C++
    109 #magic=0.01    # decrease to try to get LPMT apex degeneracy issue to appear standalone 
    110 magic=1        # CHECK ITS WORKING IN simtrace plot 
    111 
    112 export HamamatsuMaskManager__MAGIC_virtual_thickness_MM=$magic



    HamamatsuMaskManager::HamamatsuMaskManager
    MAGIC_virtual_thickness            1.00000
    MAGIC_virtual_thickness_default    0.05000
    MAGIC CHANGED BY ENVVAR : HamamatsuMaskManager__MAGIC_virtual_thickness_MM
     


FIXED FOR HAMA : After making sure the control is present+compiled+configured for Insitu
------------------------------------------------------------------------------------------

::

    ntds3_noxjsjfa    # workstation, run opticksMode:3 doing both optical simulations in one invokation

    GEOM get          # laptop
    GEOM tmpget       # laptop
    PICK=AB D=2 ~/j/ntds/ntds3.sh ana  # laptop compare 

Looking good when shooting HAMA PMTs::

    QCF qcf :  
    a.q 100000 b.q 100000 lim slice(None, None, None) 
    c2sum :    75.2304 c2n :   107.0000 c2per:     0.7031  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  75.23/107:0.703 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT BT BT SA                                                                              ' ' 0' ' 37500  37733' ' 0.7216' '     8      2']
     [' 1' 'TO BT BT BT BT SD                                                                              ' ' 1' ' 30892  30651' ' 0.9437' '     4      1']
     [' 2' 'TO BT BT BT BT BT SA                                                                           ' ' 2' ' 12429  12446' ' 0.0116' '  9412   9599']
     [' 3' 'TO BT BT BT BT BT SR SA                                                                        ' ' 3' '  3810   3811' ' 0.0001' ' 11059  10961']
     [' 4' 'TO BT BT BT BT BT SR SR SA                                                                     ' ' 4' '  1999   2016' ' 0.0720' ' 10899  10881']
     [' 5' 'TO BT BT AB                                                                                    ' ' 5' '   884    915' ' 0.5342' '    26     16']
     [' 6' 'TO BT BT BT BT BT SR SR SR SA                                                                  ' ' 6' '   572    581' ' 0.0703' ' 14725  14767']
     [' 7' 'TO BT BT BT BT BR BT BT BT BT BT BT AB                                                         ' ' 7' '   411    437' ' 0.7972' ' 11875   5112']
     [' 8' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                                   ' ' 8' '   335    337' ' 0.0060' '  7444   7474']
     [' 9' 'TO BT BT BT BT BR BT BT BT BT BT SA                                                            ' ' 9' '   332    309' ' 0.8253' '  1021   1023']
     ['10' 'TO BT BT BT BT AB                                                                              ' '10' '   319    321' ' 0.0063' '   651     34']
     ['11' 'TO BT BT BT BT BT SR BR SA                                                                     ' '11' '   309    319' ' 0.1592' ' 33584  33568']
     ['12' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                                   ' '12' '   317    282' ' 2.0451' '  8147   8160']
     ['13' 'TO BT BT BT BT BR BT BT BT BT BT BT SD                                                         ' '13' '   314    314' ' 0.0000' '  5262   5254']
     ['14' 'TO BT BT BT BT BR BT BT BT BT AB                                                               ' '14' '   281    255' ' 1.2612' '   646    265']
     ['15' 'TO BT BT BT BT BT SR SR SR BR SA                                                               ' '15' '   212    245' ' 2.3829' ' 14749  14802']
     ['16' 'TO BT BT BR BT BT BT SA                                                                        ' '16' '   243    218' ' 1.3557' '     2      0']
     ['17' 'TO BT BT BT BR BT BT BT BT SA                                                                  ' '17' '   216    211' ' 0.0585' '   206    195']
     ['18' 'TO BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT SA                                             ' '18' '   176    204' ' 2.0632' ' 15529  15472']
     ['19' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT BT BT BT BT SD                                       ' '19' '   190    172' ' 0.8950' ' 16931  11829']
     ['20' 'TO BT BT BT BT BR BT BT BT BT BT BT BT BT BT BT BT BT SA                                       ' '20' '   164    174' ' 0.2959' ' 11832  11864']
     ['21' 'TO BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT BT BT SA                                    ' '21' '   148    170' ' 1.5220' ' 17266  17964']
     ['22' 'TO BT BT BT BT BT SR SR SR BR BR SR SA                                                         ' '22' '   168    133' ' 4.0698' ' 15414  15611']
     ['23' 'TO BT BT BT BT BT BR SR SA                                                                     ' '23' '   148    161' ' 0.5469' '  9351   9270']
     ['24' 'TO BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT BT BT SD                                    ' '24' '   149    157' ' 0.2092' ' 16930  17039']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## in A but not B 
    []

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## in B but not A 
    []



TODO: Try reducing clearance to see how low it can go
--------------------------------------------------------


TODO: Test with NNVT
----------------------


TODO: Find some sigma_alpha/polish surfaces to hunt for deviations
--------------------------------------------------------------------





