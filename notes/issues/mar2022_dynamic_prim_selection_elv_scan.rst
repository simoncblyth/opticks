mar2022_dynamic_prim_selection_elv_scan
==========================================


::

    epsilon:CSGOptiX blyth$ SNAP_LIMIT=32 ./grabsnap.sh --reverse --selectspec not_elv_t --candle t103   ## slowest first

    [2022-03-28 15:36:58,403] p21670 {/Users/blyth/opticks/ana/snap.py:268} INFO -  all_snaps 279 selectspec not_elv_t snaps 137 SNAP_LIMIT 32 lim_snaps 32 
    idx         -e        time(s)           relative         enabled geometry description                                              
      0        103         0.0225             3.0917         ONLY: solidXJfixture                                                      
      1        112         0.0051             0.7009         ONLY: NNVTMCPPMTTail                                                      
      2        105         0.0027             0.3644         ONLY: HamamatsuR12860Tail                                                 
      3        111         0.0025             0.3422         ONLY: NNVTMCPPMTsMask                                                     
      4        117         0.0022             0.3051         ONLY: NNVTMCPPMTsMask_virtual                                             
      5          0         0.0018             0.2496         ONLY: sTopRock_domeAir                                                    
      6          1         0.0018             0.2474         ONLY: sTopRock_dome                                                       
      7         94         0.0018             0.2406         ONLY: sTarget                                                             
      8        123         0.0017             0.2392         ONLY: sChimneyAcrylic                                                     
      9        127         0.0017             0.2378         ONLY: sInnerWater                                                         
     10        128         0.0017             0.2366         ONLY: sReflectorInCD                                                      
     11         95         0.0017             0.2363         ONLY: sAcrylic                                                            
     12         14         0.0016             0.2160         ONLY: sAirTT                                                              
     13         15         0.0016             0.2142         ONLY: sExpHall                                                            
     14        109         0.0015             0.2025         ONLY: HamamatsuR12860_PMT_20inch_pmt_solid_1_4                            
     15        104         0.0013             0.1847         ONLY: HamamatsuR12860sMask                                                
     16        110         0.0012             0.1668         ONLY: HamamatsuR12860sMask_virtual                                        
     17        125         0.0012             0.1641         ONLY: sChimneySteel                                                       
     18        107         0.0011             0.1464         ONLY: HamamatsuR12860_PMT_20inch_inner2_solid_1_4                         
     19          5         0.0011             0.1455         ONLY: Upper_Steel_tube                                                    
     20          6         0.0010             0.1418         ONLY: Upper_Tyvek_tube                                                    
     21        130         0.0009             0.1256         ONLY: PMT_20inch_veto_inner1_solid                                        
     22        138         0.0009             0.1171         ONLY: sWorld                                                              
     23          2         0.0009             0.1171         ONLY: sDomeRockBox                                                        
     24        129         0.0009             0.1168         ONLY: mask_PMT_20inch_vetosMask                                           
     25         17         0.0008             0.1166         ONLY: sTopRock                                                            
     26        137         0.0008             0.1166         ONLY: sBottomRock                                                         
     27        136         0.0008             0.1162         ONLY: sPoolLining                                                         
     28        116         0.0008             0.1160         ONLY: NNVTMCPPMT_PMT_20inch_pmt_solid_head                                
     29        122         0.0008             0.1151         ONLY: PMT_3inch_pmt_solid                                                 
     30         16         0.0008             0.1082         ONLY: sExpRockBox                                                         
     31        135         0.0008             0.1064         ONLY: sOuterWaterPool                                                     
    idx         -e        time(s)           relative         enabled geometry description                                              


    epsilon:CSGOptiX blyth$ ./grabsnap.sh --selectspec only_elv_t --candle t103    ## fastest first 

    [2022-03-28 15:33:31,235] p21298 {/Users/blyth/opticks/ana/snap.py:268} INFO -  all_snaps 279 selectspec only_elv_t snaps 142 SNAP_LIMIT 256 lim_snaps 142 
    idx         -e        time(s)           relative         enabled geometry description                                              
      0       t103         0.0073             1.0000         EXCL: solidXJfixture                                                      
      1         t1         0.0118             1.6169         EXCL: sTopRock_dome                                                       
      2         t0         0.0119             1.6306         EXCL: sTopRock_domeAir                                                    
      3       t112         0.0119             1.6311         EXCL: NNVTMCPPMTTail                                                      
      4          t         0.0119             1.6317         ALL                                                                       
      5       t127         0.0119             1.6393         EXCL: sInnerWater                                                         
      6        t97         0.0120             1.6488         EXCL: sStrut                                                              
      7        t29         0.0121             1.6558         EXCL: GLw2.equ_bt01_FlangeI_Web_FlangeII                                  
      8       t136         0.0121             1.6563         EXCL: sPoolLining                                                         
      9        t95         0.0121             1.6576         EXCL: sAcrylic                                                            
     10       t128         0.0121             1.6576         EXCL: sReflectorInCD                                                      
     11       t105         0.0121             1.6657         EXCL: HamamatsuR12860Tail                                                 
     12        t60         0.0121             1.6670         EXCL: GLb3.bt09_FlangeI_Web_FlangeII                                      
     13       t135         0.0121             1.6673         EXCL: sOuterWaterPool                                                     
     14       t104         0.0122             1.6684         EXCL: HamamatsuR12860sMask                                                
     15       t137         0.0122             1.6687         EXCL: sBottomRock                                                         
     16       t131         0.0122             1.6729         EXCL: PMT_20inch_veto_inner2_solid                                        
     17        t94         0.0122             1.6732         EXCL: sTarget                                                             
     18        t62         0.0122             1.6751         EXCL: GLb3.bt11_FlangeI_Web_FlangeII                                      
     19       t109         0.0122             1.6753         EXCL: HamamatsuR12860_PMT_20inch_pmt_solid_1_4                            
     20        t37         0.0122             1.6754         EXCL: GLw1.bt08_bt09_FlangeI_Web_FlangeII                                 
     21        t65         0.0122             1.6761         EXCL: GZ1.A03_04_FlangeI_Web_FlangeII                                     
     22       t111         0.0122             1.6768         EXCL: NNVTMCPPMTsMask                                                     
     23       t114         0.0122             1.6771         EXCL: NNVTMCPPMT_PMT_20inch_inner2_solid_head                             
     24        t49         0.0122             1.6779         EXCL: GLb1.up02_FlangeI_Web_FlangeII                                      
     25        t72         0.0122             1.6782         EXCL: GZ1.B04_05_FlangeI_Web_FlangeII                                     
     26       t100         0.0122             1.6784         EXCL: base_steel                                                          
     27        t26         0.0122             1.6794         EXCL: GLw1.up02_up03_FlangeI_Web_FlangeII                                 
     28        t20         0.0122             1.6795         EXCL: GLw1.up08_up09_FlangeI_Web_FlangeII                                 
     29        t59         0.0122             1.6795         EXCL: GLb1.bt08_FlangeI_Web_FlangeII                            





