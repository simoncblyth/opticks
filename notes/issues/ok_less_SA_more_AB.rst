ok_less_SA_more_AB : may be due to degenerates
==================================================

Summary
----------

* OK lacks SA cf G4 "SI BT BT BT SA" 
* OK has more "SI BT BT AB" bulk absorb in the water
* BUT: shooting photons in water is in good G4/OK agreement (when avoid PMTs by shooting the Tyvek) 

Below presents evidence that the degenerate Pyrex///Pyrex and Water///Pyrex boundaries (+0.001 mm)
are the (or at least one) source of this discrepancy. 

Hmm how to proceed.

1. avoid the Pyrex/Pyrex degenerate +0.001mm by fixing the source geometry problem
2. runtime workaround by trying to notice the inconsistency in the boundaries and make some assumption about which is correct
3. notice the near degeneracy in the GPU geometry translation and skip the problematic and pointless Pyrex/Pyrex boundary 


* 1: is the "correct" way, but more disruptive and difficult as need to change source geometry 

  * the degenerate has already forced doing microstep skipping the the G4 emulation, but that 
    is just a bookkeeping issue 
  * overall the pointless boundary seems not to cause a problem for Geant4 and it would be difficult to remove
    from JUNO offline because it simplifies the surfaces 

* 2: is probably quicker to do in the short term, but liable to bringing its own problems
* 3: **BEST APPROACH** because it avoids complicating kernels and is the most honest way, simplifying 
     the GPU geometry to avoid the degeneracy and its problems
       

After Skipping Degenerate Pyrex///Pyrex
-------------------------------------------

* excess OK AB seems fixed
* looks like 3BT to 4BT mismatch, CRecorder G4 microStep skipping perhaps not always working would explain 3 vs 4 mismatch 

  * this issue was there previously, just now it becomes the most prominent one  


::

    In [3]: ab.his[:40]
    Out[3]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11142     11142       310.47/62 =  5.01  (pval:1.000 prob:0.000)  
       n             iseq         a         b    a-b               c2          a/b                   b/a           [ns] label
    0000               42      1666      1621     45             0.62        1.028 +- 0.025        0.973 +- 0.024  [2 ] SI AB
    0001            7ccc2      1367      1258    109             4.53        1.087 +- 0.029        0.920 +- 0.026  [5 ] SI BT BT BT SD       ## OK EXCESS SI-3BT-SD
    0002            8ccc2       624       766   -142            14.51        0.815 +- 0.033        1.228 +- 0.044  [5 ] SI BT BT BT SA       ## OK LACKS SI-3BT-SA 
    0003           7ccc62       680       543    137            15.35        1.252 +- 0.048        0.799 +- 0.034  [6 ] SI SC BT BT BT SD    ## OK EXCESS SI-SC-3BT-SD    
    0004             8cc2       570       496     74             5.14        1.149 +- 0.048        0.870 +- 0.039  [4 ] SI BT BT SA
    0005              452       408       495    -87             8.38        0.824 +- 0.041        1.213 +- 0.055  [3 ] SI RE AB
    0006           7ccc52       405       385     20             0.51        1.052 +- 0.052        0.951 +- 0.048  [6 ] SI RE BT BT BT SD
    0007              462       399       351     48             3.07        1.137 +- 0.057        0.880 +- 0.047  [3 ] SI SC AB
    0008           8ccc62       270       258     12             0.27        1.047 +- 0.064        0.956 +- 0.059  [6 ] SI SC BT BT BT SA
    0009          7ccc662       255       195     60             8.00        1.308 +- 0.082        0.765 +- 0.055  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       195       180     15             0.60        1.083 +- 0.078        0.923 +- 0.069  [5 ] SI SC BT BT SA
    0011           8ccc52       176       191    -15             0.61        0.921 +- 0.069        1.085 +- 0.079  [6 ] SI RE BT BT BT SA
    0012          7ccc652       186       165     21             1.26        1.127 +- 0.083        0.887 +- 0.069  [7 ] SI RE SC BT BT BT SD
    0013               41       156       160     -4             0.05        0.975 +- 0.078        1.026 +- 0.081  [2 ] CK AB
    0014             4552       118       152    -34             4.28        0.776 +- 0.071        1.288 +- 0.104  [4 ] SI RE RE AB
    0015            8cc52       136       133      3             0.03        1.023 +- 0.088        0.978 +- 0.085  [5 ] SI RE BT BT SA
    0016          7ccc552       123       120      3             0.04        1.025 +- 0.092        0.976 +- 0.089  [7 ] SI RE RE BT BT BT SD
    0017             4662       125       114     11             0.51        1.096 +- 0.098        0.912 +- 0.085  [4 ] SI SC SC AB
    0018             4652       118       108     10             0.44        1.093 +- 0.101        0.915 +- 0.088  [4 ] SI RE SC AB
    0019             4cc2       121       104     17             1.28        1.163 +- 0.106        0.860 +- 0.084  [4 ] SI BT BT AB                  ## NOW CONSISTENT
    0020           7cccc2        50       151   -101            50.75        0.331 +- 0.047        3.020 +- 0.246  [6 ] SI BT BT BT BT SD            ## OK LACKS SI-4BT-SD 
    0021          8ccc662        78        99    -21             2.49        0.788 +- 0.089        1.269 +- 0.128  [7 ] SI SC SC BT BT BT SA
    0022         7ccc6662        76        82     -6             0.23        0.927 +- 0.106        1.079 +- 0.119  [8 ] SI SC SC SC BT BT BT SD
    0023          8ccc652        72        72      0             0.00        1.000 +- 0.118        1.000 +- 0.118  [7 ] SI RE SC BT BT BT SA
    0024          8ccc552        62        69     -7             0.37        0.899 +- 0.114        1.113 +- 0.134  [7 ] SI RE RE BT BT BT SA
    0025           8cc662        57        65     -8             0.52        0.877 +- 0.116        1.140 +- 0.141  [6 ] SI SC SC BT BT SA
    0026         7ccc6652        57        63     -6             0.30        0.905 +- 0.120        1.105 +- 0.139  [8 ] SI RE SC SC BT BT BT SD
    0027             4562        51        57     -6             0.33        0.895 +- 0.125        1.118 +- 0.148  [4 ] SI SC RE AB
    0028         7ccc6552        50        57     -7             0.46        0.877 +- 0.124        1.140 +- 0.151  [8 ] SI RE RE SC BT BT BT SD
    0029          7ccc562        68        38     30             8.49        1.789 +- 0.217        0.559 +- 0.091  [7 ] SI SC RE BT BT BT SD
    0030              4c2        58        46     12             1.38        1.261 +- 0.166        0.793 +- 0.117  [3 ] SI BT AB
    0031           8cc652        48        54     -6             0.35        0.889 +- 0.128        1.125 +- 0.153  [6 ] SI RE SC BT BT SA
    0032          7cccc62        19        77    -58            35.04        0.247 +- 0.057        4.053 +- 0.462  [7 ] SI SC BT BT BT BT SD       ## OK LACKS SI-SC-4BT-SD
    0033           8cccc2        28        67    -39            16.01        0.418 +- 0.079        2.393 +- 0.292  [6 ] SI BT BT BT BT SA          ## OK LACKS SI-4BT-SA
    0034            4cc62        50        40     10             1.11        1.250 +- 0.177        0.800 +- 0.126  [5 ] SI SC BT BT AB
    0035            46662        40        43     -3             0.11        0.930 +- 0.147        1.075 +- 0.164  [5 ] SI SC SC SC AB
    0036           8cc552        30        50    -20             5.00        0.600 +- 0.110        1.667 +- 0.236  [6 ] SI RE RE BT BT SA
    0037            4ccc2        56        17     39            20.84        3.294 +- 0.440        0.304 +- 0.074  [5 ] SI BT BT BT AB
    0038         7ccc5552        33        40     -7             0.67        0.825 +- 0.144        1.212 +- 0.192  [8 ] SI RE RE RE BT BT BT SD
    .                              11142     11142       310.47/62 =  5.01  (pval:1.000 prob:0.000)  

    In [4]: 




Are there any other same material boundaries ?
-------------------------------------------------

::

    epsilon:ana blyth$ ipython blib.py
     nbnd  36 nmat  17 nsur  23 
      0 :   1 : **Galactic///Galactic** 
      1 :   2 : Galactic///Rock 
      2 :   3 : Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pTopRock/Air 
      3 :   4 : **Air///Air** 
      4 :   5 : Air///LS 
      5 :   6 : Air///Steel 
      6 :   7 : Air///Tyvek 
      7 :   8 : Air///Aluminium 
      8 :   9 : Aluminium///Adhesive 
      9 :  10 : Adhesive///TiO2Coating 
     10 :  11 : TiO2Coating///Scintillator 
     11 :  12 : Rock///Tyvek 
     12 :  13 : Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pPoolLining/vetoWater 
     13 :  14 : vetoWater///LatticedShellSteel 
     14 :  15 : vetoWater/CDTyvekSurface//Tyvek 
     15 :  16 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water 
     16 :  17 : Water///Acrylic 
     17 :  18 : Acrylic///LS 
     18 :  19 : LS///Acrylic 
     19 :  20 : LS///PE_PA 
     20 :  21 : Water///Steel 
     21 :  22 : Water///PE_PA 
     22 :  23 : Water///Pyrex 
     23 :  24 : **Pyrex///Pyrex** 
     24 :  25 : Pyrex/NNVTMCPPMT_photocathode_logsurf2/NNVTMCPPMT_photocathode_logsurf1/Vacuum 
     25 :  26 : Pyrex//NNVTMCPPMT_mirror_logsurf1/Vacuum 
     26 :  27 : Pyrex/HamamatsuR12860_photocathode_logsurf2/HamamatsuR12860_photocathode_logsurf1/Vacuum 
     27 :  28 : Pyrex//HamamatsuR12860_mirror_logsurf1/Vacuum 
     28 :  29 : **Water///Water** 
     29 :  30 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
     30 :  31 : Pyrex//PMT_3inch_absorb_logsurf1/Vacuum 
     31 :  32 : Water///LS 
     32 :  33 : Water/Steel_surface/Steel_surface/Steel 
     33 :  34 : vetoWater///Water 
     34 :  35 : Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum 
     35 :  36 : Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum 
    epsilon:ana blyth$ 


How to find the geometry with those borders ?
--------------------------------------------------

Trace backwards. boundary comes from instanceIdentity.z::

     32 RT_PROGRAM void closest_hit_propagate()
     33 {
     34      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     35      float cos_theta = dot(n,ray.direction);
     36 
     37      prd.distance_to_boundary = t ;   // standard semantic attrib for this not available in raygen, so must pass it
     38 
     39      unsigned boundaryIndex = ( instanceIdentity.z & 0xffff ) ;
     40      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     41      prd.identity = instanceIdentity ;
     42      prd.surface_normal = cos_theta > 0.f ? -n : n ;
     43 }


::

    245 glm::uvec4 GVolume::getIdentity() const
    246 {
    247     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
    248     return id ;
    249 }
    250 
    251 /**
    252 GVolumne::getShapeIdentity
    253 ----------------------------
    254 
    255 The shape identity packs mesh index and boundary index together.
    256 This info is used GPU side by::
    257 
    258    oxrap/cu/material1_propagate.cu:closest_hit_propagate
    259 
    260 ::
    261 
    262     id = np.load("all_volume_identity.npy")
    263 
    264     bidx = ( id[:,2] >>  0)  & 0xffff ) 
    265     midx = ( id[:,2] >> 16)  & 0xffff ) 
    266 
    267 
    268 **/
    269 
    270 unsigned GVolume::getShapeIdentity() const
    271 {
    272     return OpticksShape::Encode( getMeshIndex(), getBoundary() );
    273 }



Occurrence of each boundary::

    #!/usr/bin/env python

    import numpy as np
    from opticks.ana.key import keydir
    from opticks.ana.blib import BLib
    KEYDIR=keydir()
    blib = BLib()

    if __name__ == '__main__':
        avi = np.load(os.path.join(KEYDIR, "GNodeLib/all_volume_identity.npy"))

        bidx = ( avi[:,2] >>  0)  & 0xffff 
        midx = ( avi[:,2] >> 16)  & 0xffff 

        b,n = np.unique( bidx, return_counts=True)

        for i in range(len(b)): 
            print("%3d : %7d : %s " % (b[i],n[i],blib.bname(b[i])))
        pass


::
        
    epsilon:ana blyth$ ipython bidx.py 
      0 :       1 : Galactic///Galactic 
      1 :       2 : Galactic///Rock 
      2 :       1 : Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pTopRock/Air 
      3 :     191 : Air///Air 
      4 :       1 : Air///LS 
      5 :       1 : Air///Steel 
      6 :       1 : Air///Tyvek 
      7 :     504 : Air///Aluminium 
      8 :     504 : Aluminium///Adhesive 
      9 :   32256 : Adhesive///TiO2Coating 
     10 :   32256 : TiO2Coating///Scintillator 
                                                 top-tracker related
     11 :       1 : Rock///Tyvek 
     12 :       1 : Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pPoolLining/vetoWater 
     13 :    2120 : vetoWater///LatticedShellSteel 
     14 :       1 : vetoWater/CDTyvekSurface//Tyvek 
     15 :       1 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water 
     16 :    3048 : Water///Acrylic 
     17 :       1 : Acrylic///LS 

     18 :      46 : LS///Acrylic 
     19 :       8 : LS///PE_PA 
                      small number of items inside LS?

     20 :   27960 : Water///Steel 
     21 :      56 : Water///PE_PA 

     22 :   45612 : Water///Pyrex 

            12612+5000+25600+2400 = 45612    all PMTs including veto 

     23 :   20012 : Pyrex///Pyrex 

            12612+5000+2400 = 20012         all PMTs excluding veto


     24 :   12612 : Pyrex/NNVTMCPPMT_photocathode_logsurf2/NNVTMCPPMT_photocathode_logsurf1/Vacuum 
     25 :   12612 : Pyrex//NNVTMCPPMT_mirror_logsurf1/Vacuum 

     26 :    5000 : Pyrex/HamamatsuR12860_photocathode_logsurf2/HamamatsuR12860_photocathode_logsurf1/Vacuum 
     27 :    5000 : Pyrex//HamamatsuR12860_mirror_logsurf1/Vacuum 

     28 :   25601 : Water///Water 
                               3inch envelope + 1  ?

     29 :   25600 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
     30 :   25600 : Pyrex//PMT_3inch_absorb_logsurf1/Vacuum 

     31 :       1 : Water///LS 
     32 :       1 : Water/Steel_surface/Steel_surface/Steel 

     33 :    2400 : vetoWater///Water 
     34 :    2400 : Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum 
     35 :    2400 : Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum 


    epsilon:ana blyth$ 


How to skip the 20012 degenerates ? All named *body_solid*
--------------------------------------------------------------


ipython -i ggeo.py::

    In [8]: nn = np.where( gg.bidx == 23 )[0]

    In [10]: gg.midx
    Out[10]: array([126,  12,  11, ..., 120, 118, 119], dtype=uint32)

    In [11]: gg.midx[np.where(gg.bidx == 23)]
    Out[11]: array([100, 104, 100, ..., 120, 120, 120], dtype=uint32)

    In [12]: mm = gg.midx[np.where(gg.bidx == 23)]

    In [13]: mm.shape
    Out[13]: (20012,)

    In [14]: np.unique(mm, return_counts=True)
    Out[14]: (array([100, 104, 120], dtype=uint32), array([12612,  5000,  2400]))

    In [17]: gg.msn[100]
    Out[17]: u'NNVTMCPPMT_body_solid0x3a905c0'

    In [18]: gg.msn[104]
    Out[18]: u'HamamatsuR12860_body_solid_1_90x3a992f0'

    In [19]: gg.msn[120]
    Out[19]: u'PMT_20inch_veto_body_solid_1_20x3a8b930'


::

    In [38]: w = np.where(np.char.find( a, "body_solid") != -1)

    In [39]: a[w]
    Out[39]: 
    array([u'NNVTMCPPMT_body_solid0x3a905c0', u'HamamatsuR12860_body_solid_1_90x3a992f0', u'PMT_3inch_body_solid_ell_ell_helper0x421dec0', u'PMT_20inch_veto_body_solid_1_20x3a8b930',
           u'HamamatsuR12860_body_solid_1_90x3a992f0'], dtype='<U44')



::

    --skipsolidname NNVTMCPPMT_body_solid,HamamatsuR12860_body_solid_1_9,PMT_20inch_veto_body_solid_1_2

::

    tds-skipsolidname(){ echo $(tds-skipsolidname-) | tr " " "," ; }
    tds-skipsolidname-(){ cat << EON | grep -v ^#
    #NNVTMCPPMTsMask_virtual
    #HamamatsuR12860sMask_virtual
    mask_PMT_20inch_vetosMask_virtual
    NNVTMCPPMT_body_solid
    HamamatsuR12860_body_solid_1_9
    PMT_20inch_veto_body_solid_1_2
    EON
    }




How difficult to remove the Pyrex +0.001mm : epidermis ? VERY 
----------------------------------------------------------------

* that layer has one benefit of making the PMT border surface implementation self contained
* without it will need to feed in the containing water PV to form the border surface 


The external PV (the water) will need to take the role of body_phys in the borders::

    662 HamamatsuR12860PMTManager::helper_make_optical_surface()
    663 {   
    664     new G4LogicalBorderSurface(GetName()+"_photocathode_logsurf1",
    665             inner1_phys, body_phys,
    666             Photocathode_opsurf);
    667     new G4LogicalBorderSurface(GetName()+"_photocathode_logsurf2",
    668             body_phys, inner1_phys,
    669             Photocathode_opsurf);
    670     new G4LogicalBorderSurface(GetName()+"_mirror_logsurf1",
    671             inner2_phys, body_phys,
    672             m_mirror_opsurf);
    673     new G4LogicalBorderSurface(GetName()+"_mirror_logsurf2",
    674             body_phys, inner2_phys,
    675             m_mirror_opsurf);
    676 }



Looking again after the DsG4Scintillation update
---------------------------------------------------

::

    In [4]: ab.his[:30]
    Out[4]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11142     11142       616.50/66 =  9.34  (pval:1.000 prob:0.000)  
       n             iseq         a         b    a-b               c2          a/b                   b/a           [ns] label
    0000               42      1666      1621     45             0.62        1.028 +- 0.025        0.973 +- 0.024  [2 ] SI AB
    0001            7ccc2      1264      1258      6             0.01        1.005 +- 0.028        0.995 +- 0.028  [5 ] SI BT BT BT SD
    0002            8ccc2       581       766   -185            25.41        0.758 +- 0.031        1.318 +- 0.048  [5 ] SI BT BT BT SA
                                              ^^^^^^^^^^ OK lacks surface abs            
    0003           7ccc62       579       543     36             1.16        1.066 +- 0.044        0.938 +- 0.040  [6 ] SI SC BT BT BT SD
    0004             8cc2       570       496     74             5.14        1.149 +- 0.048        0.870 +- 0.039  [4 ] SI BT BT SA
    0005              452       408       495    -87             8.38        0.824 +- 0.041        1.213 +- 0.055  [3 ] SI RE AB
    0006              462       399       351     48             3.07        1.137 +- 0.057        0.880 +- 0.047  [3 ] SI SC AB
    0007           7ccc52       360       385    -25             0.84        0.935 +- 0.049        1.069 +- 0.055  [6 ] SI RE BT BT BT SD
    0008           8ccc62       239       258    -19             0.73        0.926 +- 0.060        1.079 +- 0.067  [6 ] SI SC BT BT BT SA
    0009          7ccc662       218       195     23             1.28        1.118 +- 0.076        0.894 +- 0.064  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       195       180     15             0.60        1.083 +- 0.078        0.923 +- 0.069  [5 ] SI SC BT BT SA
    0011             4cc2       268       104    164            72.30        2.577 +- 0.157        0.388 +- 0.038  [4 ] SI BT BT AB
                                               ^^^^^^ OK excess bulk AB
    0012           8ccc52       160       191    -31             2.74        0.838 +- 0.066        1.194 +- 0.086  [6 ] SI RE BT BT BT SA
    0013          7ccc652       163       165     -2             0.01        0.988 +- 0.077        1.012 +- 0.079  [7 ] SI RE SC BT BT BT SD
    0014               41       156       160     -4             0.05        0.975 +- 0.078        1.026 +- 0.081  [2 ] CK AB
    0015             4552       118       152    -34             4.28        0.776 +- 0.071        1.288 +- 0.104  [4 ] SI RE RE AB
    0016            8cc52       136       133      3             0.03        1.023 +- 0.088        0.978 +- 0.085  [5 ] SI RE BT BT SA
    0017             4662       125       114     11             0.51        1.096 +- 0.098        0.912 +- 0.085  [4 ] SI SC SC AB
    0018            4cc62       189        40    149            96.95        4.725 +- 0.344        0.212 +- 0.033  [5 ] SI SC BT BT AB
                                                ^^^^^ OK excess bulk AB
    0019          7ccc552       109       120    -11             0.53        0.908 +- 0.087        1.101 +- 0.100  [7 ] SI RE RE BT BT BT SD
    0020             4652       118       108     10             0.44        1.093 +- 0.101        0.915 +- 0.088  [4 ] SI RE SC AB
    0021           7cccc2        50       151   -101            50.75        0.331 +- 0.047        3.020 +- 0.246  [6 ] SI BT BT BT BT SD
                                                ^^^^^^^ OK lacks SI-3BT-SD 
    0022          8ccc662        63        99    -36             8.00        0.636 +- 0.080        1.571 +- 0.158  [7 ] SI SC SC BT BT BT SA
    0023         7ccc6662        60        82    -22             3.41        0.732 +- 0.094        1.367 +- 0.151  [8 ] SI SC SC SC BT BT BT SD
    0024          8ccc652        65        72     -7             0.36        0.903 +- 0.112        1.108 +- 0.131  [7 ] SI RE SC BT BT BT SA
    0025           8cc662        57        65     -8             0.52        0.877 +- 0.116        1.140 +- 0.141  [6 ] SI SC SC BT BT SA
    0026            4cc52        94        28     66            35.70        3.357 +- 0.346        0.298 +- 0.056  [5 ] SI RE BT BT AB
    0027          8ccc552        53        69    -16             2.10        0.768 +- 0.106        1.302 +- 0.157  [7 ] SI RE RE BT BT BT SA
    0028             4562        51        57     -6             0.33        0.895 +- 0.125        1.118 +- 0.148  [4 ] SI SC RE AB
    .                              11142     11142       616.50/66 =  9.34  (pval:1.000 prob:0.000)  


SA : SURFACE_ABORB compare the sims
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


generate.cu::


    832         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    833         {
    834             command = propagate_at_surface(p, s, rng);
    835             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    836             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    837         }
    838         else
    839         {
    840             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    841             // tacit CONTINUE
    842         }


propagate.h::

    679 __device__ int
    680 propagate_at_surface(Photon &p, State &s, curandState &rng)
    681 {
    682     float u_surface = curand_uniform(&rng);
    683 #ifdef WITH_ALIGN_DEV
    684     float u_surface_burn = curand_uniform(&rng);
    685 #endif
    686 
    687 #ifdef WITH_ALIGN_DEV_DEBUG
    688     rtPrintf("propagate_at_surface   u_OpBoundary_DiDiAbsorbDetectReflect:%.9g \n", u_surface);
    689     rtPrintf("propagate_at_surface   u_OpBoundary_DoAbsorption:%.9g \n", u_surface_burn);
    690 #endif
    691 
    692     if( u_surface < s.surface.y )   // absorb   
    693     {
    694         s.flag = SURFACE_ABSORB ;
    695         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    696         return BREAK ;
    697     }
    698     else if ( u_surface < s.surface.y + s.surface.x )  // absorb + detect
    699     {
    700         s.flag = SURFACE_DETECT ;
    701         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    702         return BREAK ;
    703     }
    704     else if (u_surface  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    705     {
    706         s.flag = SURFACE_DREFLECT ;
    707         propagate_at_diffuse_reflector_geant4_style(p, s, rng);
    708         return CONTINUE;
    709     }
    710     else
    711     {
    712         s.flag = SURFACE_SREFLECT ;
    713         //propagate_at_specular_reflector(p, s, rng );
    714         propagate_at_specular_reflector_geant4_style(p, s, rng );
    715         return CONTINUE;
    716     }
    717 }

::

     32 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     33 {   
     34     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     35     // >0 outward going photon
     36     // <0 inward going photon
     37     //
     38     // NB the line is above the details of the payload (ie how many float4 per matsur) 
     39     //    it is just 
     40     //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
     41     //
     42     
     43     int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ;
     44     
     45     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     46     //  
     47     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     48     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;
     49     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;
     50     
     51     //  consider photons arriving at PMT cathode surface
     52     //  geometry normals are expected to be out of the PMT 
     53     //
     54     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     55     
     56     s.material1 = boundary_lookup( wavelength, m1_line, 0);
     57     s.m1group2  = boundary_lookup( wavelength, m1_line, 1);
     58     
     59     s.material2 = boundary_lookup( wavelength, m2_line, 0);
     60     s.surface   = boundary_lookup( wavelength, su_line, 0);
     61     
     62     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     63     
     64     s.index.x = optical_buffer[m1_line].x ; // m1 index
     65     s.index.y = optical_buffer[m2_line].x ; // m2 index 
     66     s.index.z = optical_buffer[su_line].x ; // su index
     67     s.index.w = identity.w   ;
     68     
     69     s.identity = identity ;
     70 
     71 }




ana/surface.py SA is coming from 1-SD onto the logsurf
-----------------------------------------------------------

* hmm the issue of very close surface degenerates might have an impact if they 
  result in getting a boundary without the surface

  * check the boundary histories, and make it easier to do so 

* also note lots of wavelength dependence

::

    In [1]: run surface.py
    INFO:opticks.ana.main:envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'g4live', 'pfx': 'OKTest', 'cat': 'g4live'} 
    INFO:opticks.ana.key:ppos 4
          wl      sd      sa      sr      dr UpperChimneyTyvekSurface
    [[[300.    0.    0.9   0.    0.1]
      [400.    0.    0.9   0.    0.1]
      [500.    0.    0.9   0.    0.1]
      [600.    0.    0.9   0.    0.1]]]
          wl      sd      sa      sr      dr NNVTMCPPMT_photocathode_logsurf1
    [[[300.       0.041    0.959    0.       0.    ]
      [400.       0.8409   0.1591   0.       0.    ]
      [500.       0.5155   0.4845   0.       0.    ]
      [600.       0.1171   0.8829   0.       0.    ]]]
          wl      sd      sa      sr      dr NNVTMCPPMT_mirror_logsurf1
    [[[300.       0.       0.0001   0.9999   0.    ]
      [400.       0.       0.0001   0.9999   0.    ]
      [500.       0.       0.0001   0.9999   0.    ]
      [600.       0.       0.0001   0.9999   0.    ]]]
          wl      sd      sa      sr      dr NNVTMCPPMT_photocathode_logsurf2
    [[[300.       0.041    0.959    0.       0.    ]
      [400.       0.8409   0.1591   0.       0.    ]
      [500.       0.5155   0.4845   0.       0.    ]
      [600.       0.1171   0.8829   0.       0.    ]]]
          wl      sd      sa      sr      dr HamamatsuR12860_photocathode_logsurf1
    [[[300.       0.0401   0.9599   0.       0.    ]
      [400.       0.8376   0.1624   0.       0.    ]
      [500.       0.4741   0.5259   0.       0.    ]
      [600.       0.0612   0.9388   0.       0.    ]]]
          wl      sd      sa      sr      dr HamamatsuR12860_mirror_logsurf1
    [[[300.       0.       0.0001   0.9999   0.    ]
      [400.       0.       0.0001   0.9999   0.    ]
      [500.       0.       0.0001   0.9999   0.    ]
      [600.       0.       0.0001   0.9999   0.    ]]]
          wl      sd      sa      sr      dr HamamatsuR12860_photocathode_logsurf2
    [[[300.       0.0401   0.9599   0.       0.    ]
      [400.       0.8376   0.1624   0.       0.    ]
      [500.       0.4741   0.5259   0.       0.    ]
      [600.       0.0612   0.9388   0.       0.    ]]]
          wl      sd      sa      sr      dr PMT_3inch_photocathode_logsurf1
    [[[300.       0.046    0.954    0.       0.    ]
      [400.       0.7655   0.2345   0.       0.    ]
      [500.       0.6437   0.3563   0.       0.    ]
      [600.       0.1751   0.8249   0.       0.    ]]]
          wl      sd      sa      sr      dr PMT_3inch_absorb_logsurf1
    [[[300.   0.   1.   0.   0.]
      [400.   0.   1.   0.   0.]
      [500.   0.   1.   0.   0.]
      [600.   0.   1.   0.   0.]]]
          wl      sd      sa      sr      dr PMT_3inch_photocathode_logsurf2
    [[[300.       0.046    0.954    0.       0.    ]
      [400.       0.7655   0.2345   0.       0.    ]
      [500.       0.6437   0.3563   0.       0.    ]
      [600.       0.1751   0.8249   0.       0.    ]]]
          wl      sd      sa      sr      dr PMT_3inch_absorb_logsurf3
    [[[300.   0.   1.   0.   0.]
      [400.   0.   1.   0.   0.]
      [500.   0.   1.   0.   0.]
      [600.   0.   1.   0.   0.]]]
          wl      sd      sa      sr      dr PMT_20inch_veto_photocathode_logsurf1
    [[[300.       0.0212   0.9788   0.       0.    ]
      [400.       0.8034   0.1966   0.       0.    ]
      [500.       0.5149   0.4851   0.       0.    ]
      [600.       0.1292   0.8708   0.       0.    ]]]
          wl      sd      sa      sr      dr PMT_20inch_veto_mirror_logsurf1
    [[[300.       0.       0.0001   0.9999   0.    ]
      [400.       0.       0.0001   0.9999   0.    ]
      [500.       0.       0.0001   0.9999   0.    ]
      [600.       0.       0.0001   0.9999   0.    ]]]
          wl      sd      sa      sr      dr PMT_20inch_veto_photocathode_logsurf2
    [[[300.       0.0212   0.9788   0.       0.    ]
      [400.       0.8034   0.1966   0.       0.    ]
      [500.       0.5149   0.4851   0.       0.    ]
      [600.       0.1292   0.8708   0.       0.    ]]]
          wl      sd      sa      sr      dr CDTyvekSurface
    [[[300.       0.       0.2693   0.       0.7307]
      [400.       0.       0.08     0.       0.92  ]
      [500.       0.       0.09     0.       0.91  ]
      [600.       0.       0.09     0.       0.91  ]]]
          wl      sd      sa      sr      dr Steel_surface
    [[[300.    0.    0.6   0.    0.4]
      [400.    0.    0.6   0.    0.4]
      [500.    0.    0.6   0.    0.4]
      [600.    0.    0.6   0.    0.4]]]
          wl      sd      sa      sr      dr Implicit_RINDEX_NoRINDEX_pExpHall_pTopRock
    [[[300.   0.   1.   0.   0.]
      [400.   0.   1.   0.   0.]
      [500.   0.   1.   0.   0.]
      [600.   0.   1.   0.   0.]]]
          wl      sd      sa      sr      dr Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pPoolLining
    [[[300.   0.   1.   0.   0.]
      [400.   0.   1.   0.   0.]
      [500.   0.   1.   0.   0.]
      [600.   0.   1.   0.   0.]]]
          wl      sd      sa      sr      dr Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector
    [[[300.   0.   1.   0.   0.]
      [400.   0.   1.   0.   0.]
      [500.   0.   1.   0.   0.]
      [600.   0.   1.   0.   0.]]]
          wl      sd      sa      sr      dr perfectDetectSurface
    [[[300.   1.   0.   0.   0.]
      [400.   1.   0.   0.   0.]
      [500.   1.   0.   0.   0.]
      [600.   1.   0.   0.   0.]]]
          wl      sd      sa      sr      dr perfectAbsorbSurface
    [[[300.   0.   1.   0.   0.]
      [400.   0.   1.   0.   0.]
      [500.   0.   1.   0.   0.]
      [600.   0.   1.   0.   0.]]]
          wl      sd      sa      sr      dr perfectSpecularSurface
    [[[300.   0.   0.   1.   0.]
      [400.   0.   0.   1.   0.]
      [500.   0.   0.   1.   0.]
      [600.   0.   0.   1.   0.]]]
          wl      sd      sa      sr      dr perfectDiffuseSurface
    [[[300.   0.   0.   0.   1.]
      [400.   0.   0.   0.   1.]
      [500.   0.   0.   0.   1.]
      [600.   0.   0.   0.   1.]]]

    In [2]: 



Checking boundary histories
-----------------------------

::

    In [6]: a.bn.view(np.int8)
    Out[6]: 
    A([[[ 18,  17, -23, ...,   0,   0,   0]],

       [[ 18,  18,   0, ...,   0,   0,   0]],

       [[ 18,  17, -24, ...,   0,   0,   0]],

       ...,

       [[ 18,  17, -23, ...,   0,   0,   0]],

       [[ 18,  18,  18, ...,   0,   0,   0]],

       [[ 18,  18,  17, ...,   0,   0,   0]]], dtype=int8)

    In [7]: a.bn.view(np.int8).shape
    Out[7]: (11142, 1, 16)


::

    In [9]: als[10:11]
    Out[9]: SI BT BT SA

    In [10]: print(a.blib.format(a.bn[10]))
     18 : Acrylic///LS
     17 : Water///Acrylic
     16 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water

    In [11]: a.bn[10]
    Out[11]: A([18, 17, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=int8)


::      

               Ty/Wa            /   /   
                /            Wa/Ac /  
               /              /   /  
              /              / Ac/LS
             /              /   /
            .  . . . . . . / . /. . . SI
           /              /   /
          /              /   /
         /              /   /
        /              /   /
       /              /   /
      /              /   /
     /              /   /



ABSLENGTH Check
~~~~~~~~~~~~~~~~~

* looking in GMaterialLib has no surprises, need to dump at point of use


GMaterialLib::

      63 const char* GMaterialLib::keyspec =
      64 "refractive_index:RINDEX,"
      65 "absorption_length:ABSLENGTH,"
      66 "scattering_length:RAYLEIGH,"
      67 "reemission_prob:REEMISSIONPROB,"
      68 "group_velocity:GROUPVEL,"
      69 "extra_y:EXTRA_Y,"
      70 "extra_z:EXTRA_Z,"
      71 "extra_w:EXTRA_W,"
      72 "detect:EFFICIENCY,"
      73 ;


From the GMaterialLib on epsilon with an old geocache::

    In [11]: run material.py
    [{__init__            :proplib.py:151} INFO     - names : None 
    [{__init__            :proplib.py:161} INFO     - npath : /usr/local/opticks/geocache/OKX4Test_lWorld0x32a96e0_PV_g4live/g4ok_gltf/a3cbac8189a032341f76682cdb4f47b6/1/GItemList/GMaterialLib.txt 
    [{__init__            :proplib.py:168} INFO     - names : ['LS', 'Steel', 'Tyvek', 'Air', 'Scintillator', 'TiO2Coating', 'Adhesive', 'Aluminium', 'Rock', 'LatticedShellSteel', 'Acrylic', 'PE_PA', 'Vacuum', 'Pyrex', 'Water', 'vetoWater', 'Galactic'] 
    [{opticks_args        :main.py   :140} INFO     - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'g4live', 'pfx': 'OKTest', 'cat': 'g4live'} 
    [{<module>            :material.py:195} INFO     - mat Water 
            wavelen      rindex      abslen     scatlen    reemprob    groupvel LS
    [[[   300.          1.5264      0.975    4887.5513      0.7214    177.2066]
      [   400.          1.5       195.5178  17976.7012      0.8004    189.7664]
      [   500.          1.4902 114196.2188  43987.5156      0.1231    195.3692]
      [   600.          1.4837  46056.8906 116999.7344      0.0483    198.683 ]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Steel
    [[[    300.           1.           0.001  1000000.           0.         299.7924]
      [    400.           1.           0.001  1000000.           0.         299.7924]
      [    500.           1.           0.001  1000000.           0.         299.7924]
      [    600.           1.           0.001  1000000.           0.         299.7924]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Tyvek
    [[[    300.           1.       10000.     1000000.           0.         299.7924]
      [    400.           1.       10000.     1000000.           0.         299.7924]
      [    500.           1.       10000.     1000000.           0.         299.7924]
      [    600.           1.       10000.     1000000.           0.         299.7924]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Air
    [[[     300.            1.0003 10000000.      1000000.            0.          299.7115]
      [     400.            1.0003 10000000.      1000000.            0.          299.7115]
      [     500.            1.0003 10000000.      1000000.            0.          299.7115]
      [     600.            1.0003 10000000.      1000000.            0.          299.7115]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Scintillator
    [[[    300.           1.     1000000.     1000000.           0.         299.7924]
      [    400.           1.     1000000.     1000000.           0.         299.7924]
      [    500.           1.     1000000.     1000000.           0.         299.7924]
      [    600.           1.     1000000.     1000000.           0.         299.7924]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel TiO2Coating
    [[[    300.           1.     1000000.     1000000.           0.         299.7924]
      [    400.           1.     1000000.     1000000.           0.         299.7924]
      [    500.           1.     1000000.     1000000.           0.         299.7924]
      [    600.           1.     1000000.     1000000.           0.         299.7924]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Adhesive
    [[[    300.           1.     1000000.     1000000.           0.         299.7924]
      [    400.           1.     1000000.     1000000.           0.         299.7924]
      [    500.           1.     1000000.     1000000.           0.         299.7924]
      [    600.           1.     1000000.     1000000.           0.         299.7924]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Aluminium
    [[[    300.           1.     1000000.     1000000.           0.         299.7924]
      [    400.           1.     1000000.     1000000.           0.         299.7924]
      [    500.           1.     1000000.     1000000.           0.         299.7924]
      [    600.           1.     1000000.     1000000.           0.         299.7924]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Rock
    [[[    300.           1.           0.001  1000000.           0.         299.7924]
      [    400.           1.           0.001  1000000.           0.         299.7924]
      [    500.           1.           0.001  1000000.           0.         299.7924]
      [    600.           1.           0.001  1000000.           0.         299.7924]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel LatticedShellSteel
    [[[    300.           1.           0.001  1000000.           0.         299.7924]
      [    400.           1.           0.001  1000000.           0.         299.7924]
      [    500.           1.           0.001  1000000.           0.         299.7924]
      [    600.           1.           0.001  1000000.           0.         299.7924]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Acrylic
    [[[    300.           1.5358      29.0775 1000000.           0.         175.9265]
      [    400.           1.5078     822.0058 1000000.           0.         187.7579]
      [    500.           1.4977    8908.     1000000.           0.         195.7688]
      [    600.           1.4922    8908.     1000000.           0.         198.2241]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel PE_PA
    [[[    300.           1.51         4.9401 1000000.           0.         198.538 ]
      [    400.           1.51         3.9277 1000000.           0.         198.538 ]
      [    500.           1.51         9.3682 1000000.           0.         198.538 ]
      [    600.           1.51        13.8064 1000000.           0.         198.538 ]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Vacuum
    [[[3.0000e+02 1.0000e+00 1.0000e+09 1.0000e+06 0.0000e+00 2.9979e+02]
      [4.0000e+02 1.0000e+00 1.0000e+09 1.0000e+06 0.0000e+00 2.9979e+02]
      [5.0000e+02 1.0000e+00 1.0000e+09 1.0000e+06 0.0000e+00 2.9979e+02]
      [6.0000e+02 1.0000e+00 1.0000e+09 1.0000e+06 0.0000e+00 2.9979e+02]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Pyrex
    [[[    300.           1.5061    1000.     1000000.           0.         195.0881]
      [    400.           1.4865    1341.0769 1000000.           0.         193.9326]
      [    500.           1.478     1999.3562 1000000.           0.         198.9286]
      [    600.           1.4734     996.954  1000000.           0.         200.8115]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Water
    [[[    300.           1.3608    9039.2441 1000000.           0.         212.4812]
      [    400.           1.355    29940.1895 1000000.           0.         218.0326]
      [    500.           1.3492   39363.5898 1000000.           0.         217.1819]
      [    600.           1.344     6529.043  1000000.           0.         218.093 ]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel vetoWater
    [[[    300.           1.3608    9039.2441 1000000.           0.         212.4812]
      [    400.           1.355    29940.1895 1000000.           0.         218.0326]
      [    500.           1.3492   39363.5898 1000000.           0.         217.1819]
      [    600.           1.344     6529.043  1000000.           0.         218.093 ]]]
            wavelen      rindex      abslen     scatlen    reemprob    groupvel Galactic
    [[[    300.           1.     1000000.     1000000.           0.         299.7924]
      [    400.           1.     1000000.     1000000.           0.         299.7924]
      [    500.           1.     1000000.     1000000.           0.         299.7924]
      [    600.           1.     1000000.     1000000.           0.         299.7924]]]




G4OpAbsorption::GetMeanFreePath 
---------------------------------

g4-cls G4OpAbsorption::

    138     if ( aMaterialPropertyTable ) {
    139        AttenuationLengthVector = aMaterialPropertyTable->
    140                                                 GetProperty(kABSLENGTH);
    141            if ( AttenuationLengthVector ){
    142              AttenuationLength = AttenuationLengthVector->
    143                                          Value(thePhotonMomentum);
    144            }
    145            else {
    146 //             G4cout << "No Absorption length specified" << G4endl;
    147            }
    148         }
    149         else {
    150 //           G4cout << "No Absorption length specified" << G4endl;
    151         }
    152 
    153         return AttenuationLength;
    154 }


Observe missed Water///Pyrex border, hitting instead Pyrex///Pyrex 
-----------------------------------------------------------------------

::

    In [2]: ab.his
    Out[2]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11142     11142       616.50/66 =  9.34  (pval:1.000 prob:0.000)  
       n             iseq         a         b    a-b               c2          a/b                   b/a           [ns] label
    0000               42      1666      1621     45             0.62        1.028 +- 0.025        0.973 +- 0.024  [2 ] SI AB
    0001            7ccc2      1264      1258      6             0.01        1.005 +- 0.028        0.995 +- 0.028  [5 ] SI BT BT BT SD
    0002            8ccc2       581       766   -185            25.41        0.758 +- 0.031        1.318 +- 0.048  [5 ] SI BT BT BT SA
    0003           7ccc62       579       543     36             1.16        1.066 +- 0.044        0.938 +- 0.040  [6 ] SI SC BT BT BT SD
    0004             8cc2       570       496     74             5.14        1.149 +- 0.048        0.870 +- 0.039  [4 ] SI BT BT SA
    0005              452       408       495    -87             8.38        0.824 +- 0.041        1.213 +- 0.055  [3 ] SI RE AB
    0006              462       399       351     48             3.07        1.137 +- 0.057        0.880 +- 0.047  [3 ] SI SC AB
    0007           7ccc52       360       385    -25             0.84        0.935 +- 0.049        1.069 +- 0.055  [6 ] SI RE BT BT BT SD
    0008           8ccc62       239       258    -19             0.73        0.926 +- 0.060        1.079 +- 0.067  [6 ] SI SC BT BT BT SA
    0009          7ccc662       218       195     23             1.28        1.118 +- 0.076        0.894 +- 0.064  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       195       180     15             0.60        1.083 +- 0.078        0.923 +- 0.069  [5 ] SI SC BT BT SA
    0011             4cc2       268       104    164            72.30        2.577 +- 0.157        0.388 +- 0.038  [4 ] SI BT BT AB
    0012           8ccc52       160       191    -31             2.74        0.838 +- 0.066        1.194 +- 0.086  [6 ] SI RE BT BT BT SA
    0013          7ccc652       163       165     -2             0.01        0.988 +- 0.077        1.012 +- 0.079  [7 ] SI RE SC BT BT BT SD
    0014               41       156       160     -4             0.05        0.975 +- 0.078        1.026 +- 0.081  [2 ] CK AB
    0015             4552       118       152    -34             4.28        0.776 +- 0.071        1.288 +- 0.104  [4 ] SI RE RE AB
    0016            8cc52       136       133      3             0.03        1.023 +- 0.088        0.978 +- 0.085  [5 ] SI RE BT BT SA
    0017             4662       125       114     11             0.51        1.096 +- 0.098        0.912 +- 0.085  [4 ] SI SC SC AB
    0018            4cc62       189        40    149            96.95        4.725 +- 0.344        0.212 +- 0.033  [5 ] SI SC BT BT AB
    .                              11142     11142       616.50/66 =  9.34  (pval:1.000 prob:0.000)  



    In [3]: a.sel = "SI BT BT BT SA"      ## select the OK "SA"


    In [15]: a.bn.reshape(-1,4).view(np.int8)[:20]
    Out[15]: 
    A([[ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -27,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -27,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -27,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -27,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -27,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -27,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23, -27,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=int8)


    In [17]: print(a.blib.format(a.bn.reshape(-1,4).view(np.int8)[0]))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -23 : Water///Pyrex
    -25 : Pyrex/NNVTMCPPMT_photocathode_logsurf2/NNVTMCPPMT_photocathode_logsurf1/Vacuum

    In [18]: print(a.blib.format(a.bn.reshape(-1,4).view(np.int8)[1]))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -23 : Water///Pyrex
    -27 : Pyrex/HamamatsuR12860_photocathode_logsurf2/HamamatsuR12860_photocathode_logsurf1/Vacuum


    In [19]: print(a.blib.format(a.bn.reshape(-1,4).view(np.int8)[19]))
     18 : Acrylic///LS          # from center of the LS shoot ray,  find Ac///LS  +ve boundary means are in imat:LS 
     17 : Water///Acrylic       # at the Acrylic shoot another ray, find Wa///Ac  +ve boundary means are in imat:Ac 
    -24 : Pyrex///Pyrex         
    -25 : Pyrex/NNVTMCPPMT_photocathode_logsurf2/NNVTMCPPMT_photocathode_logsurf1/Vacuum


The boundary sequence going from Water///Acrylic to Pyrex///Pyrex is clear sign of missing a boundary, 
at first glance it might seem like missing the Water///Pyrex was not a problem 
BUT that surely means are using the ABSLENGTH (and other properties) of Pyrex and not Water for 
part of the propagation.

::

    In [29]: a.bn.reshape(-1,4).view(np.int8).shape
    Out[29]: (581, 16)

    In [31]: np.where( a.bn.reshape(-1,4).view(np.int8)[:,2] == -24 )[0]
    Out[31]: array([ 19,  27,  47,  48,  74,  80,  83, 111, 116, 130, 141, 145, 148, 152, 160, 176, 177, 180, 185, 189, 190, 229, 256, 316, 346, 405, 411, 418, 452, 469, 480, 506, 539])

    In [32]: np.where( a.bn.reshape(-1,4).view(np.int8)[:,2] == -24 )[0].shape
    Out[32]: (33,)

    In [33]: np.where( a.bn.reshape(-1,4).view(np.int8)[:,2] == -23 )[0].shape
    Out[33]: (548,)

    In [34]: 33./581.
    Out[34]: 0.05679862306368331

To automate this need to get the imat/omat indices.
Hmm, what about seqmat ?  Does that show this ?

::

    In [41]: a.seqmat_ana.table
    Out[41]: 
    seqmat_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                                581         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000            defb1        0.943         548        [5 ] LS Ac Wa Py Va
    0001            deeb1        0.057          33        [5 ] LS Ac Py Py Va
       n             iseq         frac           a    a-b      [ns] label
    .                                581         1.00 


Pyrex ABSLEN is much shorter than water... this might explain the excess AB in the "Water" 
because in 5% of cases it is being mis-identified as Pyrex.::


    451             wavelen      rindex      abslen     scatlen    reemprob    groupvel Pyrex
    452     [[[    300.           1.5061    1000.     1000000.           0.         195.0881]
    453       [    400.           1.4865    1341.0769 1000000.           0.         193.9326]
    454       [    500.           1.478     1999.3562 1000000.           0.         198.9286]
    455       [    600.           1.4734     996.954  1000000.           0.         200.8115]]]
    456             wavelen      rindex      abslen     scatlen    reemprob    groupvel Water
    457     [[[    300.           1.3608    9039.2441 1000000.           0.         212.4812]
    458       [    400.           1.355    29940.1895 1000000.           0.         218.0326]
    459       [    500.           1.3492   39363.5898 1000000.           0.         217.1819]
    460       [    600.           1.344     6529.043  1000000.           0.         218.093 ]]]
    461             wavelen      rindex      abslen     scatlen    reemprob    groupvel vetoWater



Hmm how to proceed.

1. avoid the Pyrex/Pyrex degenerate +0.001mm by fixing this geometry problem
2. workaround by trying to notice the inconsistency in the boundaries and make some assumption about which is correct

* 1 is the "correct" way, but more disruptive and difficult as need to change geometry 



Look at the excess AB in "Water"
----------------------------------


::

    In [42]: a.sel = "SI BT BT AB"
    In [44]: a.bn.shape
    Out[44]: (268, 1, 4)

    In [45]: a.seqmat_ana.table
    Out[45]: 
    seqmat_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                                268         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000             eeb1        0.590         158        [4 ] LS Ac Py Py
    0001             ffb1        0.410         110        [4 ] LS Ac Wa Wa
       n             iseq         frac           a    a-b      [ns] label
    .                                268         1.00 




    In [49]: a.bn.view(np.int8).reshape(-1,16)[:50]
    Out[49]: 
    A([[ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],




    In [52]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[0]))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -24 : Pyrex///Pyrex

    In [53]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[1]))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -24 : Pyrex///Pyrex

    In [54]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[2]))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -24 : Pyrex///Pyrex

    In [55]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[3]))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -24 : Pyrex///Pyrex

    In [56]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[4]))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -23 : Water///Pyrex

    In [57]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[5]))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -24 : Pyrex///Pyrex




