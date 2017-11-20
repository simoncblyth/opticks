tboolean-zsphere1-zsphere2-discrep
=====================================

::

    tboolean-;tboolean-zsphere1 --okg4 
    tboolean-;tboolean-zsphere2 --okg4 

    




Looks like reflection difference with the symmetrical z1:z2 -200:200::

    simon:opticksnpy blyth$ tboolean-;tboolean-zsphere1--

    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main(csgpath="/tmp/blyth/opticks/tboolean-zsphere1--")
    CSG.kwa = dict(poly="IM", resolution="40", verbosity="0", ctrl="0" )

    container = CSG("box", param=[0,0,0,1000], boundary="Rock//perfectAbsorbSurface/Vacuum", poly="MC", nx="20" )

    zsphere = CSG("zsphere", param=[0,0,0,500], param1=[-200,200,0,0],param2=[0,0,0,0],  boundary="Vacuum///GlassSchottF2" )

    CSG.Serialize([container, zsphere], args.csgpath )


    [2017-11-20 21:00:37,547] p90143 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-zsphere1)  None 0 
    A tboolean-zsphere1/torch/  1 :  20171120-2059 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/1/fdom.npy () 
    B tboolean-zsphere1/torch/ -1 :  20171120-2059 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-zsphere1--
    .                seqhis_ana  1:tboolean-zsphere1   -1:tboolean-zsphere1        c2        ab        ba 
    .                             100000    100000      3442.65/12 = 286.89  (pval:0.000 prob:1.000)  
    0000             8ccd     88627     82520           217.91        1.074 +- 0.004        0.931 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      5685      5776             0.72        0.984 +- 0.013        1.016 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5162      8007           614.63        0.645 +- 0.009        1.551 +- 0.017  [5 ] TO BT BR BT SA
    0003           8cbbcd       301      2193          1435.31        0.137 +- 0.008        7.286 +- 0.156  [6 ] TO BT BR BR BT SA
    0004            86ccd        61        69             0.49        0.884 +- 0.113        1.131 +- 0.136  [5 ] TO BT BT SC SA
    0005              86d        33        35             0.06        0.943 +- 0.164        1.061 +- 0.179  [3 ] TO SC SA
    0006              4cd        32        18             3.92        1.778 +- 0.314        0.562 +- 0.133  [3 ] TO BT AB
    0007            8c6cd        17         8             0.00        2.125 +- 0.515        0.471 +- 0.166  [5 ] TO BT SC BT SA
    0008          8cbbbcd        12       938           902.61        0.013 +- 0.004       78.167 +- 2.552  [7 ] TO BT BR BR BR BT SA
    0009          8cc6ccd        10         7             0.00        1.429 +- 0.452        0.700 +- 0.265  [7 ] TO BT BT SC BT BT SA
    0010          8cbb6cd         5         3             0.00        1.667 +- 0.745        0.600 +- 0.346  [7 ] TO BT SC BR BR BT SA
    0011           8cb6cd         5         9             0.00        0.556 +- 0.248        1.800 +- 0.600  [6 ] TO BT SC BR BT SA
    0012             4ccd         5         9             0.00        0.556 +- 0.248        1.800 +- 0.600  [4 ] TO BT BT AB
    0013           86cbcd         4        10             0.00        0.400 +- 0.200        2.500 +- 0.791  [6 ] TO BT BR BT SC SA
    0014            8cc6d         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [5 ] TO SC BT BT SA
    0015           8b6ccd         4         1             0.00        4.000 +- 2.000        0.250 +- 0.250  [6 ] TO BT BT SC BR SA
    0016               4d         4         2             0.00        2.000 +- 1.000        0.500 +- 0.354  [2 ] TO AB
    0017           8c6bcd         3         1             0.00        3.000 +- 1.732        0.333 +- 0.333  [6 ] TO BT BR SC BT SA
    0018         8cbbb6cd         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [8 ] TO BT SC BR BR BR BT SA
    0019       8cbbbbb6cd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT SC BR BR BR BR BR BT SA
    .                             100000    100000      3442.65/12 = 286.89  (pval:0.000 prob:1.000)  




But with offset z1:z2 100:200 get agreement::

    simon:opticksnpy blyth$ tboolean-;tboolean-zsphere2--

    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main(csgpath="/tmp/blyth/opticks/tboolean-zsphere2--")
    CSG.kwa = dict(poly="IM", resolution="40", verbosity="0", ctrl="0" )

    container = CSG("box", param=[0,0,0,1000], boundary="Rock//perfectAbsorbSurface/Vacuum", poly="MC", nx="20" )

    zsphere = CSG("zsphere", param=[0,0,0,500], param1=[100,200,0,0],param2=[0,0,0,0],  boundary="Vacuum///GlassSchottF2" )

    CSG.Serialize([container, zsphere], args.csgpath )

    simon:opticksnpy blyth$ 
    simon:opticksnpy blyth$ 

    [2017-11-20 21:02:58,439] p90174 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-zsphere2)  None 0 
    A tboolean-zsphere2/torch/  1 :  20171120-2100 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere2/torch/1/fdom.npy () 
    B tboolean-zsphere2/torch/ -1 :  20171120-2100 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere2/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-zsphere2--
    .                seqhis_ana  1:tboolean-zsphere2   -1:tboolean-zsphere2        c2        ab        ba 
    .                             100000    100000         6.70/6 =  1.12  (pval:0.349 prob:0.651)  
    0000             8ccd     88645     88772             0.09        0.999 +- 0.003        1.001 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      5685      5709             0.05        0.996 +- 0.013        1.004 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5168      5008             2.52        1.032 +- 0.014        0.969 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       301       301             0.00        1.000 +- 0.058        1.000 +- 0.058  [6 ] TO BT BR BR BT SA
    0004            86ccd        86        69             1.86        1.246 +- 0.134        0.802 +- 0.097  [5 ] TO BT BT SC SA
    0005              86d        33        27             0.60        1.222 +- 0.213        0.818 +- 0.157  [3 ] TO SC SA
    0006          8cc6ccd        14         7             0.00        2.000 +- 0.535        0.500 +- 0.189  [7 ] TO BT BT SC BT BT SA
    0007          8cbbbcd        12        19             1.58        0.632 +- 0.182        1.583 +- 0.363  [7 ] TO BT BR BR BR BT SA
    0008             4ccd         8        15             0.00        0.533 +- 0.189        1.875 +- 0.484  [4 ] TO BT BT AB
    0009              4cd         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [3 ] TO BT AB
    0010            8cc6d         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [5 ] TO SC BT BT SA
    0011           86cbcd         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [6 ] TO BT BR BT SC SA
    0012           8b6ccd         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [6 ] TO BT BT SC BR SA
    0013       bbbbbc6ccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT SC BT BR BR BR BR BR
    0014               4d         4         8             0.00        0.500 +- 0.250        2.000 +- 0.707  [2 ] TO AB




