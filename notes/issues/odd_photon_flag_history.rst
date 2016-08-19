Odd Photon Flag History ?
===========================

So much SC no RE looks very wrong::

    simon:ana blyth$ ./histype.py --det dayabay --tag 1 --src torch 
    ./histype.py --det dayabay --tag 1 --src torch
    [2016-08-19 16:56:02,581] p24445 {./histype.py:55} INFO - loaded ph from /tmp/blyth/opticks/evt/dayabay/torch/1/ph.npy shape (100000, 1, 2) 
         14076 TO SC SC SC SC SC SC SC SC SC 
          8432 TO AB 
          7202 TO SC AB 
          6965 TO SA 
          6512 TO SC SC SA 
          6345 TO SC SC SC SA 
          6118 TO SC SA 
          5899 TO SC SC AB 
          5623 TO SC SC SC SC SA 
          5055 TO SC SC SC SC SC SA 
          4887 TO SC SC SC AB 
          4257 TO SC SC SC SC SC SC SA 
          3787 TO SC SC SC SC AB 
          3376 TO SC SC SC SC SC SC SC SA 
          2978 TO SC SC SC SC SC AB 
          2803 TO SC SC SC SC SC SC SC SC SA 
          2411 TO SC SC SC SC SC SC AB 
          1847 TO SC SC SC SC SC SC SC AB 
          1380 TO SC SC SC SC SC SC SC SC AB 
             4 TO SC BT BT SC SC SC SC SC SC 
             3 TO SC SC SC BT BT SC SC SC SC 
             2 TO SC SC BT BT SC SC SC SC SC 
             2 TO SC SC BT BT SA 
             2 TO SC SC SC SC SC BT BT SC SC 
             2 TO SC SC SC BT BT SA 
             2 TO SC SC SC SC SC BT BT BT BT 
             2 TO SC SC SC BT BT SC SA 
             1 TO SC BT BT SC SC SC SA 
             1 TO SC SC SC BT BT BT BT SC SC 
             1 TO SC SC SC SC SC SC SC SC BT 
             1 TO SC SC BT BT BT BR BR BR BR 
             1 TO SC SC SC SC SC SC BT BT SA 
             1 TO SC SC SC SC SC BT BT SC SA 
             1 TO SC BT AB 
             1 TO SC SC SC SC SC BT BT SC AB 
             1 TO SC SC SC SC SC BT BT BT SC 
             1 TO SC SC BT AB 
             1 TO SC BT BT AB 
             1 TO SC SC SC SC SC SC BT BT SC 
             1 TO SC BT BT SA 
             1 TO SC SC SC SC BT BT SC SA 
             1 TO SC BT BT SC SA 
             1 TO SC SC SC SC SC SC BT BT BT 
             1 TO SC BT BT SC SC SC SC SA 
             1 TO SC SC SC BT BT BT BT AB 
             1 TO SC BT BT SC SC AB 
             1 TO SC SC SC BT BT AB 
             1 TO SC SC SC SC BT BT SC AB 
    [2016-08-19 16:56:02,597] p24445 {/Users/blyth/opticks/ana/seq.py:28} WARNING - code bad abbr [?0?] s [TO SC SC BT BT ?0? BT BT] 
    [2016-08-19 16:56:02,597] p24445 {/Users/blyth/opticks/ana/seq.py:32} WARNING - code sees 1 bad abbr in [TO SC SC BT BT ?0? BT BT] 
             1 TO SC SC BT BT ?0? BT BT 
             1 TO SC SC SC SC BT BT SA 
             1 TO SC SC SC BT BT SC AB 
             1 TO SC SC SC SC SC BT AB 
             1 TO SC SC BT BT BT BT AB 
             1 TO SC BT BT BT BT BT AB 
             1 TO SC BT BT BT BT SA 
    8cbbbcd TO BT BR BR BR BT SA 8cbbbcd 



The PMT is in mineral oil so no RE is expected::

    simon:ana blyth$ ./histype.py --det PmtInBox --tag 10 --src torch 
    ./histype.py --det PmtInBox --tag 10 --src torch
    [2016-08-19 16:59:02,357] p24463 {./histype.py:55} INFO - loaded ph from /tmp/blyth/opticks/evt/PmtInBox/torch/10/ph.npy shape (100000, 1, 2) 
         67948 TO BT SA 
         21648 TO BT SD 
          4581 TO BT BT SA 
          3794 TO AB 
           640 TO SC SA 
           444 TO BT AB 
           350 TO BT BT AB 
           283 TO BR SA 
            81 TO SC BT SA 
            51 TO BT BT SC SA 
            40 TO SC AB 
            36 TO BT BR BR BT SA 
            28 TO BR AB 
            20 TO SC BT SD 
             9 TO BT BT SC BT BR BT SA 
             8 TO SC SC SA 
             7 TO SC BT BT SA 
             6 TO BR SC SA 
             4 TO BT BR BR BR BR BT BT BR BT 
             4 TO BT BR BR BT AB 
             3 TO SC BR SA 
             2 TO BT BT SC BT BT BT BT BT SA 
             2 TO BT BT SC BT BR BT AB 
             2 TO SC SC BT SA 
             1 TO BT BT SC BT BT BT BT BT BT 
             1 TO BT BR AB 
             1 TO BT BT SC BT BT BT BR BT BT 
             1 TO BT BR BR AB 
             1 TO SC BT BT AB 
             1 TO BT BT SC BT BT AB 
             1 TO BR SC BT BR BT SA 
             1 TO BT BT SC BT BT BR BR BR BR 
             1 TO BT BT SC AB 
    8cbbbcd TO BT BR BR BR BT SA 8cbbbcd 



