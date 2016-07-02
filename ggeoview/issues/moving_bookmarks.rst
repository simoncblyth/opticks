Moving Bookmarks
==================

::

    ggv-;t jpmt;jpmt

    jpmt(){ 
        #local spa=retina
        local spa=hd
        ggv --jpmt --cerenkov --animtimemax 80 --load $(ggv-size-position $spa)  $*
    }



Position of bookmarks with *jpmt* when using *retina* frame size are as intended.
When using *hd* frame size (smaller with wider aspect ratio) bookmark positions change. 

Nope, issue occurs when a non-default phi rotation is in force the (as is case after doing a *V* rotate)
the bookmarks positions are greatly changed.

Workaround: include zero-phi into the Composition::home so *H* key regains intended bookmarks.



Enhance interpolated views to use newly minted bookmarks

::

    ggv-;t jpmt;jpmt --state expt



Hmm, must remember to commit trackballing into the view of newly created bookmark (SHIFT+num key) by 
afterwards pressing SPACE while on a bookmark for the interpolation to get the point.

Hmm, old bug that up is coming out zeros, which results in random up?
Workaround is to manually edit the bookmark setting the up::

     17 [view]
     18 eye=-0.4286,-0.4286,0.0000
     19 look=0.5714,0.5714,0.0000
     20 up=0.0000,0.0000,1.0000



Problem might be that the normalized up vector is just too small for the matrix, so scale by extent ?

::

    In [1]: t = "33550.0000,0.0000,0.0000,0.0000 0.0000,33550.0000,0.0000,0.0000 0.0000,0.0000,33550.0000,0.0000 0.0000,0.0000,9300.0000,1.0000"


    In [18]: a = np.fromstring(t.replace(" ",","),sep=",").reshape(4,4)

    In [19]: a
    Out[19]: 
    array([[ 33550.,      0.,      0.,      0.],
           [     0.,  33550.,      0.,      0.],
           [     0.,      0.,  33550.,      0.],
           [     0.,      0.,   9300.,      1.]])

    In [20]: i = np.linalg.inv(a)

    In [21]: i*1e7
    Out[21]: 
    array([[      298.063,         0.   ,         0.   ,         0.   ],
           [        0.   ,       298.063,         0.   ,         0.   ],
           [        0.   ,         0.   ,       298.063,         0.   ],
           [        0.   ,         0.   ,  -2771982.116,  10000000.   ]])



