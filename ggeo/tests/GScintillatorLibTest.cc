// op --gscintillatorlib

#include "NPY.hpp"

#include "Opticks.hh"
#include "GPropertyMap.hh"
#include "GScintillatorLib.hh"


#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_BODY.hh"



/*
In [1]: s = np.load("/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GScintillatorLib.npy")

In [2]: s
Out[2]: 
array([[[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.717],
        [ 207.546],
        [ 180.   ]]], dtype=float32)

In [3]: s.shape
Out[3]: (1, 4096, 1)



In [1]: a = np.load("/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GScintillatorLib.npy")

In [2]: a
Out[2]: 
array([[[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]],

       [[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]]], dtype=float32)

In [3]: a.shape
Out[3]: (2, 4096, 1)



In [4]: b = np.load("/tmp/GScintillatorLib.npy")

In [5]: b.shape
Out[5]: (2, 4096, 1)

In [6]: b
Out[6]: 
array([[[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]],

       [[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]]], dtype=float32)


In [7]: c = np.load("/tmp/GScintillatorLib0.npy")

In [8]: c
Out[8]: 
array([[[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]]], dtype=float32)

In [9]: c.shape
Out[9]: (1, 4096, 1)





 45 2016-07-06 17:58:35.348 INFO  [12948] [OpEngine::prepareOptiX@112] OpEngine::prepareOptiX (OColors)
 46 2016-07-06 17:58:35.348 INFO  [12948] [OpEngine::prepareOptiX@119] OpEngine::prepareOptiX (OSourceLib)
 47 2016-07-06 17:58:35.349 INFO  [12948] [OpEngine::prepareOptiX@124] OpEngine::prepareOptiX (OScintillatorLib)
 48 2016-07-06 17:58:35.349 INFO  [12948] [OScintillatorLib::makeReemissionTexture@39] OScintillatorLib::makeReemissionTexture  nx 4096 ny 1 ni 2 nj 4096 nk 1 step 0.000244141 empty 0
 49 *** Error in `/home/ihep/simon-dev-env/env-dev-2016july4/local/opticks/lib/OpEngineTest': free(): invalid next size (fast): 0x0000000002584420 ***
 50 ======= Backtrace: =========




*/


int main(int argc, char** argv)
{
    PLOG_(argc,argv);
    GGEO_LOG_ ; 


    Opticks* opticks = new Opticks(argc, argv);

    GScintillatorLib* slib = GScintillatorLib::load(opticks);
    slib->dump();

    const char* name = "LiquidScintillator" ;

    GPropertyMap<float>* ls = slib->getRaw(name);
    LOG(info) << " ls " << ls ; 
    if(ls)
    {
        ls->dump("ls");
    } 
    else
    {
         LOG(error) << " FAILED TO FIND " << name ;
    }



    NPY<float>* buf = slib->getBuffer();
    buf->Summary();
    const char* path = "/tmp/GScintillatorLib.npy" ;

    LOG(info) << " save GScintillatorLib buf  "
              << " to path " << path
              << " shape " << buf->getShapeString()
              ; 

    buf->save(path);


    NPY<float>* buf0 = buf->make_slice("0:1") ;
    const char* path0 = "/tmp/GScintillatorLib0.npy" ;

    LOG(info) << " save GScintillatorLib buf0  "
              << " to path " << path0
              << " shape " << buf0->getShapeString()
              ; 

    buf0->save(path0);

    


    return 0 ;
}

