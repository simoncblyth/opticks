// op --gscintillatorlib



#include "Opticks.hh"
#include "GPropertyMap.hh"
#include "GScintillatorLib.hh"


#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_CC.hh"



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




*/


int main(int argc, char** argv)
{
    PLOG_(argc,argv);
    GGEO_LOG_ ; 


    Opticks* opticks = new Opticks(argc, argv, "scint.log");

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



    return 0 ;
}

