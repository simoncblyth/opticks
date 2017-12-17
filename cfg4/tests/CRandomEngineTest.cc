

#include  <iostream>
#include "PLOG.hh"

#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksHub.hh"

#include "Randomize.hh"

#include "CG4.hh"
#include "CFG4_LOG.hh"
#include "CRandomEngine.hh"


struct CRandomEngineTest
{
    CRandomEngineTest(CG4* g4)
       :
       _ctx(g4->getCtx()),
       _engine(g4)
    {
    }

    void print(unsigned record_id)
    {
        if(!_engine.hasSequence())
        {
             LOG(error) << " engine has no RNG sequences loaded " 
                        << " create input file " << _engine.getPath()
                        << " with TRngBufTest " 
                        ; 
             return ; 
        }

        _ctx._record_id = record_id ;   
        _engine.preTrack();     // <-- required to setup the curandSequence

        LOG(info) << "record_id " << record_id ; 
 
        for(int i=0 ; i < 29 ; i++)
        {
            double u = G4UniformRand() ;
            int idxf = _engine.findIndexOfValue(u); 

            std::cout 
                << " i "    << std::setw(5) << i 
                << " idxf " << std::setw(5) << idxf 
                << " u "    << std::setw(10) << u 
                << std::endl 
                ;  


        }

        //_engine.jump(-5) ;

 
    }

    CG4Ctx&       _ctx  ; 
    CRandomEngine _engine ; 
};


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    CFG4_LOG__ ; 

    LOG(info) << argv[0] ; 

    int pindex = argc > 1 ? atoi(argv[1]) : 0 ; 

    LOG(info) << " pindex " << pindex ; 

    Opticks ok(argc, argv );
    ok.setModeOverride( OpticksMode::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    OpticksHub hub(&ok) ; 
    
    CG4* g4 = new CG4(&hub) ; 

    CRandomEngineTest ret(g4) ; 
    ret.print(pindex); 

    return 0 ; 
}


