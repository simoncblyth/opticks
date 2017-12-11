

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

        _ctx._record_id = 0 ;   
        _engine.pretrack();     // <-- required to setup the curandSequence

        LOG(info) << "record_id " << record_id ; 
 
        for(int i=0 ; i < 10 ; i++)
            std::cout << std::setw(5) << i << " : " << G4UniformRand() << std::endl  ;  

        _engine.jump(-5) ;

        for(int i=0 ; i < 10 ; i++)
            std::cout << std::setw(5) << i << " : " << G4UniformRand() << std::endl  ;  


 
    }

    CG4Ctx&       _ctx  ; 
    CRandomEngine _engine ; 
};


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    CFG4_LOG__ ; 

    LOG(info) << argv[0] ; 


    Opticks ok(argc, argv );
    ok.setModeOverride( OpticksMode::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    OpticksHub hub(&ok) ; 
    
    CG4* g4 = new CG4(&hub) ; 

    CRandomEngineTest ret(g4) ; 
    ret.print(0); 

    return 0 ; 
}


