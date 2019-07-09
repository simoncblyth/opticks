

#include  <iostream>
#include "PLOG.hh"

#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksHub.hh"

#include "Randomize.hh"

#include "CG4.hh"
#include "OPTICKS_LOG.hh"
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
                        << " create STATIC_CURAND input file " 
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
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    int pindex1 = argc > 1 ? atoi(argv[1]) : 0 ; 
    int pindex2 = argc > 2 ? atoi(argv[2]) : pindex1 + 1 ; 
    int pstep   = argc > 3 ? atoi(argv[3]) : 1 ; 

    LOG(info) 
        << " pindex1 " << pindex1 
        << " pindex2 " << pindex2  
        << " pstep " << pstep 
        ; 

    Opticks ok(argc, argv );
    ok.setModeOverride( OpticksMode::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    OpticksHub hub(&ok) ; 
    
    CG4* g4 = new CG4(&hub) ; 

    CRandomEngineTest ret(g4) ; 

    for(int pindex=pindex1 ; pindex < pindex2 ; pindex+=pstep )
    {
        ret.print(pindex); 
    }


    return 0 ; 
}


