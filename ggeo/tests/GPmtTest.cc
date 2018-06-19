//  op --pmt
//  op --pmt --apmtidx 0
//  op --pmt --apmtidx 2 
//  op --pmt --apmtslice 0:10
//

#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"

#include "Opticks.hh"

#include "GBndLib.hh"
#include "GPmt.hh"
#include "GParts.hh"
#include "GCSG.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_BODY.hh"

int main(int argc, char** argv)
{
    PLOG_COLOR(argc, argv);
    GGEO_LOG__ ;

    Opticks ok(argc, argv);
    ok.configure();

    for(int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 

    NSlice* slice = ok.getAnalyticPMTSlice();
    unsigned apmtidx = ok.getAnalyticPMTIndex();

    bool constituents = true ; 
    GBndLib* blib = GBndLib::load(&ok, constituents);
    blib->closeConstituents();

    GPmt* pmt = GPmt::load(&ok, blib, apmtidx, slice);

    LOG(info) << argv[0] << " apmtidx " << apmtidx << " pmt " << pmt ; 
    if(!pmt)
    {
        LOG(fatal) << argv[0] << " FAILED TO LOAD PMT " ; 
        return 0 ;
    }

    pmt->dump("GPmt::dump");
  
    return 0 ;
}


