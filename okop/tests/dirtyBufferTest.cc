#include "NPY.hpp"
#include "OpticksBufferControl.hh"

#include "Opticks.hh"
#include "OContext.hh"
#include "OBuf.hh"
#include "TBuf.hh"
#include "OXPPNS.hh"

#include "OPTICKS_LOG.hh"

/**

dirtyBufferTest
=================

* a failed attempt to reproduce the seeding interop failure 



**/


unsigned test_value(unsigned i)
{
    unsigned tv(0) ; 
    switch(i%10)
    {
       case 0: tv=42 ;break;
       case 1: tv=142 ;break;
       case 2: tv=242 ;break;
       case 3: tv=342 ;break;
       case 4: tv=442 ;break;
       case 5: tv=542 ;break;
       case 6: tv=642 ;break;
       case 7: tv=742 ;break;
       case 8: tv=842 ;break;
       case 9: tv=942 ;break;
    }
    return tv ; 
}

unsigned test_size(unsigned i)
{
    unsigned sz(0) ; 
    switch(i%10)
    {
       case 0: sz=100 ;break;
       case 1: sz=200 ;break;
       case 2: sz=300 ;break;
       case 3: sz=400 ;break;
       case 4: sz=500 ;break;
       case 5: sz=600 ;break;
       case 6: sz=500 ;break;
       case 7: sz=400 ;break;
       case 8: sz=300 ;break;
       case 9: sz=200 ;break;
    }
    return sz ; 
}



struct Evt {
    static Evt* make(unsigned t);
    Evt(unsigned size_, unsigned val_);
    virtual ~Evt();
    bool check();

    unsigned size ; 
    unsigned val ; 
    NPY<unsigned>* in_data ;
    NPY<unsigned>* out_data ;
};

Evt* Evt::make(unsigned t)
{
    unsigned tval = test_value(t) ; 
    unsigned tsize = test_size(t);
    LOG(info) << " t " << t << " tval " << tval << " tsize " << tsize   ; 
    Evt* evt = new Evt(tsize, tval) ;  
    return evt ;
}

Evt::Evt(unsigned size_, unsigned val_)
  :
  size(size_), 
  val(val_), 
  in_data(NPY<unsigned>::make(size,1,1)),
  out_data(NPY<unsigned>::make(size,1,1))
{
    OpticksBufferControl out_ctrl(out_data->getBufferControlPtr());
    out_ctrl.add(OpticksBufferControl::VERBOSE_MODE_);
    out_ctrl.add(OpticksBufferControl::OPTIX_NON_INTEROP_); // needed otherwise download is skipped

    in_data->fill(val);
}

bool Evt::check()
{
    return out_data->isConstant(val*2) ;
}

Evt::~Evt()
{
   in_data->reset();
   out_data->reset();
}
 

struct OEvt 
{
   OEvt(optix::Context& context, unsigned size);

   optix::Buffer  in_buffer ; 
   optix::Buffer out_buffer ; 
   OBuf*  ibuf ; 
   OBuf*  obuf ; 

   void resize(unsigned size);
};

OEvt::OEvt(optix::Context& context, unsigned size )  
{
    RTformat fmt = RT_FORMAT_UNSIGNED_INT ;

    in_buffer = context->createBuffer( RT_BUFFER_INPUT );
    out_buffer = context->createBuffer( RT_BUFFER_OUTPUT ); 

    in_buffer->setFormat(fmt);
    out_buffer->setFormat(fmt);

    ibuf = new OBuf("in",in_buffer);
    obuf = new OBuf("out",out_buffer);

    resize(size);

    context["in_buffer"]->setBuffer(in_buffer);  
    context["out_buffer"]->setBuffer(out_buffer);  
}

void OEvt::resize(unsigned size)
{
    in_buffer->setSize(size);
    out_buffer->setSize(size);
}


void pure_upload_launch_download( unsigned ntest, 
                                  int entry, 
                                  OContext* ctx, 
                                  OEvt& oevt
                                )

{
    for(unsigned t=0 ; t < ntest ; t++ )
    {
        Evt* evt = Evt::make(t) ;  

        oevt.resize( evt->size ) ; 

        OContext::upload<unsigned>( oevt.in_buffer, evt->in_data );

        ctx->launch( OContext::LAUNCH, entry, evt->size, 1, NULL ); 

        OContext::download<unsigned>( oevt.out_buffer, evt->out_data ); 

        assert(evt->check()) ;

        delete evt ;  
    } 
}

// TBuf::upload is standin for seeding 

/** 
   in real case, 

     1)  upload genstep buffer
     2)  invoke seeder to read from genstep and write to seeds 


**/


void dirty_upload_launch_download( unsigned ntest, 
                                   int entry, 
                                   OContext* ctx, 
                                   OEvt& oevt
 )
{
    for(unsigned t=0 ; t < ntest ; t++ )
    {
        Evt* evt = Evt::make(t) ;  

        oevt.resize( evt->size ) ; 

        // OBuf state comes entirely(?) from the optix::Buffer handle, so just 
        // have to make sure to resize optix Buffer first to get a correct CBufSpec

        CBufSpec s_in = oevt.ibuf->bufspec();   // getDevicePointer happens here with OBufBase::bufspec

        TBuf t_in("in", s_in );

        //t_in.upload(evt->in_data);   // THIS WORKS
        t_in.fill(evt->val);           // ALSO WORKS 


        ctx->launch( OContext::LAUNCH, entry, evt->size, 1, NULL ); 

        OContext::download<unsigned>( oevt.out_buffer, evt->out_data ); 

        assert(evt->check()) ;

        delete evt ;  
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned ntest = 10 ; 
    unsigned size = 100 ; 

    Opticks ok(argc, argv, "--compute");
    ok.configure() ;

   
    
    OContext* ctx = OContext::Create(&ok );
 
    optix::Context context = ctx->getContext() ;

    OEvt oevt(context, size );

    int entry = ctx->addEntry("dirtyBufferTest.cu", "dirtyBufferTest", "exception");

    ctx->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, NULL);

    //pure_upload_launch_download( ntest, entry, ctx, oevt );
    dirty_upload_launch_download( ntest, entry, ctx, oevt );

    delete ctx ; 


    return 0 ; 
}
