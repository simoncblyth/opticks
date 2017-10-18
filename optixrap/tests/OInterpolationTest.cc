#include "SSys.hh"
#include "BFile.hh"
#include "NPY.hpp"

#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"

#include "OBndLib.hh"
#include "OLaunchTest.hh"
#include "OContext.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "OScene.hh"


#include "SYSRAP_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "OXRAP_LOG.hh"

#include "PLOG.hh"

/**
OInterpolationTest
===================

**/

class OInterpolationTest 
{
    public:
         OInterpolationTest(Opticks* ok, OContext* ocontext, OBndLib* obnd, const char* base);
         void launch(optix::Context& context);
         int ana();
    private:
         Opticks*    m_ok ;
         bool        m_interpol ; 
         const char* m_progname ; 
         const char* m_name ; 
         const char* m_ana ; 
         OContext*   m_ocontext;
         OBndLib*    m_obnd ;
         const char* m_base ;  
         unsigned    m_nb ; 
         unsigned    m_nx ; 
         unsigned    m_ny ; 
         NPY<float>* m_out ; 
};



OInterpolationTest::OInterpolationTest(Opticks* ok, OContext* ocontext, OBndLib* obnd, const char* base)
   :
     m_ok(ok),
     m_interpol(!ok->hasOpt("nointerpol")),
     m_progname( m_interpol ? "OInterpolationTest" : "OIdentityTest" ),
     m_name(NULL),
     m_ana(NULL),
     m_ocontext(ocontext),
     m_obnd(obnd),
     m_base(strdup(base)), 
     m_nb(obnd->getNumBnd()), 
     m_out(NULL)
{
     
    if(m_interpol)
    {
        m_name = "OInterpolationTest_interpol.npy" ;
        m_ana = "$OPTICKS_HOME/optixrap/tests/OInterpolationTest_interpol.py" ;
        m_nx = 820 - 60 + 1 ;     // 761 : 1 nm steps 
        m_ny = m_obnd->getHeight();  // total number of float4 props
    }
    else  // identity 
    {
        m_name = "OInterpolationTest_identity.npy" ;
        m_ana = "$OPTICKS_HOME/optixrap/tests/OInterpolationTest_identity.py" ;
        m_nx = m_obnd->getWidth();   // number of wavelength samples
        m_ny = m_obnd->getHeight();  // total number of float4 props
    } 
}


void OInterpolationTest::launch(optix::Context& context)
{
    LOG(info) << "OInterpolationTest::launch"
              << " nb " << std::setw(5) << m_nb
              << " nx " << std::setw(5) << m_nx
              << " ny " << std::setw(5) << m_ny
              << " progname " << std::setw(30) << m_progname
              << " name " << m_name 
              << " base " << m_base 
              ;

    optix::Buffer buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_nx, m_ny);
    context["out_buffer"]->setBuffer(buffer);   

    OLaunchTest olt(m_ocontext, m_ok, "OInterpolationTest.cu.ptx", m_progname, "exception");
    olt.setWidth(  m_nx );   
    olt.setHeight( m_ny/8 );   // kernel loops over eight for matsur and num_float4
    olt.launch();

    m_out = NPY<float>::make(m_nx, m_ny, 4);
    m_out->read( buffer->map() );
    buffer->unmap(); 
    m_out->save(m_base, m_name);
}


int OInterpolationTest::ana()
{
    std::string path = BFile::FormPath(m_ana);
    return SSys::exec("python",path.c_str());
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);    

    SYSRAP_LOG__ ; 
    OKCORE_LOG__ ; 
    GGEO_LOG__ ; 
    OXRAP_LOG__ ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);
    OScene sc(&hub);

    LOG(info) << " ok " ; 

    OContext* ocontext = sc.getOContext();
    optix::Context context = ocontext->getContext();

    const char* base = "$TMP/InterpolationTest"  ; 

    OBndLib* obnd = sc.getOBndLib();

    GBndLib* blib = obnd->getBndLib();
    blib->saveAllOverride(base);     

    // Not appropriate to write into geocache from a test
    // BUT ok to override write into $TMP, allowing 
    // python analysis to pick up the potentially dynamic boundary names

    OInterpolationTest tst(&ok , ocontext, obnd, base);
    tst.launch(context);
    return tst.ana();

}

