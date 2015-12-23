#include "NumpyEvt.hpp"

#include "uif.h"
#include "NPY.hpp"
#include "G4StepNPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Parameters.hpp"
#include "GLMFormat.hpp"
#include "Timer.hpp"
#include "Index.hpp"
#include "stringutil.hpp"

#include "limits.h"
#include "assert.h"
#include <sstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* NumpyEvt::genstep = "genstep" ; 
const char* NumpyEvt::photon  = "photon" ; 
const char* NumpyEvt::record  = "record" ; 
const char* NumpyEvt::phosel = "phosel" ; 
const char* NumpyEvt::recsel  = "recsel" ; 
const char* NumpyEvt::sequence  = "sequence" ; 
const char* NumpyEvt::aux = "aux" ; 

void NumpyEvt::init()
{
    m_timer = new Timer("NumpyEvt"); 
    m_timer->setVerbose(false);
    m_parameters = new Parameters ;
}

ViewNPY* NumpyEvt::operator [](const char* spec)
{
    std::vector<std::string> elem ; 
    split(elem, spec, '.');

    if(elem.size() != 2 ) assert(0);

    MultiViewNPY* mvn(NULL); 
    if(     elem[0] == genstep) mvn = m_genstep_attr ;  
    else if(elem[0] == photon)  mvn = m_photon_attr ;
    else if(elem[0] == record)  mvn = m_record_attr ;
    else if(elem[0] == phosel)  mvn = m_phosel_attr ;
    else if(elem[0] == recsel)  mvn = m_recsel_attr ;
    else if(elem[0] == sequence)  mvn = m_sequence_attr ;
    else if(elem[0] == aux)      mvn = m_aux_attr ;

    assert(mvn);
    return (*mvn)[elem[1].c_str()] ;
}



void NumpyEvt::setGenstepData(NPY<float>* genstep)
{
    m_genstep_data = genstep  ;

    //                                                j k l sz   type        norm   iatt
    ViewNPY* vpos = new ViewNPY("vpos",m_genstep_data,1,0,0,4,ViewNPY::FLOAT,false,false);    // (x0, t0)                     2nd GenStep quad 
    ViewNPY* vdir = new ViewNPY("vdir",m_genstep_data,2,0,0,4,ViewNPY::FLOAT,false,false);    // (DeltaPosition, step_length) 3rd GenStep quad

    m_genstep_attr = new MultiViewNPY("genstep_attr");
    m_genstep_attr->add(vpos);
    m_genstep_attr->add(vdir);

    // attribute offset calulated by  npy->getByteIndex(0,j,k) 
    // assuming the size of the attribute type matches that of the NPY<T>

    m_num_gensteps = m_genstep_data->getShape(0) ;
    m_num_photons = m_genstep_data->getUSum(0,3);

    m_timer->start();

    createHostBuffers();
    createHostIndexBuffers();

    m_timer->stop();
    //m_timer->dump();
}


void NumpyEvt::prepareForIndexing()
{
    assert(m_num_photons > 0 );
    createHostIndexBuffers();
}


void NumpyEvt::createHostBuffers()
{
    LOG(info) << "NumpyEvt::createHostBuffers "
              << " flat " << m_flat 
              << " m_num_photons " << m_num_photons  
              << " m_maxrec " << m_maxrec
               ;

    NPY<float>* pho = NPY<float>::make(m_num_photons, 4, 4); // must match GPU side photon.h:PNUMQUAD
    setPhotonData(pho);   

    NPY<unsigned long long>* seq = NPY<unsigned long long>::make(m_num_photons, 1, 2);  // shape (np,1,2) (formerly initialized to 0)
    setSequenceData(seq);   

    if(m_flat)
    {
        unsigned int num_records = getNumRecords();
        NPY<short>* rec = NPY<short>::make(num_records, 2, 4);  // shape (nr,2,4) formerly initialized to SHRT_MIN
        setRecordData(rec);   

        NPY<unsigned char>* recsel = NPY<unsigned char>::make(num_records,1,4); // shape (nr,1,4) (formerly initialized to 0) 
        setRecselData(recsel);   

        NPY<short>* aux = NPY<short>::make(num_records, 1, 4);  // shape (nr,1,4)
        setAuxData(aux);   
    }
    else
    {
        NPY<short>* rec = NPY<short>::make(m_num_photons, m_maxrec, 2, 4); 
        setRecordData(rec);   

        NPY<unsigned char>* recsel = NPY<unsigned char>::make(m_num_photons, m_maxrec,1,4); // shape (nr,1,4) (formerly initialized to 0) 
        setRecselData(recsel);   

        NPY<short>* aux = NPY<short>::make(m_num_photons, m_maxrec, 1, 4);  // shape (nr,1,4)
        setAuxData(aux);   
    }


    NPY<float>* fdom = NPY<float>::make(3,1,4);
    setFDomain(fdom);

    NPY<int>* idom = NPY<int>::make(1,1,4);
    setIDomain(idom);

    // these small ones can be zeroed directly 
    fdom->zero();
    idom->zero();

    LOG(info) << "NumpyEvt::createHostBuffers DONE " ;

    (*m_timer)("createHostBuffers");
}





void NumpyEvt::createHostIndexBuffers()
{
    LOG(info) << "NumpyEvt::createHostIndexBuffers "
              << " flat " << m_flat 
              << " m_num_photons " << m_num_photons  
              << " m_maxrec " << m_maxrec
               ;

    NPY<unsigned char>* phosel = NPY<unsigned char>::make(m_num_photons,1,4); // shape (np,1,4) (formerly initialized to 0)
    setPhoselData(phosel);   

    if(m_flat)
    {
        unsigned int num_records = getNumRecords();

        NPY<unsigned char>* recsel = NPY<unsigned char>::make(num_records,1,4); // shape (nr,1,4) (formerly initialized to 0) 
        setRecselData(recsel);   
    }
    else
    {
        NPY<unsigned char>* recsel = NPY<unsigned char>::make(m_num_photons, m_maxrec,1,4); // shape (nr,1,4) (formerly initialized to 0) 
        setRecselData(recsel);   
    }
}





void NumpyEvt::dumpDomains(const char* msg)
{
    LOG(info) << msg 
              << "\n space_domain      " << gformat(m_space_domain)
              << "\n time_domain       " << gformat(m_time_domain)
              << "\n wavelength_domain " << gformat(m_wavelength_domain)
              ;
}

void NumpyEvt::updateDomainsBuffer()
{
    NPY<float>* fdom = getFDomain();

    fdom->setQuad(m_space_domain     , 0);
    fdom->setQuad(m_time_domain      , 1);
    fdom->setQuad(m_wavelength_domain, 2);

    NPY<int>* idom = getIDomain();
    idom->setQuad(m_settings, 0 );
}

void NumpyEvt::readDomainsBuffer()
{
    NPY<float>* fdom = getFDomain();

    if(fdom)
    {
        m_space_domain = fdom->getQuad(0);
        m_time_domain = fdom->getQuad(1);
        m_wavelength_domain = fdom->getQuad(2);
    }
    else
    {
        LOG(warning) << "NumpyEvt::readDomainsBuffer"
                     << " fdom NULL "
                     ;
    }


    NPY<int>* idom = getIDomain();

    if(idom)
    {
        m_settings = idom->getQuad(0); 
        m_maxrec = m_settings.w ; 

        LOG(info) << "NumpyEvt::readDomainsBuffer" 
                  << " from idom settings m_maxrec " << m_maxrec 
                  ;
    }
    else
    {
        LOG(warning) << "NumpyEvt::readDomainsBuffer"
                     << " idom NULL "
                     ;
 
    }
 

}



void NumpyEvt::zero()
{
    m_photon_data->zero();
    m_phosel_data->zero();
    m_record_data->zero();
    m_recsel_data->zero();
    m_sequence_data->zero();
    m_aux_data->zero();
}



void NumpyEvt::seedPhotonData()
{
    //
    // NB cf with 
    //           Scene::uploadEvt
    //           Scene::uploadSelection
    //           OptiXEngine::init
    //
    // stuff genstep index into the photon allocation 
    // to allow generation to access appropriate genstep 
    //
    // see thrustrap-/iexpand.h and iexpandTest.cc for an 
    // initial try at moving genstep identification per photon
    // to GPU side  
    //

    G4StepNPY gs(m_genstep_data);  

    unsigned int numStep   = m_genstep_data->getShape(0);
    unsigned int numPhoton = m_photon_data->getShape(0);
    assert(numPhoton == m_num_photons);

    unsigned int count(0) ;
    for(unsigned int index=0 ; index < numStep ; index++)
    {
        unsigned int npho = m_genstep_data->getUInt(index, 0, 3);
        if(gs.isCerenkovStep(index))
        {
            //assert(npho > 0 && npho < 150); // by observation of Cerenkov steps
            assert(npho > 0 && npho < 3000);  
        }
        else if(gs.isScintillationStep(index))
        {
            assert(npho >= 0 && npho < 5000);     // by observation of Scintillation steps                  
        } 

        for(unsigned int n=0 ; n < npho ; ++n)
        { 
            assert(count < numPhoton);
            m_photon_data->setUInt(count, 0,0,0, index );  // set "phead" : repeating step index for every photon to be generated for the step
            count += 1 ;         
        }  // over photons for each step
    }      // over gen steps


    LOG(info) << "NumpyEvt::setGenstepData " 
              << " stepId(0) " << gs.getStepId(0) 
              << " genstep length " << numStep 
              << " photon length " << numPhoton
              << "  num_photons " << m_num_photons  ; 

    assert(count == m_num_photons ); 
    assert(count == numPhoton ); 
    // not m_num_photons-1 as last incremented count value is not used by setUInt
    (*m_timer)("seedPhotonData");
}


void NumpyEvt::setPhotonData(NPY<float>* photon_data)
{
    m_photon_data = photon_data  ;
    if(m_num_photons == 0) 
    {
        m_num_photons = photon_data->getShape(0) ;

        LOG(info) << "NumpyEvt::setPhotonData"
                  << " setting m_num_photons from shape(0) " << m_num_photons 
                  ;
    }
    else
    {
        assert(m_num_photons == photon_data->getShape(0));
    }

    m_photon_data->setDynamic();  // need to update with seeding so GL_DYNAMIC_DRAW needed 
    m_photon_attr = new MultiViewNPY("photon_attr");
    //                                                  j k l,sz   type          norm   iatt
    m_photon_attr->add(new ViewNPY("vpos",m_photon_data,0,0,0,4,ViewNPY::FLOAT, false, false));      // 1st quad
    m_photon_attr->add(new ViewNPY("vdir",m_photon_data,1,0,0,4,ViewNPY::FLOAT, false, false));      // 2nd quad
    m_photon_attr->add(new ViewNPY("vpol",m_photon_data,2,0,0,4,ViewNPY::FLOAT, false, false));      // 3rd quad
    m_photon_attr->add(new ViewNPY("iflg",m_photon_data,3,0,0,4,ViewNPY::INT  , false, true ));      // 4th quad

    //
    //  photon array 
    //  ~~~~~~~~~~~~~
    //     
    //  vpos  xxxx yyyy zzzz wwww    position, time           [:,0,:4]
    //  vdir  xxxx yyyy zzzz wwww    direction, wavelength    [:,1,:4]
    //  vpol  xxxx yyyy zzzz wwww    polarization weight      [:,2,:4] 
    //  iflg  xxxx yyyy zzzz wwww                             [:,3,:4]
    //
    //
    //  record array
    //  ~~~~~~~~~~~~~~
    //       
    //              4*short(snorm)
    //          ________
    //  rpos    xxyyzzww 
    //  rpol->  xyzwaabb <-rflg 
    //          ----^^^^
    //     4*ubyte     2*ushort   
    //     (unorm)     (iatt)
    //
    //
    //
    // corresponds to GPU side cu/photon.h:psave and rsave 
    //
}

void NumpyEvt::setAuxData(NPY<short>* aux_data)
{
    m_aux_data = aux_data  ;
    m_aux_attr = new MultiViewNPY("aux_attr");
    //                                            j k l sz   type                  norm   iatt
    ViewNPY* ibnd = new ViewNPY("ibnd",m_aux_data,0,0,0,4,ViewNPY::SHORT          ,false,  true);
    m_aux_attr->add(ibnd);
}

void NumpyEvt::setRecordData(NPY<short>* record_data)
{
    m_record_data = record_data  ;

    //                                               j k l sz   type                  norm   iatt
    ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0,4,ViewNPY::SHORT          ,true,  false);
    ViewNPY* rpol = new ViewNPY("rpol",m_record_data,1,0,0,4,ViewNPY::UNSIGNED_BYTE  ,true,  false);   

    ViewNPY* rflg = new ViewNPY("rflg",m_record_data,1,2,0,2,ViewNPY::UNSIGNED_SHORT ,false, true);   
    // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  

    ViewNPY* rflq = new ViewNPY("rflq",m_record_data,1,2,0,4,ViewNPY::UNSIGNED_BYTE  ,false, true);   
    // NB k=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo
    

    // ViewNPY::TYPE need not match the NPY<T>,
    // OpenGL shaders will view the data as of the ViewNPY::TYPE, 
    // informed via glVertexAttribPointer/glVertexAttribIPointer 
    // in oglrap-/Rdr::address(ViewNPY* vnpy)
 
    // standard byte offsets obtained from from sizeof(T)*value_offset 
    //rpol->setCustomOffset(sizeof(unsigned char)*rpol->getValueOffset());
    // this is not needed

    m_record_attr = new MultiViewNPY("record_attr");

    m_record_attr->add(rpos);
    m_record_attr->add(rpol);
    m_record_attr->add(rflg);
    m_record_attr->add(rflq);
}


void NumpyEvt::setPhoselData(NPY<unsigned char>* phosel_data)
{
    m_phosel_data = phosel_data ;
    if(!m_phosel_data) return ; 

    //                                               j k l sz   type                norm   iatt
    ViewNPY* psel = new ViewNPY("psel",m_phosel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true);
    m_phosel_attr = new MultiViewNPY("phosel_attr");
    m_phosel_attr->add(psel);
}


void NumpyEvt::setRecselData(NPY<unsigned char>* recsel_data)
{
    m_recsel_data = recsel_data ;
    if(!m_recsel_data) return ; 
    //                                               j k l sz   type                norm   iatt
    ViewNPY* rsel = new ViewNPY("rsel",m_recsel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true);
    m_recsel_attr = new MultiViewNPY("recsel_attr");
    m_recsel_attr->add(rsel);
}


void NumpyEvt::setSequenceData(NPY<unsigned long long>* sequence_data)
{
    m_sequence_data = sequence_data  ;
    assert(sizeof(unsigned long long) == 4*sizeof(unsigned short));  
    //
    // 64 bit uint used to hold the sequence flag sequence 
    // is presented to OpenGL shaders as 4 *16bit ushort 
    // as intend to reuse the sequence bit space for the indices and count 
    // via some diddling 
    //
    //      Have not taken the diddling route, 
    //      instead using separate Recsel/Phosel buffers for the indices
    // 
    //                                                 j k l sz   type                norm   iatt
    ViewNPY* phis = new ViewNPY("phis",m_sequence_data,0,0,0,4,ViewNPY::UNSIGNED_SHORT,false,  true);
    ViewNPY* pmat = new ViewNPY("pmat",m_sequence_data,0,1,0,4,ViewNPY::UNSIGNED_SHORT,false,  true);
    m_sequence_attr = new MultiViewNPY("sequence_attr");
    m_sequence_attr->add(phis);
    m_sequence_attr->add(pmat);

}


void NumpyEvt::dumpPhotonData()
{
    if(!m_photon_data) return ;
    dumpPhotonData(m_photon_data);
}

void NumpyEvt::dumpPhotonData(NPY<float>* photons)
{
    std::cout << photons->description("NumpyEvt::dumpPhotonData") << std::endl ;

    for(unsigned int i=0 ; i < photons->getShape(0) ; i++)
    {
        if(i%10000 == 0)
        {
            unsigned int ux = photons->getUInt(i,0,0); 
            float fx = photons->getFloat(i,0,0); 
            float fy = photons->getFloat(i,0,1); 
            float fz = photons->getFloat(i,0,2); 
            float fw = photons->getFloat(i,0,3); 
            printf(" ph  %7u   ux %7u   fxyzw %10.3f %10.3f %10.3f %10.3f \n", i, ux, fx, fy, fz, fw );             
        }
    }  
}

std::string NumpyEvt::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " ;
    if(m_genstep_data)  ss << m_genstep_data->description("m_genstep_data") ;
    if(m_photon_data)   ss << m_photon_data->description("m_photon_data") ;
    return ss.str();
}



void NumpyEvt::save(bool verbose)
{
    const char* udet = strlen(m_cat) > 0 ? m_cat : m_det ; 

    LOG(info) << "NumpyEvt::save"
              << " typ: " << m_typ
              << " tag: " << m_tag
              << " det: " << m_det
              << " cat: " << m_cat
              << " udet: " << udet 
              ;    

    // genstep normally not saved as it exists already coming from elsewhere,
    //  but for TorchStep that insnt the case

    NPY<float>* dpho = getPhotonData();
    dpho->setVerbose(verbose);
    dpho->save("ox%s", m_typ,  m_tag, udet);

    NPY<short>* drec = getRecordData();    
    drec->setVerbose(verbose);
    drec->save("rx%s", m_typ,  m_tag, udet);

    NPY<unsigned long long>* dhis = getSequenceData();
    dhis->setVerbose(verbose);
    dhis->save("ph%s", m_typ,  m_tag, udet);

    NPY<short>* daux = getAuxData();
    daux->setVerbose(verbose);
    daux->save("au%s", m_typ,  m_tag, udet);



    NPY<unsigned char>* ps = getPhoselData();
    ps->setVerbose(verbose);
    ps->save("ps%s", m_typ,  m_tag, udet);

    NPY<unsigned char>* rs = getRecselData();
    rs->setVerbose(verbose);
    rs->save("rs%s", m_typ,  m_tag, udet);


    updateDomainsBuffer();

    NPY<float>* fdom = getFDomain();
    fdom->save("fdom%s", m_typ,  m_tag, udet);

    NPY<int>* idom = getIDomain();
    idom->save("idom%s", m_typ,  m_tag, udet);


    if(m_seqhis)
    {
        std::string sh_dir = NPYBase::directory("sh%s", m_typ, udet );
        m_seqhis->save(sh_dir.c_str(), m_tag);        
    }

    if(m_seqmat)
    {
        std::string sm_dir = NPYBase::directory("sm%s", m_typ, udet );
        m_seqmat->save(sm_dir.c_str(), m_tag);        
    }


}


void NumpyEvt::load(bool verbose)
{
    const char* udet = strlen(m_cat) > 0 ? m_cat : m_det ; 

    NPY<int>*   idom = NPY<int>::load("idom%s", m_typ,  m_tag, udet );
    if(!idom)
    {
        m_noload = true ; 
        LOG(warning) << "NumpyEvt::load NO SUCH EVENT : RUN WITHOUT --load OPTION TO CREATE IT " 
                     << " typ: " << m_typ
                     << " tag: " << m_tag
                     << " det: " << m_det
                     << " cat: " << m_cat
                     << " udet: " << udet 
                    ;     
        return ; 
    }

    NPY<float>* fdom = NPY<float>::load("fdom%s", m_typ,  m_tag, udet );

    setIDomain(idom);
    setFDomain(fdom);

    readDomainsBuffer();
    dumpDomains("NumpyEvt::load dumpDomains");

    std::string sh_dir = NPYBase::directory("sh%s", m_typ, udet );
    m_seqhis = Index::load(sh_dir.c_str(), m_tag, "History_Sequence" );
 
    std::string sm_dir = NPYBase::directory("sm%s", m_typ, udet );
    m_seqmat = Index::load(sm_dir.c_str(), m_tag, "Material_Sequence");


    NPY<float>* ox = NPY<float>::load("ox%s", m_typ,  m_tag, udet );
    NPY<unsigned long long>* ph = NPY<unsigned long long>::load("ph%s", m_typ,  m_tag, udet );
    NPY<short>* rx = NPY<short>::load("rx%s", m_typ,  m_tag, udet );
    NPY<short>* au = NPY<short>::load("au%s", m_typ,  m_tag, udet );

    NPY<unsigned char>* ps = NPY<unsigned char>::load("ps%s", m_typ,  m_tag, udet );
    NPY<unsigned char>* rs = NPY<unsigned char>::load("rs%s", m_typ,  m_tag, udet );


    unsigned int num_photons = ox->getShape(0);
    unsigned int num_history = ph->getShape(0);
    unsigned int num_phosel  = ps ? ps->getShape(0) : 0 ;
    assert(num_photons == num_history );
    assert(num_phosel == 0 || num_photons == num_phosel );

    unsigned int num_records = rx->getShape(0);
    unsigned int num_aux     = au->getShape(0);
    unsigned int num_recsel  = rs ? rs->getShape(0) : 0 ;
    assert(num_records == num_aux ); 
    assert(num_recsel == 0 || num_records == num_recsel );

    if(num_records == num_photons*m_maxrec)
    {
        LOG(info) << "NumpyEvt::load flat records (Opticks style) detected " ;
    } 
    else if(num_records == num_photons)
    {
        LOG(info) << "NumpyEvt::load non-flat records (cfg4- style) detected :  RESHAPING " ;
        {
            rx->Summary("rx init");
            unsigned int ni = rx->getShape(0);
            unsigned int nj = rx->getShape(1);
            unsigned int nk = rx->getShape(2);
            unsigned int nl = rx->getShape(3);
            rx->reshape(ni*nj, nk, nl, 0);
            rx->Summary("rx reshaped");
        }       
        
        if(rs)
        {
            rs->Summary("rs init");
            unsigned int ni = rs->getShape(0);
            unsigned int nj = rs->getShape(1);
            unsigned int nk = rs->getShape(2);
            unsigned int nl = rs->getShape(3);
            rs->reshape(ni*nj, nk, nl, 0);
            rs->Summary("rs reshaped");
        }       
        {
            au->Summary("au init");
            unsigned int ni = au->getShape(0);
            unsigned int nj = au->getShape(1);
            unsigned int nk = au->getShape(2);
            unsigned int nl = au->getShape(3);
            au->reshape(ni*nj, nk, nl, 0);       
            au->Summary("au reshaped");
        }
    }

    setPhotonData(ox);
    setSequenceData(ph);
    setRecordData(rx);
    setAuxData(au);

    setPhoselData(ps);
    setRecselData(rs);

    if(verbose)
    {
        ox->Summary("ox");
        rx->Summary("rx");
        ph->Summary("ph");
        au->Summary("au");
        fdom->Summary("fdom");
        idom->Summary("idom");

        if(ps) ps->Summary("ps");
        if(rs) rs->Summary("rs");
    }

}

bool NumpyEvt::isIndexed()
{
    return m_phosel_data != NULL && m_recsel_data != NULL && m_seqhis != NULL && m_seqmat != NULL ;
}




NPY<float>* NumpyEvt::loadGenstepFromFile(int modulo)
{
    LOG(info) << "NumpyEvt::loadGenstepFromFile  "
              << " typ " << m_typ
              << " tag " << m_tag
              << " det " << m_det
              ;

    NPY<float>* npy = NPY<float>::load(m_typ, m_tag, m_det ) ;

    m_parameters->add<std::string>("genstepAsLoaded",   npy->getDigestString()  );

    m_parameters->add<int>("Modulo", modulo );

    if(modulo > 0)
    {
        LOG(warning) << "App::loadGenstepFromFile applying modulo scaledown " << modulo ;
        npy = NPY<float>::make_modulo(npy, modulo);
        m_parameters->add<std::string>("genstepModulo",   npy->getDigestString()  );
    }
    return npy ;
}

