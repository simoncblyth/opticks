// npy-
#include "BStr.hh"
#include "BDirect.hh"

// okc-
#include "Opticks.hh"
#include "OpticksResource.hh"

// ggeo-
#include "GMesh.hh"

#include "MWrap.hh"
#include "MTool.hh"
#include "MMesh.hh"

#include "PLOG.hh"


MTool::MTool() : m_noise(0) 
{
}

std::string& MTool::getOut()
{
    return m_out ; 
}
std::string& MTool::getErr()
{
    return m_err ; 
}
unsigned int MTool::getNoise()
{
    return m_noise ; 
}


unsigned int MTool::countMeshComponents(GMesh* gmesh)
{
    unsigned int ret ; 
    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {    
         cout_redirect out_(coutbuf.rdbuf());
         cerr_redirect err_(cerrbuf.rdbuf()); 

         ret = countMeshComponents_(gmesh); 
    }    

    m_out = coutbuf.str();
    m_err = cerrbuf.str();
    m_noise = m_out.size() + m_err.size() ;
 
    return ret ; 
}



unsigned int MTool::countMeshComponents_(GMesh* gmesh)
{
    MWrap<MMesh> wsrc(new MMesh);

    wsrc.load(gmesh);

    unsigned int ncomp = wsrc.labelConnectedComponentVertices("component"); 

    return ncomp ; 
}


GMesh* MTool::joinSplitUnion(GMesh* gmesh, Opticks* opticks)
{
    OpticksResource* resource = opticks->getResource();

    // hmm this is pure static, could create an MTool instance
    // if find need to split this up a bit 

    LOG(info) << "MTool::joinSplitUnion " 
              << " index " << gmesh->getIndex() 
              << " shortname " << gmesh->getShortName()
              ;

    MWrap<MMesh> wsrc(new MMesh);

    wsrc.load(gmesh);

    int ncomp = wsrc.labelConnectedComponentVertices("component"); 

    LOG(trace) << "MTool::joinSplitUnion " 
               << " index " << gmesh->getIndex() 
               << " shortname " << gmesh->getShortName()
               << " ncomp " << ncomp
               ;

    if(ncomp != 2)
    {
        wsrc.dump("wsrc", 0);
        LOG(warning) << "MTool::joinSplitUnion expecting GMesh with topology count 2, but found : " << ncomp ; 
        return gmesh ; 
    }

    typedef MMesh::VertexHandle VH ; 
    typedef std::map<VH,VH> VHM ;

    MWrap<MMesh> wa(new MMesh);
    MWrap<MMesh> wb(new MMesh);

    VHM s2c_0 ;  
    wsrc.partialCopyTo(wa.getMesh(), "component", 0, s2c_0);

    VHM s2c_1 ;  
    wsrc.partialCopyTo(wb.getMesh(), "component", 1, s2c_1);

//#ifdef DEBUG 
    wa.dump("wa",0);
    wb.dump("wb",0);

    wa.write("$TMP/comp%d.off", 0 );
    wb.write("$TMP/comp%d.off", 1 );
//#endif

    wa.calcFaceCentroids("centroid"); 
    wb.calcFaceCentroids("centroid"); 

    // xyz delta maximum and w: minimal dot product of normals, -0.999 means very nearly back-to-back
    //glm::vec4 delta(10.f, 10.f, 10.f, -0.999 ); 

    glm::vec4 delta = resource->getMeshfixFacePairingCriteria();

    // centroid is the input property, paired is output 
    MWrap<MMesh>::labelSpatialPairs( wa.getMesh(), wb.getMesh(), delta, "centroid", "paired");

    LOG(trace) << "MWrap<MMesh>::labelSpatialPairs DONE " ; 

    wa.deleteFaces("paired");

    LOG(trace) << "wa.deleteFaces DONE " ; 

    wb.deleteFaces("paired");

    LOG(trace) << "wb.deleteFaces DONE " ; 

    wa.collectBoundaryLoop();
    wb.collectBoundaryLoop();

    VHM a2b = MWrap<MMesh>::findBoundaryVertexMap(&wa, &wb );  

    MWrap<MMesh> wdst(new MMesh);

    wdst.createWithWeldedBoundary( &wa, &wb, a2b );

    GMesh* result = wdst.createGMesh(); 
 
    return result  ; 
}




