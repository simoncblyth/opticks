Reimplementing the Composition tree of classes for the new workflow ?
=======================================================================

* Prior :doc:`remove-OpticksCore-dependency-from-CSGOptiX`

Overview
----------

*Composition* has a boatload of functionality used for interactive visualization control
implemented in a raft of classes : Camera, View, Trackball, ... 
For CSGOptiX ray trace rendering the vast majority of this is not used. 

Core functionality that is needed::

* Composition::setCenterExtent
* Composition::getEyeUVW

TODO: reimplement this core into sysrap structs 

* SGLM
* SCenterExtentFrame 


The modular structure of Composition would make it more effort 
that its worth to replace its innards with the new implementation. 

Interactive visualization in the new workflow is not on the horizon,
due to laptop constraints anyway. 
Non-interactive snap rendering is sufficient, and more appropriate for
working with a remote GPU workstation. 

Longterm interactive visualization needs a full reimplementation anyhow, 
Composition is very old code that has become far too organic. 


Other Players : SCenterExtentFrame, SGLM
---------------------------------------------

* SCenterExtentFrame provides model2world and world2model (with RTP tangential frame implementation)




Composition::setCenterExtent
-------------------------------

::

    1401 void Composition::setCenterExtent(const glm::vec4& ce, bool autocam, const qat4* m2w, const qat4* w2m )
    1402 {   
    1403     // this is invoked by App::uploadGeometry/Scene::setTarget
    1404     
    1405     m_center_extent.x = ce.x ;
    1406     m_center_extent.y = ce.y ;
    1407     m_center_extent.z = ce.z ;
    1408     m_center_extent.w = ce.w ;
    1409     m_extent = ce.w ;
    1410     
    1411     //setModel2World_old(ce); 
    1412     
    1413     bool m2w_valid = m2w && m2w->is_identity() == false ;
    1414     bool w2m_valid = w2m && w2m->is_identity() == false ;
    1415     
    1416     if( m2w_valid && w2m_valid )
    1417     {   
    1418         setModel2World_qt(m2w, w2m);
    ///   externally provided model2world 
    1419     }
    1420     else
    1421     {   
    1422         bool rtp_tangential = false ; 
    1423         setModel2World_ce(ce, rtp_tangential );
    1424     }
    1425     
    1426     update();
    1427     
    1428     if(autocam)
    1429     {   
    1430         aim(ce);
    1431     }
    1432 }



::

    1448 void Composition::setModel2World_old(const glm::vec4& ce)
    1449 {
    1450     // old way : to be replaced once SCenterExtentFrame is checked
    1451     glm::vec4 ce_(ce.x,ce.y,ce.z,ce.w);
    1452     glm::vec3 sc(ce.w);
    1453     glm::vec3 tr(ce.x, ce.y, ce.z);
    1454     glm::vec3 isc(1.f/ce.w);
    1455 
    1456     m_world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);
    1457     m_model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    1458 
    1459     dump("Composition::setModel2World_old");
    1460 }
    1461 
    1462 void Composition::setModel2World_ce(const glm::vec4& ce, bool rtp_tangential )
    1463 {
    1464     SCenterExtentFrame<double> cef( ce.x, ce.y, ce.z, ce.w, rtp_tangential );
    1465     m_model2world = cef.model2world ;
    1466     m_world2model = cef.world2model ;

    ///  NOTE THIS NARROWS MATRIX ELEMENTS FROM DOUBLE TO FLOAT 

    1467 
    1468     dump("Composition::setModel2World_ce");
    1469 }


    1473 /**
    1474 Composition::setModel2World_qt
    1475 --------------------------------
    1476 
    1477 Invoked by Composition::setCenterExtent when a non-null m2w qat4 is provided.
    1478 
    1479 **/
    1480 
    1481 void Composition::setModel2World_qt(const qat4* m2w, const qat4* w2m )
    1482 {
    1483     //Tran<double>* tvi = Tran<double>::ConvertToTran(m2w); 
    1484     
    1485     assert( m2w != nullptr );
    1486     assert( w2m != nullptr ); 
    1487     m_model2world = glm::make_mat4x4<float>(m2w->cdata());
    1488     m_world2model = glm::make_mat4x4<float>(w2m->cdata());
    1489     
    1490     dump("Composition::setModel2World_qt");
    1491 }   



Hmm Composition::setCenterExtent with externally provided m2w w2m is a recent addition
----------------------------------------------------------------------------------------

Used by CSGOptiX/tests/CSGOptiXSimtraceTest.cc with matrices provided by CSGGenstep::

    094     // create center-extent gensteps 
     95     CSGGenstep* gsm = fd->genstep ;    // THIS IS THE GENSTEP MAKER : NOT THE GS THEMSELVES 
    100 
    101     gsm->create(moi, ce_offset, ce_scale ); // SEvent::MakeCenterExtentGensteps
    ...
    107 
    108     cx.setComposition(gsm->ce, gsm->m2w, gsm->w2m );
    109     cx.setCEGS(gsm->cegs);   // sets peta metadata


::


    016 /**
     17 TODO: adopt SCenterExtentGenstep.hh
     18 **/
     19 

    122 void CSGGenstep::locate(const char* moi_)
    123 {
    124     moi = strdup(moi_) ;
    125 
    126     foundry->parseMOI(midx, mord, iidx, moi );
    127 
    128     LOG(info) << " moi " << moi << " midx " << midx << " mord " << mord << " iidx " << iidx ;
    129     if( midx == -1 )
    130     {
    131         LOG(fatal)
    132             << " failed CSGFoundry::parseMOI for moi [" << moi << "]"
    133             ;
    134         return ;
    135     }
    136 
    137 
    138     int rc = foundry->getCenterExtent(ce, midx, mord, iidx, m2w, w2m ) ;
    139 
    140     LOG(info) << " rc " << rc << " MOI.ce ("
    141               << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;
    142 
    143     LOG(info) << "m2w" << *m2w ;
    144     LOG(info) << "w2m" << *w2m ;
    145 
    146     geotran = Tran<double>::FromPair( m2w, w2m, 1e-6 );    // Tran from stran.h 
    147 
    148     //geotran = Tran<double>::ConvertToTran( qt );    // Tran from stran.h 
    149     // matrix gets inverted by Tran<double>
    150 
    151     //override_locate(); 
    152 }

