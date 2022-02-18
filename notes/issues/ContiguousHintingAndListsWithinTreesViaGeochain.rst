ContiguousHintingAndListsWithinTreesViaGeochain
===================================================


::

     812 /**
     813 CSGMaker::makeBooleanListList
     814 -------------------------------
     815 
     816 This generalizes from CSGMaker::makeUnionListBoxSphere in order to test in a more flexible way.
     817 
     818 TODO: need to apply the experience from here to the GeoChain conversions with CSG_CONTIGUOUS hinting, 
     819 should the subOffset be set at NCSG level ?
     820 
     821 **/
     822 
     823 CSGSolid* CSGMaker::makeBooleanListList( const char* label,
     824        unsigned op_,
     825        unsigned ltype,
     826        unsigned rtype,
     827        std::vector<CSGNode>& lhs,
     828        std::vector<CSGNode>& rhs,
     829        const std::vector<const Tran<double>*>* ltran,
     830        const std::vector<const Tran<double>*>* rtran
     831     )
     832 {



GeoChain
-----------


1. X4SolidMaker::AltXJfixtureConstruction creates the G4VSolid with "CSG_CONTIGUOUS" string hint 
   within the name of a G4UnionSolid

2. X4Solid::convertUnionSolid notices the hinting  and changes the ordinary boolean tree nnode root
   for that hinted node into a nmultiunion collecting leaf nodes and transforms

   * so we have a smaller boolean tree with an nmultiunion list node 

::

     269 void X4Solid::convertUnionSolid()
     270 {
     271     convertBooleanSolid() ;
     272 
     273     bool contiguous = hasHintContiguous();
     274     if(contiguous)
     275     {
     276         changeToContiguousSolid() ;
     277     }
     278 }

     383 /**
     384 X4Solid::changeToContiguousSolid
     385 ---------------------------------
     386 
     387 Hmm need to collect all leaves of the subtree rooted here into a
     388 compound like the above multiunion  
     389 
     390 Need to apply the X4Solid conversion to the leaves only
     391 and just collect flattened transforms from the operator nodes above them  
     392 
     393 Hmm probably simplest to apply the normal convertBooleanSolid and 
     394 then replace the nnode subtree. Because thats using the nnode 
     395 lingo should do thing within nmultiunion
     396 
     397 Just need to collect the list of nodes. Hmm maybe flatten transforms ?
     398 
     399 **/
     400 
     401 void X4Solid::changeToContiguousSolid()
     402 {
     403     LOG(LEVEL) << "[" ;
     404     nnode* subtree = getRoot();
     405     nmultiunion* root = nmultiunion::CreateFromTree(CSG_CONTIGUOUS, subtree) ;
     406     setRoot(root);
     407     LOG(LEVEL) << "]" ;
     408 }



3. 

