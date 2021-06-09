G4Opticks_why_sudden_switch_to_pro_embedded_commandline
========================================================


* presumably something formerly OPTICKS_EMBEDDED_COMMANDLINE was defined and it is no more, 


::

     083 const char* G4Opticks::OPTICKS_EMBEDDED_COMMANDLINE = "OPTICKS_EMBEDDED_COMMANDLINE" ;
      84 const char* G4Opticks::OPTICKS_EMBEDDED_COMMANDLINE_EXTRA = "OPTICKS_EMBEDDED_COMMANDLINE_EXTRA" ;
      85 const char* G4Opticks::fEmbeddedCommandLine_pro = " --compute --embedded --xanalytic --production --nosave" ;
      86 const char* G4Opticks::fEmbeddedCommandLine_dev = " --compute --embedded --xanalytic --save --natural --printenabled --pindex 0" ;
      87 
      88 /**
      89 G4Opticks::EmbeddedCommandLine
      90 --------------------------------
      91 
      92 When the OPTICKS_EMBEDDED_COMMANDLINE envvar is not defined the default value of "pro" 
      93 is used. If the envvar OPTICKS_EMBEDDED_COMMANDLINE is defined with 
      94 special values of "dev" or "pro" then the corresponding static 
      95 variable default commandlines are used for the embedded Opticks commandline.
      96 Other values of the envvar are passed asis to the Opticks instanciation.
      97 
      98 Calls to G4Opticks::setEmbeddedCommandLineExtra made prior to 
      99 Opticks instanciation in G4Opticks::setGeometry will append to the 
     100 embedded commandline setup via envvar or default. Caution that duplication 
     101 of options or invalid combinations of options will cause asserts.
     102 
     103 **/
     104 
     105 std::string G4Opticks::EmbeddedCommandLine(const char* extra1, const char* extra2 )  // static
     106 {
     107     const char* ecl  = SSys::getenvvar(OPTICKS_EMBEDDED_COMMANDLINE, "pro") ;
     108 
     109     char mode = '?' ;
     110     const char* explanation = "" ;
     111     if(strcmp(ecl, "pro") == 0)
     112     {
     113         ecl = fEmbeddedCommandLine_pro ;
     114         mode = 'P' ;
     115         explanation = "using \"pro\" (production) commandline without event saving " ;
     116     }
     117     else if(strcmp(ecl, "dev") == 0)
     118     {
     119         ecl = fEmbeddedCommandLine_dev ;
     120         mode = 'D' ;
     121         explanation = "using \"dev\" (development) commandline with full event saving " ;
     122     }
     123     else
     124     {
     125         mode = 'A' ;
     126         explanation = "using custom commandline (for experts only) " ;
     127     }
     128 
     129     const char* eclx = SSys::getenvvar(OPTICKS_EMBEDDED_COMMANDLINE_EXTRA, "") ;

