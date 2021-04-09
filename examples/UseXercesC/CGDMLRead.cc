#include "CGDMLRead.hh"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "CGDMLErrorHandler.hh"



double atod_( const char* a ) 
{
    std::string s(a);
    std::istringstream iss(s);
    double d ; 
    iss >> d ; 
    return d ; 
}

std::string Transcode(const XMLCh* const toTranscode)
{
   char* char_str = xercesc::XMLString::transcode(toTranscode);
   std::string my_str(char_str);
   xercesc::XMLString::release(&char_str);
   return my_str;
}


CGDMLRead::CGDMLRead( const char* path, bool kludge_truncated_matrix_)
    :
    validate(false),
    kludge_truncated_matrix(kludge_truncated_matrix_),
    handler(new CGDMLErrorHandler(!validate)),
    parser(new xercesc::XercesDOMParser),
    doc(nullptr),
    element(nullptr)
{
    std::cout << "reading " << path << std::endl ; 

    if (validate)
    {   
        parser->setValidationScheme(xercesc::XercesDOMParser::Val_Always);
    }   
    parser->setValidationSchemaFullChecking(validate);
    parser->setCreateEntityReferenceNodes(false); 
     // Entities will be automatically resolved by Xerces

    parser->setDoNamespaces(true);
    parser->setDoSchema(validate);
    parser->setErrorHandler(handler);

    try 
    { 
        parser->parse(path); 
    }
    catch (const xercesc::XMLException &e) 
    { 
        std::cout << "GDML: " << Transcode(e.getMessage()) << std::endl; 
    }
    catch (const xercesc::DOMException &e) 
    { 
        std::cout << "G4GDML: " << Transcode(e.getMessage()) << std::endl; 
    }

    doc = parser->getDocument();

    if (!doc)
    {   
       std::cout << "Unable to open document " << path  << std::endl ; 
       return ;
    }   

    element = doc->getDocumentElement();
    std::cout << "documenElement " << element << std::endl ; 
    assert( element); 

    for(xercesc::DOMNode* iter = element->getFirstChild(); iter != 0; iter = iter->getNextSibling())
    {
        if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE)  { continue; }

        const xercesc::DOMElement* const child = dynamic_cast<xercesc::DOMElement*>(iter); 

        if (!child)
        {
            std::cout << "InvalidRead : No child " << std::endl ; 
            return ;
        }

        const std::string tag = Transcode(child->getTagName());

        if (tag=="define")
        { 
            DefineRead(child);    
        }  
        else
        {
            std::cout << " tag " << tag << std::endl ; 
        }

        /*

          if (tag=="define")    { DefineRead(child);    } else
          if (tag=="materials") { MaterialsRead(child); } else
          if (tag=="solids")    { SolidsRead(child);    } else
          if (tag=="setup")     { SetupRead(child);     } else
          if (tag=="structure") { StructureRead(child); } else
          if (tag=="userinfo")  { UserinfoRead(child);  } else
          if (tag=="extension") { ExtensionRead(child); }
          else
          {
            G4String error_msg = "Unknown tag in gdml: " + tag;
            G4Exception("G4GDMLRead::Read()", "InvalidRead",
                        FatalException, error_msg);
          }
       */

       }
}


void CGDMLRead::MatrixRead( const xercesc::DOMElement* const matrixElement, bool& truncated_values )
{
    std::string name = ""; 
    int coldim  = 0;
    std::string values = ""; 

    const xercesc::DOMNamedNodeMap* const attributes = matrixElement->getAttributes();
    XMLSize_t attributeCount = attributes->getLength();

    for (XMLSize_t attribute_index=0; attribute_index<attributeCount; attribute_index++)
    {   
        xercesc::DOMNode* node = attributes->item(attribute_index);

        if (node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE) { continue; }

        const xercesc::DOMAttr* const attribute = dynamic_cast<xercesc::DOMAttr*>(node); 
  
        if (!attribute)
        {   
            std::cout << "MatrixRead : InvalidRead : No attribute found" << std::endl;  
            return;
        }   
        const std::string attName = Transcode(attribute->getName());
        const std::string attValue = Transcode(attribute->getValue());

        if (attName=="name")   { name  = attValue ; } else
        //if (attName=="name")   { name  = GenerateName(attValue); } else
        //if (attName=="coldim") { coldim = eval.EvaluateInteger(attValue); } else
        if (attName=="values") { values = attValue; }
    }   

    std::vector<double> valueList;
    std::stringstream ss; 
    ss.str(values.c_str())  ;
    char delim = ' ' ; 
    std::string s;
    while (std::getline(ss, s, delim)) valueList.push_back(atod_(s.c_str())) ; 


   truncated_values = false ; 
   if(values.length() >= 9999 || valueList.size() % 2 != 0) truncated_values = true ; 
   if(truncated_values) 
   {
       xercesc::DOMElement* me = const_cast<xercesc::DOMElement*>(matrixElement);  
       truncated_matrixElement.push_back(me); 
       if(kludge_truncated_matrix) KludgeTruncatedMatrix(me); 
   }

    bool dump = false ; 
    if(dump)
    std::cout 
        << "CGDMLRead::MatrixRead"   
        << " " << ( truncated_values ? "**" : "  " )
        << " values.lenth " << std::setw(7) << values.size() 
        << " last50 " << std::setw(50) << values.substr(std::max(0,int(values.length())-50)) 
        << " valueList.size " << std::setw(10) << valueList.size()
        << " " << ( truncated_values ? "**" : "  " )
        << " name " << name 
        << std::endl 
        ; 
}

std::string CGDMLRead::KludgeFix( const char* values )
{
    std::stringstream ss; 
    ss.str(values)  ;

    std::vector<std::string> elem ; 
    char delim = ' ' ; 
    std::string s ; 
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 

    unsigned num_trim = elem.size() % 2 == 1 ? 1 : 2 ; 
    unsigned i0 = 0 ; 
    unsigned i1 = elem.size() - num_trim ; 

    std::stringstream kk ; 
    for(unsigned i=i0 ; i < i1 ; i++)
    { 
        std::string k = elem[i]; 
        kk << k ;
        if(i < i1 - 1 ) kk << delim  ; 
    }

    std::string kludged = kk.str();         
    return kludged ;   
}

void CGDMLRead::KludgeTruncatedMatrix(xercesc::DOMElement* matrixElement )
{
    xercesc::DOMNamedNodeMap* attributes = matrixElement->getAttributes();
    XMLSize_t attributeCount = attributes->getLength();

    for (XMLSize_t attribute_index=0; attribute_index<attributeCount; attribute_index++)
    {   
        xercesc::DOMNode* node = attributes->item(attribute_index);
        if (node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE) { continue; }
        xercesc::DOMAttr* attribute = dynamic_cast<xercesc::DOMAttr*>(node); 
        const std::string attName = Transcode(attribute->getName());

        if( attName == "values" )
        {
            const std::string attValueOri = Transcode(attribute->getValue());
            std::string attValueKlu = KludgeFix(attValueOri.c_str());             
            std::cout 
                << "CGDMLRead::KludgeTruncatedMatrix" 
                << " attName " << attName 
                << " attValueOri.length " << attValueOri.length() 
                << " attValueKlu.length " << attValueKlu.length() 
                << std::endl
                 ; 

            std::cout 
                << "CGDMLRead::KludgeTruncatedMatrix" << std::endl
                << " attValueOri.length " << attValueOri.length() << std::endl
                << " attValueKlu.length " << attValueKlu.length() << std::endl 
                << " attValueOri.last50 " << std::setw(50) << attValueOri.substr(std::max(0,int(attValueOri.length())-50)) << std::endl 
                << " attValueKlu.last50 " << std::setw(50) << attValueKlu.substr(std::max(0,int(attValueKlu.length())-50)) << std::endl
                ; 

            xercesc::XMLString::transcode(attValueKlu.c_str() , tempStr, 9999);
            attribute->setValue(tempStr); 
        }
    }
}

Constant CGDMLRead::ConstantRead( const xercesc::DOMElement* const constantElement )
{
    Constant c = {} ;
    c.name = "" ; 
    c.value = 0.0 ; 
    c.constantElement = const_cast<xercesc::DOMElement*>(constantElement) ; 

    const xercesc::DOMNamedNodeMap* const attributes = constantElement->getAttributes(); 
    XMLSize_t attributeCount = attributes->getLength();

    for (XMLSize_t attribute_index=0; attribute_index<attributeCount; attribute_index++) 
    {   
        xercesc::DOMNode* node = attributes->item(attribute_index);
        if (node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE) { continue; }
        const xercesc::DOMAttr* const attribute = dynamic_cast<xercesc::DOMAttr*>(node);   
        assert(attribute);
        const std::string attName = Transcode(attribute->getName());
        const std::string attValue = Transcode(attribute->getValue());
        if (attName=="name")  { c.name = attValue; }  else
        if (attName=="value") { c.value = atod_(attValue.c_str()); }
   }   
    return c ; 
}

void CGDMLRead::DefineRead( const xercesc::DOMElement* const defineElement )
{
    assert( the_defineElement == nullptr ); 
    the_defineElement = const_cast<xercesc::DOMElement*>(defineElement) ; 

    std::cout << "CGDMLRead::DefineRead" << std::endl ;  

    xercesc::DOMElement* modifiableDefineElement = const_cast<xercesc::DOMElement*>(defineElement); 


   for (xercesc::DOMNode* iter = defineElement->getFirstChild(); iter != 0; iter = iter->getNextSibling())
   {   
      if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE) { continue; }

      const xercesc::DOMElement* const child = dynamic_cast<xercesc::DOMElement*>(iter);
      assert( child ); 
      const std::string tag = Transcode(child->getTagName());

      bool truncated_matrix_values = false ; 

      if (tag=="constant")
      { 
          Constant c = ConstantRead(child); 
          constants.push_back(c); 
      } 
      else if (tag=="matrix")     
      { 
          MatrixRead(child, truncated_matrix_values); 
      }  
      else 
/*
      if (tag=="position")   { PositionRead(child); } else 
      if (tag=="rotation")   { RotationRead(child); } else 
      if (tag=="scale")      { ScaleRead(child); } else 
      if (tag=="variable")   { VariableRead(child); } else 
      if (tag=="quantity")   { QuantityRead(child); } else 
      if (tag=="expression") { ExpressionRead(child); } else
*/
      {   
          std::cout << "Unknown tag in define " << tag << std::endl ; 
      }   
   }   


   std::cout << "CGDMLRead::DefineRead constants.size " << constants.size() << std::endl ; 
   for(unsigned i=0 ; i < constants.size() ; i++)
   {
       const Constant& c = constants[i] ;
       std::cout 
           << " c.name " << std::setw(20) << c.name 
           << " c.value " << std::setw(10) << c.value 
           << " c.constantElement " << c.constantElement 
           << std::endl 
           ;  
       modifiableDefineElement->removeChild(c.constantElement); 
   }


}





CGDMLRead::~CGDMLRead()
{
    delete handler ; 
    delete parser ; 
}



