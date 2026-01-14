
#include <cstring>
#include <csignal>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "ssys.h"
#include "OpticksCSG.h"


struct OpticksCSGTest
{
    static int Main();

    static int TypeCodeVec();
    static int Type();
    static int TypeMask();

    static int HintCode(const char* name);
    static int HintCode();
    static int OffsetType();
    static int MaskString();

    static constexpr const char* HINTCODE_NAMES = R"LITERAL(
Hello_CSG_CONTIGUOUS
Hello_CSG_DISCONTIGUOUS
Hello_CSG_OVERLAP
Name_without_any_hint
)LITERAL";

};


int OpticksCSGTest::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST,"ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"TypeCodeVec")) rc += TypeCodeVec();
    if(ALL||0==strcmp(TEST,"Type")) rc += Type();
    if(ALL||0==strcmp(TEST,"TypeMask")) rc += TypeMask();
    if(ALL||0==strcmp(TEST,"HintCode")) rc += HintCode();
    if(ALL||0==strcmp(TEST,"OffsetType")) rc += OffsetType();
    if(ALL||0==strcmp(TEST,"MaskString")) rc += MaskString();

    return rc ;
}


int OpticksCSGTest::TypeCodeVec()
{
    const char* names = "torus,notsupported,cutcylinder,phicut,halfspace" ;
    std::vector<int> typecode ;
    CSG::TypeCodeVec(typecode, names, ',');

    size_t sz = typecode.size() ;

    assert( sz == 5 );
    std::cout
       << "OpticksCSGTest::TypeCodeVec"
       << " names [" << names << "]"
       << " sz " << sz
       << "\n"
       ;

    for(size_t i=0 ; i < sz ; i++ ) std::cout
        << " i "  << std::setw(4) << i
        << " tc " << std::setw(4) << typecode[i]
        << " CSG::Name(tc)  " << CSG::Name(typecode[i])
        << "\n"
        ;

    return 0;
}



int OpticksCSGTest::Type()
{
    for(unsigned i=0 ; i < 100 ; i++)
    {
        OpticksCSG_t type = (OpticksCSG_t)i ;
        if(!CSG::Exists(type)) continue ;

        const char*  name = CSG::Name( type );

        std::cout
                   << " type " << std::setw(3) << type
                   << " name " << std::setw(20) << name
                   << std::endl ;


    }
    return 0 ;
}

int OpticksCSGTest::TypeMask()
{

    // UID
    // 000  ___
    // 001  __D
    // 010  _I_
    // 011  _ID
    // 100  U__
    // 101  U_D
    // 110  UI_
    // 111  UID

    std::vector<unsigned> masks = {{
         0u,
         CSG::Mask(CSG_DIFFERENCE),
         CSG::Mask(CSG_INTERSECTION),
         CSG::Mask(CSG_INTERSECTION) | CSG::Mask(CSG_DIFFERENCE),
         CSG::Mask(CSG_UNION),
         CSG::Mask(CSG_UNION) | CSG::Mask(CSG_DIFFERENCE),
         CSG::Mask(CSG_UNION) | CSG::Mask(CSG_INTERSECTION),
         CSG::Mask(CSG_UNION) | CSG::Mask(CSG_INTERSECTION) | CSG::Mask(CSG_DIFFERENCE)
    }};

    for(unsigned i=0 ; i < masks.size() ; i++)
    {
        unsigned mask = masks[i] ;
        std::cout
            << " i " << std::setw(5) << i
            << " mask " << std::setw(5) << mask
            << " CSG::TypeMask(mask) " << std::setw(10) << CSG::TypeMask(mask)
            << " CSG::IsPositiveMask(mask) " << std::setw(2) << CSG::IsPositiveMask(mask)
            << std::endl
            ;
    }
    return 0 ;
}


int OpticksCSGTest::HintCode(const char* name)
{
     unsigned hintcode = CSG::HintCode(name);
     std::cout
         << " name " << std::setw(40) << name
         << " hintcode " << std::setw(6) << hintcode
         << " CSG::Name(hintcode) " << std::setw(15) << CSG::Name(hintcode)
         << std::endl
         ;
     return 0 ;
}

int OpticksCSGTest::HintCode()
{
    int rc = 0 ;
    std::stringstream ss(HINTCODE_NAMES) ;
    std::string name ;
    while (std::getline(ss, name)) if(!name.empty()) rc += HintCode(name.c_str());
    return rc ;
}


int OpticksCSGTest::OffsetType()
{
    std::vector<unsigned> types = {
            CSG_ZERO,
            CSG_TREE,
                CSG_UNION,
                CSG_INTERSECTION,
                CSG_DIFFERENCE,
            CSG_LIST,
                CSG_CONTIGUOUS,
                CSG_DISCONTIGUOUS,
                CSG_OVERLAP,
            CSG_LEAF,
                CSG_SPHERE,
                CSG_BOX,
                CSG_ZSPHERE,
                CSG_TUBS,
                CSG_CYLINDER,
                CSG_SLAB,
                CSG_PLANE,
                CSG_CONE,
                CSG_BOX3,
                CSG_TRAPEZOID,
                CSG_CONVEXPOLYHEDRON,
                CSG_DISC,
                CSG_SEGMENT,
                CSG_ELLIPSOID,
                CSG_TORUS,
                CSG_HYPERBOLOID,
                CSG_CUBIC,
                CSG_INFCYLINDER,
                CSG_PHICUT,
                CSG_THETACUT,
                CSG_UNDEFINED
       };

    for(unsigned i=0 ; i < types.size() ; i++)
    {
         OpticksCSG_t type = (OpticksCSG_t)types[i];
         const char* name = CSG::Name(type) ;
         int type2 = CSG::TypeCode(name);
         const char* name2 = CSG::Name(type2) ;

         unsigned offset_type = CSG::OffsetType(type);
         unsigned type3 = CSG::TypeFromOffsetType( offset_type );

         std::cout
              << " i " << std::setw(3) << i
              << " type " << std::setw(3) << type
              << " offset_type " << std::setw(3) << offset_type
              << " CSG::Tag(type) " << std::setw(10) << CSG::Tag(type)
              << " CSG::Name(type) " << std::setw(15) << name
              << " CSG::Name(type2) " << std::setw(15) << name2
              << " CSG::IsPrimitive(type) " << std::setw(2) << CSG::IsPrimitive(type)
              << " CSG::IsList(type) " << std::setw(2) << CSG::IsList(type)
              << " CSG::IsCompound(type) " << std::setw(2) << CSG::IsCompound(type)
              << " CSG::IsLeaf(type) " << std::setw(2) << CSG::IsLeaf(type)
              << std::endl
              ;

         bool type2_expect = type2 == type ;
         bool type3_expect = type3 == type ;

         assert( type2_expect );
         assert( type3_expect );

         if(!type2_expect) std::raise(SIGINT);
         if(!type3_expect) std::raise(SIGINT);
    }
    return 0 ;
}

int OpticksCSGTest::MaskString()
{
    unsigned mask = CSG::Mask(CSG_SPHERE) | CSG::Mask(CSG_UNION)  ;
    std::cout << CSG::MaskString(mask) << std::endl ;

    unsigned typemask = 0 ;

    for(unsigned i=0 ; i < 32 ; i++)
    {
        OpticksCSG_t type = (OpticksCSG_t)CSG::TypeFromOffsetType(i) ;
        typemask |= CSG::Mask(type);

        const char* name =  CSG::Name(type)  ;
        if( name == nullptr ) continue ;

        std::cout
            << " i " << std::setw(3) << i
            << " type " << std::setw(3) << type
            << " CSG::Name(type) " << std::setw(20) << CSG::Name(type)
            << " typemask " << std::setw(10) << typemask
            << " CSG::MaskString(typemask) " << CSG::MaskString(typemask)
            << std::endl
            ;

    }
    return 0 ;
}

int main(){ return OpticksCSGTest::Main(); }

