#include <sstream>
#include <iostream>
#include <iomanip>

#include "OGeoStat.hh"


OGeoStat::OGeoStat( unsigned mmIndex_, unsigned numPrim_, unsigned numPart_, unsigned numTran_, unsigned numPlan_ )
       :
       mmIndex(mmIndex_),
       numPrim(numPrim_),
       numPart(numPart_),
       numTran(numTran_),
       numPlan(numPlan_)
{
}

std::string OGeoStat::desc()
{
    std::stringstream ss ; 
    ss << " mmIndex " << std::setw(3) << mmIndex 
       << " numPrim " << std::setw(5) << numPrim 
       << " numPart " << std::setw(5) << numPart
       << " numTran(triples) " << std::setw(5) << numTran
       << " numPlan " << std::setw(5) << numPlan
       ;
    return ss.str(); 
}


