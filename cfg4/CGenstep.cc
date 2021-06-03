#include <iomanip>
#include <sstream>
#include <boost/dynamic_bitset.hpp> 
#include "CGenstep.hh"

CGenstep::CGenstep( unsigned index_ , unsigned photons_, unsigned offset_, char gentype_ )
    :
    index(index_),
    photons(photons_),
    offset(offset_),
    gentype(gentype_),
    mask(new boost::dynamic_bitset<>(photons))
{
}

unsigned CGenstep::getRecordId(unsigned index_, unsigned photon_id) const 
{
    assert( index_ == index ); 
    int record_id = int(photon_id) - offset ; 
    assert( record_id > -1 && record_id < photons );  
    unsigned i = unsigned(record_id) ;   
    return i ; 
}

void CGenstep::markRecordId(unsigned index_, unsigned photon_id)
{
    unsigned i = getRecordId(index_, photon_id); 
    set(i); 
}


CGenstep::~CGenstep()
{
    delete mask ; 
}

void CGenstep::set(unsigned i)
{
    assert( i < photons);
    mask->set(i); 
}

bool CGenstep::all() const 
{
    return mask->all(); 
}
bool CGenstep::any() const 
{
    return mask->any(); 
}
unsigned CGenstep::count() const 
{
    return mask->count(); 
}




std::string CGenstep::desc(const char* msg) const
{
   std::stringstream ss ; 
   if(msg) ss << msg << " " ;  
   ss
       << gentype 
       << " " 
       << " idx" << std::setw(4) << index 
       << " pho" << std::setw(5) << photons 
       << " off " << std::setw(6) << offset 
       << " msk.count " << mask->count()
       << " msk.all " << mask->all()
       ;

   if( mask->size() >= 100 ) ss << std::endl ; 
   ss << " mask " << *mask ;
   std::string s = ss.str(); 
   return s; 
}


