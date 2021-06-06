#include <iomanip>
#include <sstream>
#include <boost/dynamic_bitset.hpp> 
#include "CGenstep.hh"
#include "OpticksGenstep.hh"

CGenstep::CGenstep()
    :
    index(0),
    photons(0),
    offset(0),
    gentype('?'),
    mask(nullptr)
{
}

CGenstep::CGenstep( unsigned index_ , unsigned photons_, unsigned offset_, char gentype_ )
    :
    index(index_),
    photons(photons_),
    offset(offset_),
    gentype(gentype_),
    mask(nullptr)
{
}


unsigned CGenstep::getGenflag() const
{
    return OpticksGenstep::GentypeToPhotonFlag(gentype); 
}



CGenstep::~CGenstep()
{
    delete mask ; 
}

void CGenstep::set(unsigned i)
{
    assert( i < photons);
    if( mask == nullptr ) mask = new boost::dynamic_bitset<>(photons) ; 
    mask->set(i); 
}

bool CGenstep::all() const 
{
    return mask ? mask->all() : false ; 
}
bool CGenstep::any() const 
{
    return mask ? mask->any() : false ; 
}
unsigned CGenstep::count() const 
{
    return mask ? mask->count() : 0 ; 
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
       ;

   if(mask)
   {
      ss 
          << " msk.count " << mask->count()
          << " msk.all " << mask->all()
          ;

      if( mask->size() >= 100 ) ss << std::endl ; 
      ss << " mask " << *mask ;
   }

   std::string s = ss.str(); 
   return s; 
}


