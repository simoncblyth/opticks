#include "OPTICKS_LOG.hh"
#include "OpticksIdentity.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    
    LOG(info); 
   
    assert( 0xff     == 255 );       // repeat limit 
    assert( 0xffff   == 65535 );     // placement limit 
    assert( 0xff     == 255 );       // offset limit for repeat_index > 0 
    assert( 0xffffff == 16777215 );  // special cased offset limit for repeat_index 0, however constrained to placement_index == 0  

    std::vector<unsigned> placement_counts = { 1           , 0xffff + 1, 0xffff + 1 } ;
    std::vector<unsigned> offset_counts    = { 0xffffff + 1, 0xff   + 1, 0xff   + 1 } ;

    unsigned count(0); 
    for(unsigned r=0 ; r < 3 ; r++)
    {
        for(unsigned p=0 ; p < placement_counts[r] ; p++)
        {
            for(unsigned o=0 ; o < offset_counts[r] ; o++)
            {
                OpticksIdentity id(r,p,o); 

                assert( id.getRepeatIndex() == r ); 
                assert( id.getPlacementIndex() == p ); 
                assert( id.getOffsetIndex() == o ); 

                unsigned identifier = id.getEncodedIdentifier() ; 

                assert( OpticksIdentity::RepeatIndex(identifier) == r );  
                assert( OpticksIdentity::PlacementIndex(identifier) == p );  
                assert( OpticksIdentity::OffsetIndex(identifier) == o );  

                if(count < 100)
                std::cout << "id  " << id.desc() << std::endl ; 

                OpticksIdentity id2(id.getEncodedIdentifier()); 
                //std::cout << "id2 " << id2.desc() << std::endl ; 

                assert( id2.getRepeatIndex() == r ); 
                assert( id2.getPlacementIndex() == p ); 
                assert( id2.getOffsetIndex() == o ); 


                unsigned encoded_identifier = OpticksIdentity::Encode(r,p,o); 

                OpticksIdentity id3(encoded_identifier); 
                assert( id3.getRepeatIndex() == r ); 
                assert( id3.getPlacementIndex() == p ); 
                assert( id3.getOffsetIndex() == o ); 

                count += 1 ;  
            }
        }
    }

    LOG(info) << count ; 

    return 0 ;
}

// om-;TEST=OpticksIdentityTest om-t
