#ifndef LAUNCHSEQUENCE_H
#define LAUNCHSEQUENCE_H

#include <vector>
#include <stdio.h>

struct Launch {
   Launch(unsigned int _thread_offset, 
          unsigned int _threads_per_launch, 
          unsigned int _blocks_per_launch, 
          unsigned int _threads_per_block,
          unsigned int _sequence_index 
         )
          : 
          thread_offset(_thread_offset), 
          threads_per_launch(_threads_per_launch), 
          blocks_per_launch(_blocks_per_launch),
          threads_per_block(_threads_per_block), 
          sequence_index(_sequence_index) 
          {}


   void Summary(const char* msg) const
   {
       printf(" %s sequence_index %3d  thread_offset %6u  threads_per_launch %6u blocks_per_launch %6u   threads_per_block %6u  \n", 
           msg,
           sequence_index,
           thread_offset, 
           threads_per_launch,
           blocks_per_launch,
           threads_per_block );
   } 

   unsigned int thread_offset ; 
   unsigned int threads_per_launch ; 
   unsigned int blocks_per_launch ; 
   unsigned int threads_per_block ; 
   unsigned int sequence_index ; 
}; 




class LaunchSequence {
public:
    LaunchSequence( unsigned int items, unsigned int threads_per_block , unsigned int max_blocks) : 
        m_items(items),
        m_threads_per_block(threads_per_block),
        m_max_blocks(max_blocks) 
    {
        unsigned int thread_offset = 0 ;

        unsigned int sequence_index = 0 ;

        while( thread_offset < m_items )
        {
             unsigned int remaining = m_items - thread_offset ;

             unsigned int blocks_per_launch = remaining / m_threads_per_block ;
             if(remaining % m_threads_per_block != 0) blocks_per_launch += 1 ;  
 
             if( blocks_per_launch > m_max_blocks ) blocks_per_launch = max_blocks ; 

             unsigned int threads_per_launch = blocks_per_launch * m_threads_per_block ; 
             if(threads_per_launch > remaining) threads_per_launch = remaining ;

             m_launches.push_back(Launch(thread_offset, threads_per_launch, blocks_per_launch, m_threads_per_block, sequence_index));

             thread_offset += threads_per_launch ; 
             sequence_index += 1 ;
        }
    }

    unsigned int getNumLaunches(){     return m_launches.size(); }
    unsigned int getItems(){           return m_items ; }
    unsigned int getThreadsPerBlock(){ return m_threads_per_block ; }
    unsigned int getMaxBlocks(){       return m_max_blocks ; }

    const Launch& getLaunch(unsigned int i){ return m_launches[i] ; }

    void Summary(const char* msg)
    {
        unsigned int nlaunch = getNumLaunches();
        printf("%s workitems %7u  threads_per_block %5u  max_blocks %6u nlaunch %3u \n", 
            msg,
            m_items,
            m_threads_per_block,
            m_max_blocks,
            nlaunch ); 

        for(unsigned int i=0 ; i<nlaunch ; i++ )
        {
            const Launch& l = getLaunch(i) ;
            l.Summary(msg);
        } 
    }

    virtual ~LaunchSequence()
    {
    }


private:
    unsigned int m_items ; 
    unsigned int m_threads_per_block ;
    unsigned int m_max_blocks ;

    std::vector<Launch> m_launches ;


};



#endif
