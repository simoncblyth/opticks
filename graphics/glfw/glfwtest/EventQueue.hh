#ifndef EVENTQUEUE_H
#define EVENTQUEUE_H

#include <deque>
#include <iostream>
#include <boost/array.hpp>

#define EVENTQUEUE_ITEMSIZE 16
typedef boost::array<char, EVENTQUEUE_ITEMSIZE> EventQueueItem_t ; 

class EventQueue {
public:     
    EventQueue();

    void push(EventQueueItem_t item);
    EventQueueItem_t pop();
    bool isEmpty();

private:    
    std::deque<EventQueueItem_t> m_queue;
};


#endif 

