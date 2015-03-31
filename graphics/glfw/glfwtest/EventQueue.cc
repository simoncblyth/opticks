#include "EventQueue.hh"

EventQueue::EventQueue() {
    m_queue.clear();
}

void EventQueue::push(EventQueueItem_t item ) 
{
    m_queue.push_front(item);
}   

bool EventQueue::isEmpty() 
{
    return m_queue.empty();
}

EventQueueItem_t EventQueue::pop() 
{
    EventQueueItem_t temp;
    temp = m_queue.front();
    m_queue.pop_front();
    return temp;
}


