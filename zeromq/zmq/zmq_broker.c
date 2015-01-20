// Based on zguide examples: rrbroker.c, msgqueue.c

/*

* http://api.zeromq.org/3-2:zmq-proxy

When the frontend is a ZMQ_ROUTER socket, and the backend is a ZMQ_DEALER
socket, the proxy shall act as a shared queue that collects requests from a set
of clients, and distributes these fairly among a set of services. Requests
shall be fair-queued from frontend connections and distributed evenly across
backend connections. Replies shall automatically return to the client that made
the original request.

If the capture socket is not NULL, the proxy shall send all messages, received
on both frontend and backend, to the capture socket. The capture socket should
be a ZMQ_PUB, ZMQ_DEALER, ZMQ_PUSH, or ZMQ_PAIR socket.

*/

#include "zhelpers.h"

int main (int argc, char *argv [])
{
    char* frontend_ = getenv("FRONTEND");
    char* backend_ = getenv("BACKEND");

    s_console("I: %s start", argv[0] );
    s_console("I: bind frontend ROUTER:[%s]", frontend_ );
    s_console("I: bind backend DEALER:[%s]", backend_ );

    int rc ; 
    void* ctx = zmq_ctx_new ();
 
    void* frontend = zmq_socket(ctx, ZMQ_ROUTER); 
    rc = zmq_bind(frontend, frontend_);
    assert( rc == 0 );

    void* backend  = zmq_socket(ctx, ZMQ_DEALER);  
    rc = zmq_bind(backend,  backend_);
    assert( rc == 0 );

    void* capture = NULL ; 

    s_console("I: enter proxy loop");
    zmq_proxy( frontend, backend, capture );

    // never get here
    zmq_close( frontend );
    zmq_close( backend );
    zmq_ctx_destroy (ctx);
    return 0;
}
