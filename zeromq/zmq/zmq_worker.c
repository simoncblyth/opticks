// Based on zguide example: rrworker.c

#include "zhelpers.h"

int main (int argc, char *argv [])
{
    char* backend = getenv("BACKEND");
    s_console("I: %s starting ", argv[0]);
    s_console("I: connect REP socket to backend [%s] ", backend);

    void* ctx = zmq_ctx_new ();
    void* responder = zmq_socket(ctx, ZMQ_REP);
    zmq_connect (responder, backend);

    while (1) {

        char* req = s_recv(responder);
        s_console("recv req: %s", req);
        free (req);

        s_sleep (1000);  // milliseconds

        char* rep = "hello" ; 
        s_console("send rep: %s", rep);
        s_send (responder, rep); 
    }   

    zmq_close( responder );
    zmq_ctx_destroy (ctx);
    return 0;

}
