#include "zhelpers.h"

int main (int argc, char *argv [])
{
    char* frontend = getenv("FRONTEND");
    s_console("I: %s starting", argv[0]);
    s_console("I: connect REQ socket to frontend [%s]", frontend);

    void* ctx = zmq_ctx_new ();
    void* requester = zmq_socket(ctx, ZMQ_REQ);
    zmq_connect (requester, frontend);

    while (1) {

        char* req = "REQ HELLO ZMQ";
        s_console("send req: %s", req);
        s_send(requester, req );

        char* rep = s_recv(requester);
        s_console("recv rep: %s", rep);
        free(rep);

        s_sleep (1000);  // milliseconds
    }   

    zmq_close( requester );
    zmq_ctx_destroy (ctx);
    return 0;

}
