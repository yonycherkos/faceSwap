start

    $ pip install grpcio
    $ pip install grpcio-tools 


generate proto files 

    $ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. image_swap.proto



start the server

    $ python server.py
    

start client 

    $ python client.py
