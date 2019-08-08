import sys
import time

sys.path.insert(0, 'Service/')

from server import *

server = Server()
server.start_server()

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop_server()
