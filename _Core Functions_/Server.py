import socketserver
import json
import Status
from time import gmtime, strftime

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        print(str(self.client_address[0])+":   "+str(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        data = self.request.recv(1024).strip()
        data = data.decode("utf-8")
        response = ""
        try:
            data = json.loads(data)
            if data["ACTION"] == "STATUS":
                number, percentage = Status.read_status("/-KNN Approach-",True,True,False)
                response = self.send_data({"Number":number,"Accuracy":percentage})
        except Exception as e:
            print("Json decode error")
            print(e)
        reply = response+"\r\n"
        self.request.sendall(reply.encode("utf-8"))

    def send_data(self, data: dict):
        data = json.dumps(data)
        return data

if __name__ == "__main__":
    HOST, PORT = "172.31.21.201", 6969
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    server.serve_forever()
