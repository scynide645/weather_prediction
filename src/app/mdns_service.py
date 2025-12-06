import socket
from zeroconf import ServiceInfo, Zeroconf
from zeroconf._exceptions import NonUniqueNameException

def get_ip():
    try:
        sd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sd.connect(("8.8.8.8", 80))
        ip = sd.getsockname()[0]
        sd.close()
        print("memakai ip wifi")
        return ip
    except:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print('Memakai ip device')
        return ip

def start_mdns(port: int):
    ip = get_ip()

    zeroconf = Zeroconf()

    try:
        info = ServiceInfo(
            type_="_http._tcp.local.",
            name="flask-server._http._tcp.local.",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={},
            server="flask-server.local."
        )

        zeroconf.register_service(info)
        print(f"[mDNS] Registered flask-server.local → {ip}:{port}")

    except NonUniqueNameException:
        print("[mDNS] Nama service sudah ada. Skip register mDNS.")

    return zeroconf
