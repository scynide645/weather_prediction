from src.app import create_app
import sys
import signal
from src.app.mdns_service import start_mdns

app = create_app()

def shutdown_mDNS(sig, frame):
    print("\n[Shutdown] Mematikan mDNS...")
    if app.config.get("ZEROCONF"):
        zc = app.config["ZEROCONF"]
        zc.unregister_all_services()
        zc.close()
    sys.exit(0)
    
signal.signal(signal.SIGINT, shutdown_mDNS)

if __name__ == '__main__':
    zc = start_mdns(5000)
    app.config["ZEROCONF"] = zc
    
    app.run(host='0.0.0.0', port=5000, debug=True)
