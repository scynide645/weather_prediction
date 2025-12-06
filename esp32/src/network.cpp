#include <config.h>
#include <network.h>

bool mDNS = false;
const char* serverHost = "flask-server.local";

void initWifi(){
    WiFi.begin(SSID, PASS);
    
    delay(3000);
    Serial.println("Connecting Wifi...");
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(2000);
        Serial.println("Connecting Wifi ... ");
    }
    Serial.println("Wifi Connected !");
}

void initmDNS(){
    Serial.println("Memulai mDNS...");
    if (!MDNS.begin("esp32-weather")){
        Serial.println("mDNS gagal...");
        mDNS = false;
        return;
    }
    Serial.println("mDNS Berhasil...");
    mDNS = true;
}