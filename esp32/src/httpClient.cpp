#include <HTTPClient.h>
#include <network.h>
#include <ArduinoJson.h>

void sendData(float temp, float hum){
    HTTPClient http;

//     IPAddress serverIP = MDNS.queryHost(serverHost);

//     if (serverIP.toString() == "0.0.0.0") {
//         Serial.println("pencarian mDNS gagal, tidak bisa kirim data");
//         return;
// }

    String url;

    if(mDNS){
        url = "http://"+ String(serverHost) +':'+ String(serverPort)+"/routes/predict";
    }else {
        Serial.println("mDNS url ke flask tidak terdeteksi");
    }

    Serial.println("sending data to:" + url);
    http.begin(url);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<200> doc;

    doc["temperature"] = temp;
    doc["humidity"] = hum;

    String JsonData;
    serializeJson(doc, JsonData);


    int status = http.POST(JsonData);
    Serial.println("JSON: "+ JsonData);
    Serial.println("Status: " + String(status));

    http.end();

}
