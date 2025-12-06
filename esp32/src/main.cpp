#include <Arduino.h>
#include <pins.h>
#include <network.h>
#include <httpClient.h>


DHT dht(DP, dhtType);


void setup() {
  Serial.begin(115200);
  dht.begin();

  initWifi();
  initmDNS();
  delay(3000);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED){
    Serial.println("Connection Loss... Reconnecting");
    WiFi.reconnect();
    delay(3000);
    return;
  }

  delay(3000);
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();

  sendData(temp, hum);
  Serial.println("temp: "+String(temp)+", hum: "+String(hum));
}