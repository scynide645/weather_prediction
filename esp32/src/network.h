#pragma once
#include <ESPmDNS.h>
#include <WiFi.h>

extern bool mDNS;
extern const char* serverHost;
const uint16_t serverPort = 5000;

void initWifi();
void initmDNS();