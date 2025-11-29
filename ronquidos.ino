/*
  Nano 33 BLE Sense v2 – Mic PDM @ 16 kHz + debug
  Protocolo a la Pi:
    [0xAA, 0x55] + uint16_le N + N * int16_le
  Comando desde la Pi:
    'B' + uint16_le ms  -> beep ms milisegundos
*/

#include <Arduino.h>
#include <PDM.h>

// ===== Config =====
const uint32_t SAMPLE_RATE    = 16000;
const uint8_t  CHANNELS       = 1;
const uint16_t PACKET_SAMPLES = 256;   // ~16 ms por paquete
const int      PDM_GAIN       = 40;    // Aumentado de 30 a 40 para mejor captación

const int      BUZZER_PIN       = 2;   // CAMBIADO de 9 a 2
const bool     BUZZER_IS_ACTIVE = true; // true: activo ON/OFF ; false: pasivo con tone()

// ===== Estado =====
volatile bool   pdmReady = false;
volatile size_t pdmCount = 0;

// buffers
#define PDM_CHUNK 512               // chunk intermedio desde ISR
int16_t pdmChunk[PDM_CHUNK];
int16_t packetBuf[PACKET_SAMPLES];
uint16_t packIndex = 0;

void onPDMdata() {
  int bytes = PDM.available();
  if (bytes <= 0) return;
  if (bytes > (int)(PDM_CHUNK*sizeof(int16_t))) bytes = PDM_CHUNK*sizeof(int16_t);
  int got = PDM.read(pdmChunk, bytes);
  if (got > 0) {
    pdmCount = (size_t)(got / sizeof(int16_t));
    pdmReady = true;
  }
}

void sendPacket(int16_t* data, uint16_t n) {
  uint8_t hdr[4] = {0xAA, 0x55, (uint8_t)(n & 0xFF), (uint8_t)((n >> 8) & 0xFF)};
  Serial.write(hdr, 4);
  Serial.write((uint8_t*)data, n * sizeof(int16_t));
}

void buzzerBeep(uint16_t ms){
  if (BUZZER_IS_ACTIVE) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(ms);
    digitalWrite(BUZZER_PIN, LOW);
  } else {
    tone(BUZZER_PIN, 4000, ms);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.begin(115200);
  // No esperar Serial - permite funcionar sin USB conectado
  delay(1000);

  // PDM: registrar callback ANTES de begin
  PDM.onReceive(onPDMdata);
  PDM.setGain(PDM_GAIN);
  // Tamaño del buffer interno (en bytes). Aumentarlo ayuda a evitar underruns.
  PDM.setBufferSize(PACKET_SAMPLES * sizeof(int16_t) * 2);

  if (!PDM.begin(CHANNELS, SAMPLE_RATE)) {
    Serial.println("[ERR] PDM.begin() fallo");
    for(;;){ digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(150); }
  }
  Serial.println("[PDM] started");
  buzzerBeep(120);
}

void loop() {
  // 1) Comando de beep desde la Pi
  while (Serial.available() >= 1) {
    int c = Serial.read();
    if (c == 'B') {
      uint16_t ms = 600;
      unsigned long t0 = millis();
      while (Serial.available() < 2 && (millis()-t0) < 20) {}
      if (Serial.available() >= 2) {
        uint8_t lo = Serial.read();
        uint8_t hi = Serial.read();
        ms = (uint16_t)(lo | (hi << 8));
      }
      buzzerBeep(ms);
    }
  }

  // 2) Vaciar chunk PDM a paquetes de envío
  if (pdmReady) {
    noInterrupts();
    size_t n = pdmCount;
    if (n > PDM_CHUNK) n = PDM_CHUNK;
    static int16_t local[PDM_CHUNK];
    memcpy(local, pdmChunk, n * sizeof(int16_t));
    pdmCount = 0;
    pdmReady = false;
    interrupts();

    for (size_t i = 0; i < n; ++i) {
      packetBuf[packIndex++] = local[i];
      if (packIndex >= PACKET_SAMPLES) {
        sendPacket(packetBuf, PACKET_SAMPLES);
        packIndex = 0;
        // debug: LED toggle
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      }
    }
  }
}
