# Detector de Ronquidos con ML

Sistema de detección de ronquidos en tiempo real usando Arduino Nano 33 BLE Sense Rev2 y Raspberry Pi Zero.

## Arquitectura

```
[Arduino Nano 33 BLE Sense Rev2] --USB Serial--> [Raspberry Pi Zero] --Buzzer--> Alerta
        (Micrófono PDM)                           (Inferencia ML)
```

## Componentes

| Archivo | Descripción |
|---------|-------------|
| `ronquidos.ino` | Firmware Arduino - captura audio PDM y envía por USB |
| `snore_detectorv2.py` | Detector en Raspberry Pi - procesa audio y ejecuta modelo ML |
| `train_snore_scipy.py` | Script de entrenamiento (ejecutar en PC) |
| `artifacts/` | Modelos TFLite entrenados |
| `web/` | Dashboard web opcional |

## Instalación

### Arduino
1. Abrir `ronquidos.ino` en Arduino IDE
2. Seleccionar board: **Arduino Nano 33 BLE**
3. Subir el código

### Raspberry Pi Zero

```bash
# Instalar dependencias
sudo apt-get update
sudo apt-get install python3-pip python3-numpy python3-scipy

pip3 install pyserial
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime

# Ejecutar detector
python3 snore_detectorv2.py --model artifacts/snore_model_int8.tflite --serial /dev/ttyACM0
```

## Re-entrenar el modelo

Si quieres entrenar con tus propios datos:

1. Descomprime `data.zip` o crea la estructura:
   ```
   data/
     background/  # sonidos que NO son ronquidos
     snore/       # sonidos de ronquidos
   ```

2. Ejecuta el entrenamiento (en PC con TensorFlow):
   ```bash
   pip install tensorflow numpy scipy scikit-learn
   python train_snore_scipy.py
   ```

3. Copia `artifacts/snore_model_int8.tflite` a la Raspberry Pi

## Hardware

- **Arduino Nano 33 BLE Sense Rev2** - Micrófono PDM integrado
- **Raspberry Pi Zero W** - Inferencia ML
- **Buzzer pasivo** - Pin 2 del Arduino

## Parámetros ajustables

```bash
python3 snore_detectorv2.py --model artifacts/snore_model_int8.tflite \
    --serial /dev/ttyACM0 \
    --threshold 0.75 \      # Umbral de detección (0-1)
    --cooldown 3.0 \        # Segundos entre alertas
    --beep-ms 800           # Duración del beep en ms
```

## Agradecimientos

- GitHub: [@VichoPrime](https://github.com/VichoPrime)
- GitHub: [@naiki919](https://github.com/naiki919)
- GitHub: [@bluarchv](https://github.com/bluarchv)
- GitHub: [@mandimeow](https://github.com/mandimeow)
- GitHub: [@jespinozaopk](https://github.com/jespinozaopk)
