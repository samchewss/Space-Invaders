# Space Invaders con Control Facial (OpenCV)

Juego tipo *Space Invaders* controlado con la **cara** (pos. horizontal) y **parpadeos** para disparar. Renderizado con OpenCV en tiempo real.

> Nota: Esta implementación **no** usa Arcade para dibujar; todo el render se hace con OpenCV. Si deseas migrarlo a Arcade, consulta la sección "Migrar a Arcade".

---

## 🎮 Descripción
- Mueve la nave con el **centro del rostro** detectado por la cámara.
- Dispara al **parpadear** (detección de transición ojos cerrados → abiertos con *cooldown* anti-rebote).
- Enemigos avanzan en enjambre, cambian de dirección al tocar bordes y bajan por filas. Al limpiar una ola, el juego sube ligeramente la dificultad.

**Puntaje**: +10 por enemigo. **Bonus**: +100 al limpiar una ola.  
**Vidas**: 3.

---

## 🕹️ Controles
- **Mover nave**: mueve tu **cara** a izquierda/derecha frente a la cámara. El video se muestra en modo espejo (tu derecha = derecha en pantalla).
- **Disparar**: **parpadea** (ojos cerrados unos frames → abiertos). 
- **Salir**: tecla **Q**.

Parámetros útiles en `game.py`:
- `EYES_CLOSED_FRAMES` (2–4 recomendado): frames consecutivos sin ojos para considerar "cerrados".
- `BLINK_COOLDOWN` (0.25–0.5 s): tiempo mínimo entre disparos para evitar ruido.
- `ENEMY_SPEED` y `ENEMY_MOVE_INTERVAL`: controlan la dificultad base.

---

## 📦 Dependencias
- Python 3.8+ (probado en 3.13)
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [NumPy](https://pypi.org/project/numpy/)

Instalación rápida:
```bash
pip install opencv-python numpy
```

## 📦 Instalación de dependencias

Este proyecto requiere **Python 3.8+** (probado en 3.13) y las siguientes librerías:

```txt
opencv-python>=4.8
numpy>=1.24
```
Instálalas con:

```bash
pip install -r requirements.txt
```
## ▶️ Cómo ejecutar

1. Guarda el archivo del juego como `game.py`.
2. Abre una terminal en la carpeta del proyecto.
3. Ejecuta:

   ```bash
   python game.py
    ```

## 🧭 Diagrama simple del flujo

```mermaid
graph LR
  A[Cámara]
  A --> B[OpenCV: captura y espejo]
  B --> C[Detección de rostro (Haar)]
  C --> D[Mapeo X rostro → X nave]
  C --> E[ROI rostro → Detección de ojos]
  E --> F[FSM parpadeo → disparar]
  D --> G[Estado del juego]
  F --> G
  G --> H[Actualización: balas, enjambre, colisiones]
  H --> I[Render en ventana OpenCV]
```
Si prefieres Arcade para el render, el bloque I se reemplaza por "Dibujo en motor de juego (Arcade)" y el bucle principal debería integrarse al event loop de Arcade, manteniendo exactamente el mismo pipeline CV (A→F).

## 📜 Créditos de assets

- **Clasificadores Haar**: incluidos en la distribución de OpenCV (`cv2.data.haarcascades`).
  - `haarcascade_frontalface_default.xml`
  - `haarcascade_eye_tree_eyeglasses.xml` (o `haarcascade_eye.xml`)
- **Gráficos**: todas las figuras se generan con primitivas de OpenCV (no se usan sprites externos).

---

## 🧩 Migrar a Arcade (opcional)

- Mantén el módulo de **CV** igual (detección de rostro y parpadeo).
- Expón `ship_x` y el evento **disparo** como entradas para tu escena en Arcade.
- Reemplaza el render y la gestión del *loop* por las funciones de Arcade (`on_draw`, `on_update`, etc.).

---

## 🚑 Solución de problemas

- **No detecta parpadeos**: aumenta `EYES_CLOSED_FRAMES` a 3–4; mejora iluminación frontal; prueba `haarcascade_eye.xml`.
- **Demasiados disparos**: sube `BLINK_COOLDOWN` a 0.45–0.6 s.
- **Lag**: reduce resolución de la cámara o limita el tamaño de la ROI; baja `ENEMY_COLS`/`ROWS`.
