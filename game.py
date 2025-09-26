import cv2
import numpy as np
import random
import time

# --- Configuración del juego ---
MAX_LIVES = 3
SHIP_Y_OFFSET = 80         # distancia desde la parte inferior
SHIP_WIDTH = 80
SHIP_HEIGHT = 30

BULLET_SPEED = 12
BULLET_RADIUS = 6
BLINK_COOLDOWN = 0.35      # seg. mínimo entre disparos
EYES_CLOSED_FRAMES = 2     # # frames consecutivos sin ojos para considerar "cerrados"
EYES_OPEN_MAX_GAP = 10     # # frames máx. para considerar un parpadeo breve

ENEMY_COLS = 8
ENEMY_ROWS = 3
ENEMY_W = 50
ENEMY_H = 30
ENEMY_H_GAP = 20
ENEMY_V_GAP = 26
ENEMY_SPEED = 5            # píxeles por actualización horizontal
ENEMY_STEP_DOWN = 18
ENEMY_MOVE_INTERVAL = 0.06 # segundos entre pasos del enjambre

# Cargar clasificadores Haar
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# --- Estado del juego ---
lives = MAX_LIVES
score = 0

# Estado de la nave y disparos
ship_x = 0
bullets = []  # cada bala: {"x": int, "y": int}

# Estado del enjambre
enemies = []  # cada enemigo: {"x": int, "y": int, "alive": bool}
swarm_dir = 1  # 1 derecha, -1 izquierda
last_swarm_move = time.time()

def init_enemies(frame_w):
    global enemies, swarm_dir, last_swarm_move
    enemies = []
    swarm_dir = 1
    last_swarm_move = time.time()
    # centrar enjambre
    total_w = ENEMY_COLS * ENEMY_W + (ENEMY_COLS - 1) * ENEMY_H_GAP
    start_x = max(10, (frame_w - total_w) // 2)
    start_y = 60
    for r in range(ENEMY_ROWS):
        for c in range(ENEMY_COLS):
            ex = start_x + c * (ENEMY_W + ENEMY_H_GAP)
            ey = start_y + r * (ENEMY_H + ENEMY_V_GAP)
            enemies.append({"x": ex, "y": ey, "alive": True})

def enemies_bounds():
    xs = [e["x"] for e in enemies if e["alive"]]
    xe = [e["x"] + ENEMY_W for e in enemies if e["alive"]]
    if not xs:
        return (0,0)
    return (min(xs), max(xe))

# Parpadeo: simple FSM con ojos abiertos/cerrados
last_shot_time = 0.0
eyes_closed_count = 0
eyes_open_count = 0
was_closed = False
recent_face = False
last_face_seen = 0.0

# Inicializar enjambre tras conocer tamaño del frame
ret, init_frame = cap.read()
if not ret:
    print("No se pudo leer el primer frame")
    cap.release()
    cv2.destroyAllWindows()
    exit()
frame_h, frame_w = init_frame.shape[:2]
init_enemies(frame_w)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Efecto espejo
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Detección de rostro ---
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    # Dibujar rostro y obtener caja principal
    face_boxes = []
    main_face = None
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_boxes.append((x, y, x + w, y + h))
    if len(face_boxes) > 0:
        # tomar el rostro más grande (probable usuario)
        main_face = max(face_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        recent_face = True
        last_face_seen = time.time()
    else:
        # si hace rato no vemos rostro, evitar disparos fantasma
        recent_face = (time.time() - last_face_seen) < 1.0

    # --- Control de nave por posición de rostro ---
    if main_face:
        fx1, fy1, fx2, fy2 = main_face
        face_cx = (fx1 + fx2) // 2
        ship_x = int(np.clip(face_cx - SHIP_WIDTH // 2, 0, frame_w - SHIP_WIDTH))

    # --- Detección de ojos y parpadeo ---
    eyes_detected = False
    if main_face:
        fx1, fy1, fx2, fy2 = main_face
        roi_gray = gray[fy1:fy2, fx1:fx2]
        scale = 0.7
        if roi_gray.size > 0:
            roi_small = cv2.resize(roi_gray, (0,0), fx=scale, fy=scale)
            eyes = eye_cascade.detectMultiScale(roi_small, 1.1, 5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes[:2]:
                gx1 = fx1 + int(ex/scale)
                gy1 = fy1 + int(ey/scale)
                gx2 = gx1 + int(ew/scale)
                gy2 = gy1 + int(eh/scale)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 1)
            eyes_detected = len(eyes) >= 1

    if recent_face:
        if eyes_detected:
            eyes_open_count += 1
            # transición cerrado->abierto (parpadeo breve) con cooldown
            if was_closed and eyes_open_count <= EYES_OPEN_MAX_GAP:
                now = time.time()
                if (now - last_shot_time) > BLINK_COOLDOWN:
                    bx = ship_x + SHIP_WIDTH // 2
                    by = frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT // 2
                    bullets.append({"x": bx, "y": by})
                    last_shot_time = now
            was_closed = False
            eyes_closed_count = 0
        else:
            eyes_closed_count += 1
            eyes_open_count = 0
            if eyes_closed_count >= EYES_CLOSED_FRAMES:
                was_closed = True
    else:
        # sin rostro reciente, reiniciar contador
        eyes_closed_count = 0
        eyes_open_count = 0
        was_closed = False

    # --- Mover y dibujar balas ---
    new_bullets = []
    for b in bullets:
        b["y"] -= BULLET_SPEED
        if b["y"] > 0:
            new_bullets.append(b)
        cv2.circle(frame, (int(b["x"]), int(b["y"])), BULLET_RADIUS, (0, 255, 255), -1)
    bullets = new_bullets

    # --- Subir dificultad y reiniciar ola cuando no queden enemigos ---
    if not any(e["alive"] for e in enemies):
        score += 100  # bonus por limpiar ola
        ENEMY_SPEED = min(ENEMY_SPEED + 1, 12)
        ENEMY_MOVE_INTERVAL = max(ENEMY_MOVE_INTERVAL * 0.92, 0.02)
        init_enemies(frame_w)

    # --- Mover enjambre de enemigos (por intervalos, no por FPS) ---
    if time.time() - last_swarm_move > ENEMY_MOVE_INTERVAL and any(e["alive"] for e in enemies):
        last_swarm_move = time.time()
        left, right = enemies_bounds()
        # si toca borde, bajar y cambiar dirección
        if (swarm_dir == 1 and right + ENEMY_SPEED >= frame_w - 10) or (swarm_dir == -1 and left - ENEMY_SPEED <= 10):
            for e in enemies:
                if e["alive"]:
                    e["y"] += ENEMY_STEP_DOWN
            swarm_dir *= -1
        else:
            for e in enemies:
                if e["alive"]:
                    e["x"] += ENEMY_SPEED * swarm_dir

    # --- Colisiones bala-enemigo ---
    for b in bullets:
        for e in enemies:
            if not e["alive"]:
                continue
            if (e["x"] <= b["x"] <= e["x"] + ENEMY_W) and (e["y"] <= b["y"] <= e["y"] + ENEMY_H):
                e["alive"] = False
                score += 10
                b["y"] = -9999  # eliminar bala marcándola fuera de pantalla
                break

    # --- Colisiones enemigo-nave / llegar al fondo ---
    ship_rect = (ship_x, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT, ship_x + SHIP_WIDTH, frame_h - SHIP_Y_OFFSET)
    for e in enemies:
        if not e["alive"]:
            continue
        ex1, ey1, ex2, ey2 = e["x"], e["y"], e["x"] + ENEMY_W, e["y"] + ENEMY_H
        # tocar fondo
        if ey2 >= frame_h - SHIP_Y_OFFSET - 5:
            lives -= 1
            e["alive"] = False
            continue
        # colisión AABB con la nave
        sx1, sy1, sx2, sy2 = ship_rect
        if not (ex2 < sx1 or ex1 > sx2 or ey2 < sy1 or ey1 > sy2):
            lives -= 1
            e["alive"] = False

    # --- Dibujar enemigos ---
    for e in enemies:
        if not e["alive"]:
            continue
        cv2.rectangle(frame, (e["x"], e["y"]), (e["x"] + ENEMY_W, e["y"] + ENEMY_H), (0, 255, 0), 2)
        cx = e["x"] + ENEMY_W // 2
        cy = e["y"] + ENEMY_H // 2
        cv2.circle(frame, (cx - 10, cy - 5), 4, (0, 255, 0), -1)
        cv2.circle(frame, (cx + 10, cy - 5), 4, (0, 255, 0), -1)
        cv2.line(frame, (cx - 12, cy + 6), (cx + 12, cy + 6), (0, 255, 0), 2)

    # --- Dibujar nave ---
    cv2.rectangle(frame,
                  (ship_x, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT),
                  (ship_x + SHIP_WIDTH, frame_h - SHIP_Y_OFFSET),
                  (255, 255, 255), -1)
    # punta de la nave
    p1 = (ship_x + SHIP_WIDTH // 2, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT - 12)
    p2 = (ship_x + SHIP_WIDTH // 2 - 12, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT + 2)
    p3 = (ship_x + SHIP_WIDTH // 2 + 12, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT + 2)
    cv2.fillPoly(frame, [np.array([p1, p2, p3])], (200, 200, 200))

    # --- HUD ---
    cv2.putText(frame, f"Puntos: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"Vidas: {lives}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Mueve la cara para controlar. Parpadea para disparar.",
                (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # --- Fin de juego ---
    if lives <= 0:
        cv2.putText(frame, "GAME OVER", (frame_w//2 - 150, frame_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        cv2.imshow("Space Invaders con la Cara", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    cv2.imshow("Space Invaders con la Cara", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
