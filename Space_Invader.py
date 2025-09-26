import cv2
import numpy as np
import random
import time
import platform

# ==========================
# Configuración del juego
# ==========================
MAX_LIVES = 3

# Nave
SHIP_Y_OFFSET = 80          # distancia desde la parte inferior
SHIP_WIDTH = 80
SHIP_HEIGHT = 30

# Disparo / parpadeo
BULLET_SPEED = 12
BULLET_RADIUS = 6
BLINK_COOLDOWN = 0.35       # seg. mínimo entre disparos
EYES_CLOSED_FRAMES = 2      # # frames consecutivos sin ojos para considerar "cerrados"
EYES_OPEN_MAX_GAP = 10      # # frames máx. para considerar un parpadeo breve
COOLDOWN_BAR_W = 200

# Enemigos (valores base nivel 1; se incrementan por ola)
ENEMY_COLS_BASE = 6
ENEMY_ROWS_BASE = 3
ENEMY_W = 50
ENEMY_H = 30
ENEMY_H_GAP = 20
ENEMY_V_GAP = 26
ENEMY_SPEED_BASE = 3             # pix por paso
ENEMY_STEP_DOWN = 18
ENEMY_MOVE_INTERVAL_BASE = 0.08  # segundos entre pasos del enjambre

# Dibujo
MARGIN_X = 10
WINDOW_TITLE = "Space Invaders con Control Facial"

# Sonido (solo Windows; opcional)
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    try:
        import winsound
    except Exception:
        winsound = None
else:
    winsound = None

def sfx_shoot():
    if winsound:
        try: winsound.Beep(1200, 60)
        except: pass

def sfx_hit():
    if winsound:
        try: winsound.Beep(700, 70)
        except: pass

def sfx_hurt():
    if winsound:
        try: winsound.Beep(400, 100)
        except: pass

# ==========================
# Utilidades de enemigos
# ==========================
def init_enemies(frame_w, level=1):
    """
    Crea el enjambre centrado. Escala columnas/velocidad con el nivel.
    """
    cols = min(8, ENEMY_COLS_BASE + max(0, level - 1))  # crece hasta 8
    rows = ENEMY_ROWS_BASE  # puedes subir a 4 en niveles altos si quieres
    enemies = []
    # centrar enjambre
    total_w = cols * ENEMY_W + (cols - 1) * ENEMY_H_GAP
    start_x = max(MARGIN_X, (frame_w - total_w) // 2)
    start_y = 60
    for r in range(rows):
        for c in range(cols):
            ex = start_x + c * (ENEMY_W + ENEMY_H_GAP)
            ey = start_y + r * (ENEMY_H + ENEMY_V_GAP)
            enemies.append({"x": ex, "y": ey, "alive": True})
    # velocidad y cadencia según nivel
    enemy_speed = min(ENEMY_SPEED_BASE + (level - 1), 12)
    enemy_interval = max(ENEMY_MOVE_INTERVAL_BASE * (0.96 ** (level - 1)), 0.02)
    return enemies, enemy_speed, enemy_interval

def enemies_bounds(enemies):
    xs = [e["x"] for e in enemies if e["alive"]]
    xe = [e["x"] + ENEMY_W for e in enemies if e["alive"]]
    if not xs:
        return (0, 0)
    return (min(xs), max(xe))

# ==========================
# Detección CV
# ==========================
def load_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    # Robustez: si falla el de ojos con lentes, usa el genérico
    if face_cascade.empty():
        raise RuntimeError("No se cargó el cascade de rostro (haarcascade_frontalface_default.xml).")
    if eye_cascade.empty():
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if eye_cascade.empty():
            raise RuntimeError("No se cargó ningún cascade de ojos.")
    return face_cascade, eye_cascade

def detect_face_and_eyes(gray, frame, face_cascade, eye_cascade):
    """
    Retorna: (main_face_bbox | None, eyes_detected: bool)
    Dibuja opcionalmente boxes para debug visual.
    """
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    main_face = None
    face_boxes = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_boxes.append((x, y, x + w, y + h))
    if face_boxes:
        main_face = max(face_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    eyes_detected = False
    if main_face:
        fx1, fy1, fx2, fy2 = main_face
        roi_gray = gray[fy1:fy2, fx1:fx2]
        if roi_gray.size > 0:
            scale = 0.7
            roi_small = cv2.resize(roi_gray, (0, 0), fx=scale, fy=scale)
            eyes = eye_cascade.detectMultiScale(roi_small, 1.1, 5, minSize=(20, 20))
            # dibujar ojos detectados
            for (ex, ey, ew, eh) in eyes[:2]:
                gx1 = fx1 + int(ex / scale)
                gy1 = fy1 + int(ey / scale)
                gx2 = gx1 + int(ew / scale)
                gy2 = gy1 + int(eh / scale)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 1)
            eyes_detected = len(eyes) >= 1
    return main_face, eyes_detected

# ==========================
# Lógica de juego
# ==========================
def move_swarm(enemies, frame_w, swarm_dir, enemy_speed, last_swarm_move, enemy_interval):
    """
    Mueve el enjambre por intervalos; baja y cambia dirección al tocar borde.
    Retorna (swarm_dir, last_swarm_move)
    """
    now = time.time()
    if now - last_swarm_move <= enemy_interval or not any(e["alive"] for e in enemies):
        return swarm_dir, last_swarm_move
    last_swarm_move = now
    left, right = enemies_bounds(enemies)
    if (swarm_dir == 1 and right + enemy_speed >= frame_w - MARGIN_X) or (swarm_dir == -1 and left - enemy_speed <= MARGIN_X):
        for e in enemies:
            if e["alive"]:
                e["y"] += ENEMY_STEP_DOWN
        swarm_dir *= -1
    else:
        for e in enemies:
            if e["alive"]:
                e["x"] += enemy_speed * swarm_dir
    return swarm_dir, last_swarm_move

def update_bullets_and_collisions(frame, bullets, enemies):
    """
    Mueve balas, detecta colisiones con enemigos. Devuelve (nuevas_balas, score_delta, hit_flag)
    """
    score_delta = 0
    hit_flag = False
    new_bullets = []
    for b in bullets:
        b["y"] -= BULLET_SPEED
        if b["y"] > 0:
            new_bullets.append(b)
        # dibujar
        cv2.circle(frame, (int(b["x"]), int(b["y"])), BULLET_RADIUS, (0, 255, 255), -1)
        # colisiones
        for e in enemies:
            if not e["alive"]:
                continue
            if (e["x"] <= b["x"] <= e["x"] + ENEMY_W) and (e["y"] <= b["y"] <= e["y"] + ENEMY_H):
                e["alive"] = False
                score_delta += 10
                hit_flag = True
                b["y"] = -9999  # saca la bala
                break
    return new_bullets, score_delta, hit_flag

def draw_enemies(frame, enemies):
    for e in enemies:
        if not e["alive"]:
            continue
        cv2.rectangle(frame, (e["x"], e["y"]), (e["x"] + ENEMY_W, e["y"] + ENEMY_H), (0, 255, 0), 2)
        cx = e["x"] + ENEMY_W // 2
        cy = e["y"] + ENEMY_H // 2
        cv2.circle(frame, (cx - 10, cy - 5), 4, (0, 255, 0), -1)
        cv2.circle(frame, (cx + 10, cy - 5), 4, (0, 255, 0), -1)
        cv2.line(frame, (cx - 12, cy + 6), (cx + 12, cy + 6), (0, 255, 0), 2)

def draw_ship(frame, ship_x, frame_h):
    cv2.rectangle(frame,
                  (ship_x, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT),
                  (ship_x + SHIP_WIDTH, frame_h - SHIP_Y_OFFSET),
                  (255, 255, 255), -1)
    p1 = (ship_x + SHIP_WIDTH // 2, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT - 12)
    p2 = (ship_x + SHIP_WIDTH // 2 - 12, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT + 2)
    p3 = (ship_x + SHIP_WIDTH // 2 + 12, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT + 2)
    cv2.fillPoly(frame, [np.array([p1, p2, p3])], (200, 200, 200))

def draw_hud(frame, score, lives, eyes_detected, last_shot_time, frame_h):
    # Puntos / Vidas
    cv2.putText(frame, f"Puntos: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Vidas: {lives}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Estado ojos
    cv2.putText(frame, f"OJOS: {'ABIERTOS' if eyes_detected else 'CERRADOS'}",
                (10, frame_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if eyes_detected else (0, 0, 255), 2)
    # Cooldown de disparo
    cd = max(0.0, BLINK_COOLDOWN - (time.time() - last_shot_time))
    fill_w = int(COOLDOWN_BAR_W * (1 - cd / BLINK_COOLDOWN)) if BLINK_COOLDOWN > 0 else COOLDOWN_BAR_W
    cv2.putText(frame, "Cooldown", (10, frame_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(frame, (110, frame_h - 38), (110 + COOLDOWN_BAR_W, frame_h - 23), (100, 100, 100), 1)
    cv2.rectangle(frame, (110, frame_h - 38), (110 + fill_w, frame_h - 23), (0, 255, 255), -1)

def draw_menu(frame):
    cv2.putText(frame, "SPACE INVADERS (Control Facial)", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(frame, "Mueve la cara para apuntar", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "Parpadea para disparar", (40, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "ENTER: Iniciar   P: Pausa   Q: Salir", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

def draw_pause(frame):
    cv2.putText(frame, "PAUSA (P para continuar)", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

def draw_gameover(frame, score):
    cv2.putText(frame, "GAME OVER", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"Puntaje: {score}", (40, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, "R: Reiniciar   Q: Salir", (40, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# ==========================
# Main
# ==========================
def main():
    face_cascade, eye_cascade = load_cascades()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    # Estado global
    lives = MAX_LIVES
    score = 0
    level = 1
    ship_x = 0
    bullets = []
    hit_flash_until = 0
    # Enjambre
    enemies = []
    swarm_dir = 1
    last_swarm_move = time.time()
    enemy_speed = ENEMY_SPEED_BASE
    enemy_interval = ENEMY_MOVE_INTERVAL_BASE

    # Parpadeo
    last_shot_time = 0.0
    eyes_closed_count = 0
    eyes_open_count = 0
    was_closed = False
    recent_face = False
    last_face_seen = 0.0

    # Estados de juego
    state = "menu"  # "menu" | "playing" | "paused" | "gameover"

    # Tamaño frame inicial
    ret, init_frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame")
        cap.release()
        cv2.destroyAllWindows()
        return
    frame_h, frame_w = init_frame.shape[:2]
    enemies, enemy_speed, enemy_interval = init_enemies(frame_w, level)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Espejo y escala de grises
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Entrada de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Detección facial y ojos solamente si jugamos (o para mostrar en menú)
            main_face, eyes_detected = detect_face_and_eyes(gray, frame, face_cascade, eye_cascade)

            # Control por rostro (mapeo X)
            if main_face:
                fx1, fy1, fx2, fy2 = main_face
                face_cx = (fx1 + fx2) // 2
                ship_x = int(np.clip(face_cx - SHIP_WIDTH // 2, 0, frame_w - SHIP_WIDTH))
                recent_face = True
                last_face_seen = time.time()
            else:
                recent_face = (time.time() - last_face_seen) < 1.0

            # FSM de estados
            if state == "menu":
                draw_menu(frame)
                if key == 13:  # Enter
                    # reset
                    lives = MAX_LIVES
                    score = 0
                    level = 1
                    bullets.clear()
                    enemies, enemy_speed, enemy_interval = init_enemies(frame_w, level)
                    swarm_dir = 1
                    last_swarm_move = time.time()
                    state = "playing"

            elif state == "paused":
                draw_pause(frame)
                if key == ord('p'):
                    state = "playing"

            elif state == "gameover":
                draw_gameover(frame, score)
                if key == ord('r'):
                    lives = MAX_LIVES
                    score = 0
                    level = 1
                    bullets.clear()
                    enemies, enemy_speed, enemy_interval = init_enemies(frame_w, level)
                    swarm_dir = 1
                    last_swarm_move = time.time()
                    state = "playing"

            elif state == "playing":
                # Toggle pausa
                if key == ord('p'):
                    state = "paused"
                # Detección de parpadeo -> disparo
                if recent_face:
                    if eyes_detected:
                        eyes_open_count += 1
                        if was_closed and eyes_open_count <= EYES_OPEN_MAX_GAP:
                            now = time.time()
                            if (now - last_shot_time) > BLINK_COOLDOWN:
                                bx = ship_x + SHIP_WIDTH // 2
                                by = frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT // 2
                                bullets.append({"x": bx, "y": by})
                                last_shot_time = now
                                sfx_shoot()
                        was_closed = False
                        eyes_closed_count = 0
                    else:
                        eyes_closed_count += 1
                        eyes_open_count = 0
                        if eyes_closed_count >= EYES_CLOSED_FRAMES:
                            was_closed = True
                else:
                    # sin rostro reciente, reiniciar contadores
                    eyes_closed_count = 0
                    eyes_open_count = 0
                    was_closed = False

                # Balas y colisiones con enemigos
                bullets, delta, hit_flag = update_bullets_and_collisions(frame, bullets, enemies)
                if delta:
                    score += delta
                if hit_flag:
                    sfx_hit()
                    hit_flash_until = time.time() + 0.08

                # Mover enjambre
                swarm_dir, last_swarm_move = move_swarm(enemies, frame_w, swarm_dir, enemy_speed, last_swarm_move, enemy_interval)

                # Colisiones enemigo con nave o fondo
                ship_rect = (ship_x, frame_h - SHIP_Y_OFFSET - SHIP_HEIGHT, ship_x + SHIP_WIDTH, frame_h - SHIP_Y_OFFSET)
                for e in enemies:
                    if not e["alive"]:
                        continue
                    ex1, ey1, ex2, ey2 = e["x"], e["y"], e["x"] + ENEMY_W, e["y"] + ENEMY_H
                    # tocar fondo
                    if ey2 >= frame_h - SHIP_Y_OFFSET - 5:
                        lives -= 1
                        e["alive"] = False
                        sfx_hurt()
                        hit_flash_until = time.time() + 0.12
                        continue
                    # colisión AABB con la nave
                    sx1, sy1, sx2, sy2 = ship_rect
                    if not (ex2 < sx1 or ex1 > sx2 or ey2 < sy1 or ey1 > sy2):
                        lives -= 1
                        e["alive"] = False
                        sfx_hurt()
                        hit_flash_until = time.time() + 0.12

                # Siguiente ola si no quedan
                if not any(e["alive"] for e in enemies):
                    score += 100  # bonus limpiar ola
                    level += 1
                    enemies, enemy_speed, enemy_interval = init_enemies(frame_w, level)
                    swarm_dir = 1
                    last_swarm_move = time.time()

                # Dibujar escena
                draw_enemies(frame, enemies)
                draw_ship(frame, ship_x, frame_h)
                draw_hud(frame, score, lives, eyes_detected, last_shot_time, frame_h)

                # Flash de feedback
                if time.time() < hit_flash_until:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
                    alpha = 0.2
                    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Fin de juego
                if lives <= 0:
                    state = "gameover"

            # Mostrar ventana
            cv2.imshow(WINDOW_TITLE, frame)

        # fin while
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
