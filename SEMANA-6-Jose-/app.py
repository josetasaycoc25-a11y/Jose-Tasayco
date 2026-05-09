from flask import Flask, jsonify, request, render_template
import base64
import math
import random
import threading
import time
import webbrowser
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

app = Flask(__name__)

# ==========================================================
# PROYECTO CAPSTONE - SEMANA 6
# Tres en Raya con IA invencible + OpenCV liviano
# Humano = X | IA = O
# ==========================================================

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]

# Orden estratégico: centro -> esquinas -> bordes.
# Esto mantiene buen rendimiento porque Alfa-Beta encuentra mejores jugadas antes.
MOVE_ORDER = [4, 0, 2, 6, 8, 1, 3, 5, 7]

game_lock = threading.Lock()

# Cargamos varios clasificadores para mejorar la detección.
# A veces uno no detecta por iluminación, ángulo o escala; por eso probamos varios.
CASCADE_PATHS = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
    cv2.data.haarcascades + "haarcascade_profileface.xml",
]

FACE_CASCADES = []
for path in CASCADE_PATHS:
    cascade = cv2.CascadeClassifier(path)
    if not cascade.empty():
        FACE_CASCADES.append(cascade)

camera_state = {
    "camera_available": False,
    "face_detected": False,
    "face_count": 0,
    "last_detection_at": 0.0,
    "message": "Cámara pendiente",
    "boxes": [],
    "debug": "Sin análisis"
}

game_state = {
    "started": False,
    "player_name": "Jugador",
    "difficulty": "imposible",
    "board": [""] * 9,
    "turn": "X",
    "game_over": False,
    "winner": None,
    "message": "Ingresa tu nombre, elige un modo y presiona Iniciar partida.",
    "human_score": 0,
    "ai_score": 0,
    "draw_score": 0,
    "last_metrics": {
        "nodes_pure": 0,
        "nodes_alpha_beta": 0,
        "cuts": 0,
        "saved_nodes": 0,
        "saving_pct": 0,
        "score": 0,
        "mode": "Esperando partida"
    }
}


# ==========================================================
# MOTOR DEL TABLERO
# ==========================================================

def empty_board() -> List[str]:
    return [""] * 9


def available_moves(board: List[str]) -> List[int]:
    return [i for i in MOVE_ORDER if board[i] == ""]


def check_winner(board: List[str]) -> Optional[str]:
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board: List[str]) -> bool:
    return all(cell != "" for cell in board) and check_winner(board) is None


def terminal_score(board: List[str], depth: int) -> Optional[int]:
    winner = check_winner(board)

    if winner == "O":
        return 10 - depth

    if winner == "X":
        return depth - 10

    if is_draw(board):
        return 0

    return None


# ==========================================================
# MINIMAX PURO
# ==========================================================

def minimax_pure(board: List[str], maximizing: bool, depth: int, metrics: Dict[str, int]) -> Tuple[int, int]:
    metrics["nodes"] += 1

    score = terminal_score(board, depth)
    if score is not None:
        return score, -1

    best_move = -1

    if maximizing:
        best_score = -math.inf

        for move in available_moves(board):
            board[move] = "O"
            score, _ = minimax_pure(board, False, depth + 1, metrics)
            board[move] = ""

            if score > best_score:
                best_score = score
                best_move = move

        return best_score, best_move

    best_score = math.inf

    for move in available_moves(board):
        board[move] = "X"
        score, _ = minimax_pure(board, True, depth + 1, metrics)
        board[move] = ""

        if score < best_score:
            best_score = score
            best_move = move

    return best_score, best_move


# ==========================================================
# MINIMAX CON PODA ALFA-BETA
# ==========================================================

def minimax_alpha_beta(
    board: List[str],
    maximizing: bool,
    alpha: float,
    beta: float,
    depth: int,
    metrics: Dict[str, int]
) -> Tuple[int, int]:
    metrics["nodes"] += 1

    score = terminal_score(board, depth)
    if score is not None:
        return score, -1

    best_move = -1

    if maximizing:
        best_score = -math.inf

        for move in available_moves(board):
            board[move] = "O"
            score, _ = minimax_alpha_beta(board, False, alpha, beta, depth + 1, metrics)
            board[move] = ""

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

            if beta <= alpha:
                metrics["cuts"] += 1
                break

        return best_score, best_move

    best_score = math.inf

    for move in available_moves(board):
        board[move] = "X"
        score, _ = minimax_alpha_beta(board, True, alpha, beta, depth + 1, metrics)
        board[move] = ""

        if score < best_score:
            best_score = score
            best_move = move

        beta = min(beta, best_score)

        if beta <= alpha:
            metrics["cuts"] += 1
            break

    return best_score, best_move


def immediate_move(board: List[str], player: str) -> int:
    """Busca si un jugador puede ganar en la siguiente jugada."""
    for move in available_moves(board):
        board[move] = player
        if check_winner(board) == player:
            board[move] = ""
            return move
        board[move] = ""

    return -1


def alpha_beta_decision(board: List[str], include_pure_metric: bool) -> Dict:
    """Calcula jugada con Alfa-Beta y opcionalmente compara con Minimax puro."""
    ab_metrics = {"nodes": 0, "cuts": 0}
    ab_score, ab_move = minimax_alpha_beta(board[:], True, -math.inf, math.inf, 0, ab_metrics)

    pure_nodes = 0
    pure_score = ab_score
    pure_move = ab_move

    if include_pure_metric:
        pure_metrics = {"nodes": 0}
        pure_score, pure_move = minimax_pure(board[:], True, 0, pure_metrics)
        pure_nodes = pure_metrics["nodes"]

    saved = max(pure_nodes - ab_metrics["nodes"], 0)
    saving_pct = round((saved / pure_nodes) * 100, 2) if pure_nodes else 0

    return {
        "move": ab_move,
        "score": ab_score,
        "pure_score": pure_score,
        "pure_move": pure_move,
        "nodes_pure": pure_nodes,
        "nodes_alpha_beta": ab_metrics["nodes"],
        "cuts": ab_metrics["cuts"],
        "saved_nodes": saved,
        "saving_pct": saving_pct
    }


def choose_ai_move(board: List[str], difficulty: str) -> Dict:
    moves = available_moves(board)

    if not moves:
        return {
            "move": -1,
            "score": 0,
            "nodes_pure": 0,
            "nodes_alpha_beta": 0,
            "cuts": 0,
            "saved_nodes": 0,
            "saving_pct": 0,
            "mode": "Sin movimientos"
        }

    if difficulty == "facil":
        return {
            "move": random.choice(moves),
            "score": 0,
            "nodes_pure": 0,
            "nodes_alpha_beta": 0,
            "cuts": 0,
            "saved_nodes": 0,
            "saving_pct": 0,
            "mode": "Fácil: jugada aleatoria"
        }

    if difficulty == "normal":
        win = immediate_move(board, "O")
        if win != -1:
            result = alpha_beta_decision(board, include_pure_metric=False)
            result["move"] = win
            result["mode"] = "Normal: victoria inmediata"
            return result

        block = immediate_move(board, "X")
        if block != -1:
            result = alpha_beta_decision(board, include_pure_metric=False)
            result["move"] = block
            result["mode"] = "Normal: bloqueo táctico"
            return result

        if random.random() < 0.60:
            result = alpha_beta_decision(board, include_pure_metric=False)
            result["mode"] = "Normal: análisis Alfa-Beta parcial"
            return result

        return {
            "move": random.choice(moves),
            "score": 0,
            "nodes_pure": 0,
            "nodes_alpha_beta": 0,
            "cuts": 0,
            "saved_nodes": 0,
            "saving_pct": 0,
            "mode": "Normal: jugada flexible"
        }

    result = alpha_beta_decision(board, include_pure_metric=True)
    result["mode"] = "Imposible: Minimax + Poda Alfa-Beta"
    return result


def update_terminal_state() -> None:
    winner = check_winner(game_state["board"])

    if winner:
        game_state["winner"] = winner
        game_state["game_over"] = True
        game_state["turn"] = "-"

        if winner == "X":
            game_state["human_score"] += 1
            game_state["message"] = f"{game_state['player_name']} ganó. En modo imposible esto no debería ocurrir."
        else:
            game_state["ai_score"] += 1
            game_state["message"] = "La IA ganó con una jugada óptima."

        return

    if is_draw(game_state["board"]):
        game_state["winner"] = "EMPATE"
        game_state["game_over"] = True
        game_state["turn"] = "-"
        game_state["draw_score"] += 1
        game_state["message"] = "Empate óptimo. La IA no perdió."
        return


def ai_play_if_needed() -> None:
    if game_state["game_over"] or game_state["turn"] != "O":
        return

    result = choose_ai_move(game_state["board"], game_state["difficulty"])
    move = result["move"]

    if move != -1 and game_state["board"][move] == "":
        game_state["board"][move] = "O"

    game_state["last_metrics"] = result
    update_terminal_state()

    if not game_state["game_over"]:
        game_state["turn"] = "X"
        game_state["message"] = f"Turno de {game_state['player_name']}. Rostro detectado: selecciona una casilla."


def refresh_camera_staleness() -> None:
    if camera_state["last_detection_at"] <= 0:
        return

    if time.time() - camera_state["last_detection_at"] > 4.0:
        camera_state["face_detected"] = False
        camera_state["face_count"] = 0
        camera_state["boxes"] = []
        camera_state["message"] = "Rostro no actualizado"


def snapshot() -> Dict:
    refresh_camera_staleness()

    can_play = (
        game_state["started"]
        and camera_state["face_detected"]
        and not game_state["game_over"]
        and game_state["turn"] == "X"
    )

    return {
        **game_state,
        "camera": camera_state.copy(),
        "can_play": can_play
    }


# ==========================================================
# OPENCV MEJORADO Y LIVIANO
# ==========================================================

def decode_base64_image(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    raw = base64.b64decode(data_url)
    img_array = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def normalize_frame(frame):
    """Prepara imagen para detección sin hacerla pesada."""
    if frame is None:
        return None

    h, w = frame.shape[:2]

    # Tamaño balanceado: mejor que 260x195 para rostros medianos,
    # pero aún ligero para laptops.
    target_w = 480
    scale = target_w / float(w)
    target_h = int(h * scale)
    resized = cv2.resize(frame, (target_w, target_h))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Mejora contraste cuando hay luces fuertes o fondo blanco.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Otra versión ecualizada para probar si CLAHE falla.
    gray_equalized = cv2.equalizeHist(gray)

    return resized, [gray, gray_clahe, gray_equalized], scale


def merge_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Elimina cajas repetidas cuando varios cascades detectan el mismo rostro."""
    if not boxes:
        return []

    final_boxes = []

    for box in boxes:
        x, y, w, h = box
        keep = True

        for fx, fy, fw, fh in final_boxes:
            ix1 = max(x, fx)
            iy1 = max(y, fy)
            ix2 = min(x + w, fx + fw)
            iy2 = min(y + h, fy + fh)

            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = w * h
            farea = fw * fh
            union = area + farea - inter

            if union > 0 and inter / union > 0.35:
                keep = False
                break

        if keep:
            final_boxes.append(box)

    return final_boxes


def detect_faces_from_frame(frame) -> Tuple[bool, int, List[Dict], str]:
    """
    Detección más sensible:
    - Usa varios clasificadores Haar.
    - Prueba imagen normal, CLAHE y ecualizada.
    - Prueba imagen volteada para perfil.
    """
    prepared = normalize_frame(frame)
    if prepared is None:
        return False, 0, [], "frame vacío"

    resized, gray_versions, scale = prepared
    all_boxes = []
    debug_steps = []

    for cascade_index, cascade in enumerate(FACE_CASCADES):
        for gray_index, gray in enumerate(gray_versions):
            # Parámetros menos estrictos que antes.
            detected = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(34, 34),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            debug_steps.append(f"C{cascade_index}-G{gray_index}:{len(detected)}")

            for (x, y, w, h) in detected:
                # Filtra falsos positivos demasiado pequeños o alargados.
                ratio = w / float(h)
                if 0.65 <= ratio <= 1.45 and w >= 34 and h >= 34:
                    all_boxes.append((int(x), int(y), int(w), int(h)))

    # Prueba perfil en imagen volteada, útil si el rostro no está totalmente frontal.
    flipped_versions = [cv2.flip(g, 1) for g in gray_versions]
    for cascade_index, cascade in enumerate(FACE_CASCADES):
        for gray_index, gray in enumerate(flipped_versions):
            detected = cascade.detectMultiScale(
                gray,
                scaleFactor=1.10,
                minNeighbors=3,
                minSize=(34, 34),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            debug_steps.append(f"F{cascade_index}-G{gray_index}:{len(detected)}")

            frame_width = gray.shape[1]
            for (x, y, w, h) in detected:
                x_original = frame_width - x - w
                ratio = w / float(h)
                if 0.65 <= ratio <= 1.45 and w >= 34 and h >= 34:
                    all_boxes.append((int(x_original), int(y), int(w), int(h)))

    merged = merge_boxes(all_boxes)

    # Convertimos cajas a coordenadas normalizadas para dibujar en el navegador.
    boxes_normalized = []
    rh, rw = resized.shape[:2]

    for x, y, w, h in merged[:3]:
        boxes_normalized.append({
            "x": round(x / rw, 4),
            "y": round(y / rh, 4),
            "w": round(w / rw, 4),
            "h": round(h / rh, 4)
        })

    detected = len(boxes_normalized) > 0
    debug = " | ".join(debug_steps[:10])

    return detected, len(boxes_normalized), boxes_normalized, debug


# ==========================================================
# RUTAS API
# ==========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with game_lock:
        return jsonify(snapshot())


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    data = request.get_json(force=True)

    name = str(data.get("player_name", "Jugador")).strip()
    difficulty = str(data.get("difficulty", "imposible")).strip().lower()

    if not name:
        name = "Jugador"

    if difficulty not in ["facil", "normal", "imposible"]:
        difficulty = "imposible"

    with game_lock:
        game_state["started"] = True
        game_state["player_name"] = name[:30]
        game_state["difficulty"] = difficulty
        game_state["board"] = empty_board()
        game_state["turn"] = "X"
        game_state["game_over"] = False
        game_state["winner"] = None
        game_state["message"] = f"Turno de {game_state['player_name']}. Mira a la cámara y selecciona una casilla."
        game_state["last_metrics"] = {
            "nodes_pure": 0,
            "nodes_alpha_beta": 0,
            "cuts": 0,
            "saved_nodes": 0,
            "saving_pct": 0,
            "score": 0,
            "mode": "Esperando jugada humana"
        }

        return jsonify(snapshot())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with game_lock:
        game_state["board"] = empty_board()
        game_state["turn"] = "X"
        game_state["game_over"] = False
        game_state["winner"] = None
        game_state["message"] = f"Turno de {game_state['player_name']}. Mira a la cámara y selecciona una casilla."
        game_state["last_metrics"] = {
            "nodes_pure": 0,
            "nodes_alpha_beta": 0,
            "cuts": 0,
            "saved_nodes": 0,
            "saving_pct": 0,
            "score": 0,
            "mode": "Partida reiniciada"
        }

        return jsonify(snapshot())


@app.route("/api/detect_face", methods=["POST"])
def api_detect_face():
    data = request.get_json(force=True)
    image_data = data.get("image", "")

    try:
        frame = decode_base64_image(image_data)
        detected, count, boxes, debug = detect_faces_from_frame(frame)
    except Exception as exc:
        detected, count, boxes, debug = False, 0, [], f"error: {exc}"

    with game_lock:
        camera_state["camera_available"] = True
        camera_state["face_detected"] = detected
        camera_state["face_count"] = count
        camera_state["boxes"] = boxes
        camera_state["last_detection_at"] = time.time()
        camera_state["message"] = "Rostro detectado" if detected else "Rostro no detectado"
        camera_state["debug"] = debug

        return jsonify(snapshot())


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json(force=True)
    index = int(data.get("index", -1))

    with game_lock:
        refresh_camera_staleness()

        if not game_state["started"]:
            game_state["message"] = "Primero inicia la partida."
            return jsonify(snapshot()), 400

        if not camera_state["face_detected"]:
            game_state["message"] = f"{game_state['player_name']}, la cámara debe detectar tu rostro para jugar."
            return jsonify(snapshot()), 403

        if game_state["game_over"]:
            return jsonify(snapshot()), 409

        if game_state["turn"] != "X":
            return jsonify(snapshot()), 409

        if index < 0 or index > 8 or game_state["board"][index] != "":
            game_state["message"] = "Movimiento inválido. Elige una casilla vacía."
            return jsonify(snapshot()), 400

        game_state["board"][index] = "X"
        update_terminal_state()

        if not game_state["game_over"]:
            game_state["turn"] = "O"
            game_state["message"] = "La IA está pensando..."
            ai_play_if_needed()

        return jsonify(snapshot())


@app.route("/api/clear_score", methods=["POST"])
def api_clear_score():
    with game_lock:
        game_state["human_score"] = 0
        game_state["ai_score"] = 0
        game_state["draw_score"] = 0
        return jsonify(snapshot())


if __name__ == "__main__":
    def open_browser():
        time.sleep(1.0)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser, daemon=True).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
