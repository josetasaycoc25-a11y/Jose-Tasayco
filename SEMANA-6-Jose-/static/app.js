const boardEl = document.getElementById("board");
const messageBox = document.getElementById("messageBox");
const turnText = document.getElementById("turnText");
const resetBtn = document.getElementById("resetBtn");
const startBtn = document.getElementById("startBtn");
const clearScoreBtn = document.getElementById("clearScoreBtn");
const playerNameInput = document.getElementById("playerName");
const difficultySelect = document.getElementById("difficulty");

const cameraStatus = document.getElementById("cameraStatus");
const statusPill = document.getElementById("statusPill");
const faceText = document.getElementById("faceText");
const faceBox = document.getElementById("faceBox");
const faceBadge = document.getElementById("faceBadge");

const humanName = document.getElementById("humanName");
const modeName = document.getElementById("modeName");
const humanScore = document.getElementById("humanScore");
const aiScore = document.getElementById("aiScore");
const drawScore = document.getElementById("drawScore");
const aiMode = document.getElementById("aiMode");

const nodesPure = document.getElementById("nodesPure");
const nodesAB = document.getElementById("nodesAB");
const cuts = document.getElementById("cuts");
const saving = document.getElementById("saving");

const video = document.getElementById("cameraVideo");
const captureCanvas = document.getElementById("cameraCanvas");
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: false });

const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d");

let currentState = null;
let cameraReady = false;
let detectionBusy = false;
let lastBoxes = [];

function formatNumber(n) {
    return new Intl.NumberFormat("es-PE").format(n || 0);
}

function difficultyLabel(value) {
    if (value === "facil") return "Fácil";
    if (value === "normal") return "Normal";
    return "Imposible";
}

function resizeOverlayCanvas() {
    const rect = overlayCanvas.getBoundingClientRect();
    overlayCanvas.width = Math.max(1, Math.round(rect.width));
    overlayCanvas.height = Math.max(1, Math.round(rect.height));
}

function drawFaceBoxes(boxes) {
    resizeOverlayCanvas();

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!boxes || !boxes.length) {
        lastBoxes = [];
        return;
    }

    lastBoxes = boxes;

    overlayCtx.lineWidth = 4;
    overlayCtx.strokeStyle = "#2bcf91";
    overlayCtx.fillStyle = "rgba(43, 207, 145, 0.16)";
    overlayCtx.font = "bold 15px Inter, Arial";

    boxes.forEach((box) => {
        const x = box.x * overlayCanvas.width;
        const y = box.y * overlayCanvas.height;
        const w = box.w * overlayCanvas.width;
        const h = box.h * overlayCanvas.height;

        overlayCtx.fillRect(x, y, w, h);
        overlayCtx.strokeRect(x, y, w, h);

        overlayCtx.fillStyle = "#2bcf91";
        overlayCtx.fillRect(x, Math.max(0, y - 26), 148, 24);

        overlayCtx.fillStyle = "#ffffff";
        overlayCtx.fillText("Rostro humano", x + 8, Math.max(18, y - 8));

        overlayCtx.fillStyle = "rgba(43, 207, 145, 0.16)";
    });
}

function cellClass(value, canPlay) {
    const classes = ["cell"];

    if (value === "X") classes.push("x");
    if (value === "O") classes.push("o");

    if (!value && canPlay) {
        classes.push("empty", "can-play");
    }

    if (!canPlay) {
        classes.push("locked");
    }

    return classes.join(" ");
}

function render(state) {
    currentState = state;
    boardEl.innerHTML = "";

    state.board.forEach((value, index) => {
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = cellClass(value, state.can_play);
        cell.textContent = value;
        cell.setAttribute("aria-label", `Casilla ${index}`);

        if (!value && state.started && !state.game_over) {
            cell.addEventListener("click", () => sendMove(index));
        }

        boardEl.appendChild(cell);
    });

    messageBox.textContent = state.message;
    humanName.textContent = state.player_name || "Jugador";
    modeName.textContent = difficultyLabel(state.difficulty);
    humanScore.textContent = state.human_score || 0;
    aiScore.textContent = state.ai_score || 0;
    drawScore.textContent = state.draw_score || 0;

    if (!state.started) {
        turnText.textContent = "Primero inicia la partida";
    } else if (state.game_over) {
        turnText.textContent = state.winner === "EMPATE"
            ? "Resultado: empate"
            : `Resultado: ganó ${state.winner === "X" ? state.player_name : "IA"}`;
    } else {
        turnText.textContent = state.turn === "X"
            ? `Turno de ${state.player_name}`
            : "Turno de la IA";
    }

    const cam = state.camera || {};
    statusPill.classList.remove("status-ok", "status-blocked");
    faceBox.classList.remove("face-ok", "face-blocked");

    drawFaceBoxes(cam.boxes || []);

    if (cam.face_detected) {
        cameraStatus.textContent = "Rostro detectado";
        faceText.textContent = `Detectado (${cam.face_count})`;
        faceBadge.textContent = `Rostro detectado · Turno de ${state.player_name}`;
        statusPill.classList.add("status-ok");
        faceBox.classList.add("face-ok");
    } else {
        cameraStatus.textContent = cameraReady ? "Sin rostro" : "Activando cámara";
        faceText.textContent = cameraReady ? "No detectado" : "Esperando cámara";
        faceBadge.textContent = cameraReady ? "Mira de frente y acércate un poco" : "Activando cámara...";
        statusPill.classList.add("status-blocked");
        faceBox.classList.add("face-blocked");
    }

    const m = state.last_metrics || {};
    nodesPure.textContent = formatNumber(m.nodes_pure);
    nodesAB.textContent = formatNumber(m.nodes_alpha_beta);
    cuts.textContent = formatNumber(m.cuts);
    saving.textContent = `${m.saving_pct || 0}%`;
    aiMode.textContent = m.mode || "Esperando partida";
}

async function apiPost(url, payload = {}) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const state = await res.json();
    render(state);
    return { res, state };
}

async function refreshState() {
    try {
        const res = await fetch("/api/state");
        const state = await res.json();
        render(state);
    } catch {
        messageBox.textContent = "No se pudo conectar con Flask.";
    }
}

async function startGame() {
    const name = playerNameInput.value.trim() || "Jugador";
    const difficulty = difficultySelect.value;

    await detectFaceLight();

    await apiPost("/api/new_game", {
        player_name: name,
        difficulty
    });
}

async function sendMove(index) {
    if (!currentState?.started) {
        messageBox.textContent = "Primero inicia la partida.";
        return;
    }

    // Antes de jugar, fuerza una detección reciente.
    if (cameraReady) {
        await detectFaceLight();
    }

    if (!currentState?.camera?.face_detected) {
        messageBox.textContent = "Acerca tu rostro a la cámara para poder jugar.";
        return;
    }

    try {
        await apiPost("/api/move", { index });
    } catch {
        messageBox.textContent = "No se pudo enviar la jugada.";
    }
}

async function resetGame() {
    await apiPost("/api/reset");
}

async function clearScore() {
    await apiPost("/api/clear_score");
}

async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 360 },
                frameRate: { ideal: 12, max: 15 }
            },
            audio: false
        });

        video.srcObject = stream;
        cameraReady = true;
        cameraStatus.textContent = "Cámara activa";

        video.addEventListener("loadedmetadata", () => {
            resizeOverlayCanvas();
            setTimeout(detectFaceLight, 800);
        });
    } catch {
        cameraReady = false;
        cameraStatus.textContent = "Sin permiso de cámara";
        faceText.textContent = "Permiso bloqueado";
        faceBadge.textContent = "Permite el acceso a la cámara";
    }
}

async function detectFaceLight() {
    if (!cameraReady || detectionBusy || video.readyState < 2) {
        return;
    }

    detectionBusy = true;

    try {
        // Captura más grande que antes para que Haar detecte mejor.
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        const image = captureCanvas.toDataURL("image/jpeg", 0.55);

        const res = await fetch("/api/detect_face", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image })
        });

        const state = await res.json();
        render(state);
    } catch {
        faceBadge.textContent = "Error detectando rostro";
    } finally {
        detectionBusy = false;
    }
}

startBtn.addEventListener("click", startGame);
resetBtn.addEventListener("click", resetGame);
clearScoreBtn.addEventListener("click", clearScore);

playerNameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") startGame();
});

difficultySelect.addEventListener("change", () => {
    if (currentState?.started) {
        startGame();
    }
});

window.addEventListener("resize", () => {
    drawFaceBoxes(lastBoxes);
});

initCamera();
refreshState();

// Ligero pero más sensible: captura pequeña cada 1 segundo.
setInterval(detectFaceLight, 1000);
setInterval(refreshState, 4200);
