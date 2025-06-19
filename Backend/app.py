# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import chess
import torch
import torch.nn as nn
import numpy as np
import math
import requests

# --- Config ---
MODEL_PATH    = '2_290-0_306.pth'
MODEL_URL     = 'https://github.com/Allen1reality/K-Chess/releases/download/K-chess3/2_290-0_306.pth'
MOVE2ID_JSON  = 'move_to_id.json'
SIMULATIONS   = 200
CPUCT         = 1.0
DROPOUT_P     = 0.2

# --- Dual-headed Network Definition ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout_p=0.0):
        super().__init__()
        self.conv1 = SeparableConv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = SeparableConv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.dropout(out)
        return self.relu(out + identity)

class DualNet(nn.Module):
    def __init__(self, n_moves, dropout_p):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 1, dropout_p),
            ResidualBlock(64, 2, dropout_p),
            ResidualBlock(64, 1, dropout_p),
            ResidualBlock(64, 1, dropout_p),
            ResidualBlock(64, 1, dropout_p)
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.sep = nn.Sequential(
            SeparableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(), nn.Linear(128*8*8, 512), nn.ReLU(),
            nn.Dropout(dropout_p), nn.Linear(512, n_moves)
        )
        self.value_head = nn.Sequential(
            nn.Flatten(), nn.Linear(128*8*8, 512), nn.ReLU(),
            nn.Dropout(dropout_p), nn.Linear(512, 1), nn.Tanh()
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.dilated(x)
        x = self.sep(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

# --- Model Loading & MCTS Utilities ---
def load_model():
    # download model if not present
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
    # load mapping
    with open(MOVE2ID_JSON, 'r') as f:
        m2i = json.load(f)
    id2move = {int(v): k for k, v in m2i.items()}
    # init model
    model = DualNet(len(m2i), DROPOUT_P)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, id2move


def board_to_tensor(board):
    arr = np.zeros((18, 8, 8), dtype=np.float32)
    pm = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2,
          chess.ROOK:3, chess.QUEEN:4, chess.KING:5}
    for sq, p in board.piece_map().items():
        idx = pm[p.piece_type] + (0 if p.color == chess.WHITE else 6)
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        arr[idx, r, c] = 1.0
    arr[12] = board.turn == chess.WHITE
    arr[13] = board.has_kingside_castling_rights(chess.WHITE)
    arr[14] = board.has_queenside_castling_rights(chess.WHITE)
    arr[15] = board.has_kingside_castling_rights(chess.BLACK)
    arr[16] = board.has_queenside_castling_rights(chess.BLACK)
    arr[17] = board.halfmove_clock / 100.0
    return torch.from_numpy(arr).unsqueeze(0)

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
    def expanded(self): return bool(self.children)
    def value(self): return self.value_sum / self.visit_count if self.visit_count else 0.0


def ucb_score(parent, child):
    return child.value() + CPUCT * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)

def expand(node, model, id2move):
    x = board_to_tensor(node.board)
    with torch.no_grad():
        logits, value = model(x)
        probs = torch.softmax(logits[0], dim=0).numpy()
    for aid, uci in id2move.items():
        mv = chess.Move.from_uci(uci)
        if mv in node.board.legal_moves:
            nb = node.board.copy()
            nb.push(mv)
            node.children[uci] = MCTSNode(nb, node, float(probs[aid]))
    return float(value)

def backpropagate(node, value):
    cur, sign = node, 1.0
    while cur:
        cur.visit_count += 1
        cur.value_sum += sign * value
        sign = -sign
        cur = cur.parent

def mcts_search(board, model, id2move):
    root = MCTSNode(board)
    for _ in range(SIMULATIONS):
        node = root
        while node.expanded():
            node = max(node.children.values(), key=lambda c: ucb_score(node, c))
        val = expand(node, model, id2move)
        backpropagate(node, val)
    best = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    return chess.Move.from_uci(best)

# --- FastAPI App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    fen: str

model, id2move = load_model()

@app.post("/api/move")
async def get_move(req: MoveRequest):
    board = chess.Board(req.fen)
    mv = mcts_search(board, model, id2move)
    return {"move": mv.uci()}
