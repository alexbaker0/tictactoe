// Tic-Tac-Toe Zero engine: game rules, the tiny two-headed net and PUCT MCTS,
// ported from the Python (game.py / network.py / mcts.py). Small enough to run
// on the main thread; also loadable in Node for tests.

// ---------------------------------------------------------------- game
// Board: array of 9, row-major (0 1 2 / 3 4 5 / 6 7 8). +1 = X (first), -1 = O.

const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

const legalMoves = b => b.reduce((a, v, i) => (v === 0 && a.push(i), a), []);

function applyMove(b, player, m) {
  const nb = b.slice();
  nb[m] = player;
  return nb;
}

function winLine(b) {
  for (const l of LINES) {
    const s = b[l[0]] + b[l[1]] + b[l[2]];
    if (s === 3 || s === -3) return l;
  }
  return null;
}

// Outcome from X's view: 1, -1, 0 (draw) or null (ongoing).
function result(b) {
  const l = winLine(b);
  if (l) return b[l[0]];
  return b.includes(0) ? null : 0;
}

// ---------------------------------------------------------------- network
// input(18: own stones, opponent stones) -> 64 relu -> 64 relu -> policy(9) + tanh value

class Net {
  constructor(w) { this.w = w; }

  dense(x, W, b, relu) {
    const out = b.slice();
    for (let i = 0; i < x.length; i++) {
      if (x[i] === 0) continue;
      const row = W[i];
      for (let j = 0; j < out.length; j++) out[j] += x[i] * row[j];
    }
    if (relu) for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0;
    return out;
  }

  // Masked softmax over legal moves -> {probs (length 9), value} for `player`.
  predict(board, player, legal) {
    const x = new Array(18).fill(0);
    for (let i = 0; i < 9; i++) {
      const v = board[i] * player;
      if (v === 1) x[i] = 1; else if (v === -1) x[9 + i] = 1;
    }
    const { W1, b1, W2, b2, Wp, bp, Wv, bv } = this.w;
    const a2 = this.dense(this.dense(x, W1, b1, true), W2, b2, true);
    const logits = this.dense(a2, Wp, bp, false);
    const value = Math.tanh(this.dense(a2, Wv, bv, false)[0]);
    let mx = -Infinity;
    for (const m of legal) mx = Math.max(mx, logits[m]);
    const probs = new Array(9).fill(0);
    let s = 0;
    for (const m of legal) { probs[m] = Math.exp(logits[m] - mx); s += probs[m]; }
    for (const m of legal) probs[m] /= s;
    return { probs, value };
  }
}

// ---------------------------------------------------------------- MCTS (PUCT)
// Same search as mcts.py, without root noise (play.py used add_noise=False).
// Child Q is from the opponent's view, so it is negated when choosing.

class Node {
  constructor(board, player, prior) {
    this.board = board; this.player = player; this.prior = prior;
    this.N = 0; this.W = 0; this.children = null; this.terminal = undefined;
  }
  q() { return this.N > 0 ? this.W / this.N : 0; }
}

function runMCTS(net, board, player, nSims, cPuct = 1.5) {
  const expand = node => {
    const res = result(node.board);
    if (res !== null) { node.terminal = res; return res * node.player; }
    const legal = legalMoves(node.board);
    const { probs, value } = net.predict(node.board, node.player, legal);
    node.children = legal.map(m => [m, new Node(applyMove(node.board, node.player, m), -node.player, probs[m])]);
    return value;
  };

  const root = new Node(board, player, 1);
  expand(root);
  for (let s = 0; s < nSims; s++) {
    const path = [root];
    let node = root;
    while (node.children) {
      let total = 0;
      for (const [, ch] of node.children) total += ch.N;
      const sq = total > 0 ? Math.sqrt(total) : 1;
      let best = -1e18, bestCh = null;
      for (const [, ch] of node.children) {
        const score = -ch.q() + cPuct * ch.prior * sq / (1 + ch.N);
        if (score > best) { best = score; bestCh = ch; }
      }
      node = bestCh;
      path.push(node);
    }
    if (node.terminal === undefined) node.terminal = result(node.board);
    const value = node.terminal !== null ? node.terminal * node.player : expand(node);
    for (let i = path.length - 1; i >= 0; i--) {
      const n = path[i];
      n.N += 1;
      n.W += n.player === node.player ? value : -value;
    }
  }
  const counts = new Array(9).fill(0);
  for (const [m, ch] of root.children || []) counts[m] = ch.N;
  return counts;
}

if (typeof module !== "undefined") module.exports = { Net, runMCTS, applyMove, legalMoves, result, winLine };
