# Tic-Tac-Toe Zero

A tiny AlphaZero-style agent that learned tic-tac-toe purely by self-play, playable in the browser.

**Play:** https://alexbaker0.github.io/tictactoe/

- The network (18 inputs -> 64 -> 64 -> policy + value) was written and trained from scratch in NumPy, with hand-written backprop.
- `engine.js` ports the game, network and PUCT Monte Carlo Tree Search from the Python. MCTS visit counts match the Python exactly.
- Difficulty = search simulations per move: Easy 1 (the network's instinct only), Medium 25, Hard 400. At 400 it was checked against every possible sequence of opponent moves and never loses.
- "Show its thinking" overlays the share of search visits each square got on the AI's last move.
