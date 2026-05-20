# Chess


## StockFish Engine



# https://python-chess.readthedocs.io/en/latest/engine.html
# https://blog.propelauth.com/chess-analysis-in-python/


1. start the server

```bash
python tasks/classic_games/chess/stockfish_server.py serve --pool-size 64 --skill-level 19 --host 127.0.0.1 --port 8080
```

2. do a speed test

```bash
python tasks/classic_games/chess/stockfish_bench.py
python tasks/classic_games/chess/stockfish_bench.py --endpoint analyse
```

3. call client in your loop like script `tool.py`

```bash
python tool.py
```