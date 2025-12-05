uv run .\game.py --steps 100000000 --state enhanced
uv run .\game.py --evals --steps 1000000 --state enhanced
uv run .\game.py --steps 100000000 --state naive
uv run .\game.py --evals --steps 1000000 --state naive
uv run .\game.py --steps 100000000 --state basic
uv run .\game.py --evals --steps 1000000 --state basic