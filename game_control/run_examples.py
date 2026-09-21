# AI-assisted experimental code; full human and mathematical review is not established.
from pathlib import Path
import game_control

if __name__ == "__main__":
    examples = Path(__file__).with_name("examples.py")
    exec(compile(examples.read_text(encoding="utf-8"), str(examples), "exec"), vars(game_control))
