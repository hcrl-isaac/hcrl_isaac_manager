"""The ssh every pls call goes through: sessions over a shared ControlMaster take turns (ssh_slot.sh)."""

from pathlib import Path

SSH = str(Path(__file__).resolve().parents[1] / "ssh_slot.sh")
