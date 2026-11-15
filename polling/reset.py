"""Deletes the pipeline's generated data so you can start fresh.

    python reset.py               # token, state, database (asks first)
    python reset.py --log         # also the classifier log
    python reset.py --browser     # also the saved Handshake login
    python reset.py --all         # everything above
    python reset.py --yes         # skip the confirmation prompt
"""
import argparse
import os
import shutil

from polling import configpoll

ap = argparse.ArgumentParser()
ap.add_argument("--log", action="store_true", help="also delete the classifier log")
ap.add_argument("--browser", action="store_true", help="also delete the saved browser login")
ap.add_argument("--all", action="store_true", help="delete everything listed above")
ap.add_argument("--yes", action="store_true", help="don't ask for confirmation")
args = ap.parse_args()

names = ["token", "state", "database"]
if args.log or args.all:
    names.append("classifier_log")
if args.browser or args.all:
    names.append("browser_profile")

targets = []
for name in names:
    try:
        p = configpoll.path(name)
    except KeyError:
        print(f"(skipping '{name}': not in config.json paths)")
        continue
    targets.append(p)
    if name == "database":  # SQLite WAL mode leaves sidecar files
        targets += [p + "-wal", p + "-shm"]

existing = [p for p in targets if os.path.exists(p)]
if not existing:
    print("Nothing to delete.")
    raise SystemExit

print("Will delete:")
for p in existing:
    print("  ", p)
if not args.yes and input("Type 'yes' to continue: ").strip().lower() != "yes":
    print("Cancelled.")
    raise SystemExit

for p in existing:
    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    print("deleted", p)