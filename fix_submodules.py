#!/usr/bin/env python3
import subprocess
import sys

# List of commands to run (in order)
cmds = [
    ["git", "rm", "-r", "--cached", "Informer2020", "rational_kat_cu"],
    ["rm", "-rf", "Informer2020/.git", "rational_kat_cu/.git"],
    ["git", "add", "Informer2020", "rational_kat_cu"],
    ["git", "commit", "-m", "Convert Informer2020/ & rational_kat_cu/ from submodules into normal folders"],
    ["git", "push", "origin", "main"],
]

def run(cmd):
    print(f"> {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"ERROR ({result.returncode}):\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    else:
        print(result.stdout)

def main():
    for c in cmds:
        run(c)
    print("All done! Check GitHub to confirm your folders are now regular directories.")

if __name__ == "__main__":
    main()
