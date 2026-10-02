#!/usr/bin/env python3
from __future__ import annotations
import json
import pathlib
import shutil
import shlex
import sys

OUT_OF_SCOPE = {"powershell", "pwsh", "msbuild", "xcodebuild", "xcrun", "codesign", "signtool"}

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_act_command_coverage.py MANIFEST", file=sys.stderr)
        return 2
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    for item in data["commands"]:
        tokens = shlex.split(item["command"])
        tool = pathlib.PurePosixPath(tokens[0]).name if tokens else ""
        platforms = {str(value).lower() for value in item.get("platforms", ["linux"])}
        if "linux" not in platforms or tool in OUT_OF_SCOPE:
            category = "act_out_of_scope"
            reason = "Linux/act is not a declared target"
        elif shutil.which(tool) is None:
            category = "dependency_required"
            reason = f"{tool} is not available in the runner"
        else:
            category = "ubuntu_standard"
            reason = "detected executable is available"
        print(f"{category}: {item['name']} [{item.get('stage', 'custom')}] — {reason}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
