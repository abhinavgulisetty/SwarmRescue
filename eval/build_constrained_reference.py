"""Build an external, pinned NWRCA* reference without bundling AGPL source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from eval.reference_constrained import REVISION, UPSTREAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repository", default=UPSTREAM + ".git")
    args = parser.parse_args()
    output = args.out.absolute()
    if output.exists():
        raise FileExistsError("reference destination must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", args.repository, str(output)], check=True, timeout=120
    )
    subprocess.run(["git", "checkout", REVISION], cwd=output, check=True)
    source = output / "src"
    constants = source / "sys/constants.h"
    text = constants.read_text()
    assert "const dim_t DIM = 3;" in text
    constants.write_text(text.replace("const dim_t DIM = 3;", "const dim_t DIM = 2;"))
    search = source / "search/NWRCA.hpp"
    text = search.read_text()
    needle = "    global_ub[0] = COST_MAX;"
    assert text.count(needle) == 1
    budget = """
    if (const char *budget = std::getenv("EMBER_REFERENCE_DEADLINE")) {
        long value = std::strtol(budget, NULL, 10);
        if (value < 0 || value >= INT32_MAX)
            throw std::runtime_error("invalid explicit deadline");
        global_ub[1] = (cost_t)value;
    }"""
    search.write_text(
        "#include <cstdlib>\n#include <stdexcept>\n"
        + text.replace(needle, needle + budget)
    )
    dispatcher = source / "programs/roadhog.cpp"
    text, replacements = re.subn(
        r'        if\(exp.alg_name == "NWMOA3"\).*?else if\(exp.alg_name == "NWMOA"\)',
        '        if(exp.alg_name == "NWMOA")',
        dispatcher.read_text(),
        count=1,
        flags=re.DOTALL,
    )
    assert replacements == 1
    dispatcher.write_text(text)
    subprocess.run(["make", "path"], cwd=source, check=True)
    binary = source / "bin/roadhog"
    receipt = {
        "upstream": UPSTREAM,
        "revision": REVISION,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["g++", "--version"], text=True),
        "modified_files_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (constants, search, dispatcher)
        },
        "adaptation": "DIM=2; explicit deadline; unused DIM=3 dispatcher omitted",
    }
    (output / "ember-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()
