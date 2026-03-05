#!/usr/bin/env python3
"""Generate module-graph.json for the mdbook interactive architecture page.

Reads the rustdoc JSON output and produces a Cytoscape-ready graph with:
- Module nodes (compound, color-coded by category)
- Public item nodes (children of modules)
- Dependency edges between modules
"""
import json
import sys
from pathlib import Path

RUSTDOC_JSON = Path("target/doc/yao_rs.json")
OUTPUT = Path("docs/src/static/module-graph.json")

CATEGORIES = {
    "gate": "core",
    "circuit": "core",
    "state": "core",
    "apply": "simulation",
    "instruct": "simulation",
    "instruct_qubit": "simulation",
    "measure": "simulation",
    "einsum": "tensor",
    "tensors": "tensor",
    "torch_contractor": "tensor",
    "index": "utility",
    "bitutils": "utility",
    "json": "utility",
    "easybuild": "higher",
    "operator": "higher",
    "noise": "higher",
    "typst": "visualization",
}


def main():
    if not RUSTDOC_JSON.exists():
        print(
            "Error: {} not found. Run:\n"
            "  cargo +nightly rustdoc -- -Z unstable-options --output-format json".format(
                RUSTDOC_JSON
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    with open(RUSTDOC_JSON) as f:
        data = json.load(f)

    index = data["index"]

    # 1. Find all modules and their public items
    item_to_module = {}
    modules = {}
    for item_id, item in index.items():
        inner = item.get("inner", {})
        if not isinstance(inner, dict) or "module" not in inner:
            continue
        name = item.get("name")
        if name == "yao_rs":
            continue
        children = []
        for child_id in inner["module"].get("items", []):
            cid = str(child_id)
            item_to_module[cid] = name
            if cid in index:
                child = index[cid]
                child_inner = child.get("inner", {})
                kind = (
                    list(child_inner.keys())[0]
                    if isinstance(child_inner, dict)
                    else "unknown"
                )
                doc = child.get("docs", "") or ""
                # Take first sentence/line as summary
                doc_summary = doc.strip().split("\n")[0][:120] if doc.strip() else ""
                children.append({"name": child.get("name"), "kind": kind, "doc": doc_summary})
        modules[name] = {
            "name": name,
            "category": CATEGORIES.get(name, "utility"),
            "doc_path": "{}/index.html".format(name),
            "items": children,
        }

    # 2. Extract inter-module dependencies
    def find_ids(obj, found=None):
        if found is None:
            found = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "id" and isinstance(v, (int, str)):
                    found.add(str(v))
                else:
                    find_ids(v, found)
        elif isinstance(obj, list):
            for v in obj:
                find_ids(v, found)
        return found

    edges = set()
    for item_id, item in index.items():
        src = item_to_module.get(item_id)
        if not src:
            continue
        inner = item.get("inner", {})
        if not isinstance(inner, dict):
            continue
        for ref_id in find_ids(inner):
            dst = item_to_module.get(ref_id)
            if dst and dst != src:
                edges.add((src, dst))

    # 3. Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "modules": sorted(modules.values(), key=lambda m: m["name"]),
        "edges": [{"source": s, "target": t} for s, t in sorted(edges)],
    }
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(
        "Wrote {} ({} modules, {} edges)".format(
            OUTPUT, len(result["modules"]), len(result["edges"])
        )
    )


if __name__ == "__main__":
    main()
