"""Write the OpenAPI schema to a file: python -m datalyon.export_openapi openapi.json"""

import json
import sys

from .main import create_app


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    spec = create_app().openapi()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"wrote {out} ({len(spec.get('paths', {}))} paths)")


if __name__ == "__main__":
    main()
