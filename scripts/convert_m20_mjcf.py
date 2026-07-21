"""Convert the released M20 MuJoCo model to an Isaac USD asset.

The public M20 policy was trained against the released MJCF.  Isaac Lab's
generic converter does not enable the MJCF importer in every headless app, so
this project-owned wrapper enables it explicitly and records the output path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_INPUT = (
    DATA_ROOT
    / "public_experts/m20_native/source/src/M20_sdk_deploy/model/M20/mjcf/M20.xml"
)
DEFAULT_OUTPUT = DATA_ROOT / "assets/m20_mjcf_official_v1.usd"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--fix-base", action="store_true")
parser.add_argument("--import-sites", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.input.is_file():
    parser.error(f"MJCF input not found: {args.input}")
args.output.parent.mkdir(parents=True, exist_ok=True)

app = AppLauncher(args).app

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.mjcf")

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402


def main() -> None:
    # The extension registers its command during the first Kit update.
    app.update()
    cfg = MjcfConverterCfg(
        asset_path=os.path.abspath(args.input),
        usd_dir=os.path.abspath(args.output.parent),
        usd_file_name=args.output.name,
        fix_base=args.fix_base,
        import_sites=args.import_sites,
        force_usd_conversion=True,
        make_instanceable=False,
    )
    converter = MjcfConverter(cfg)
    print(f"[M20PRO-MJCF] input={args.input}", flush=True)
    print(f"[M20PRO-MJCF] output={converter.usd_path}", flush=True)


try:
    main()
finally:
    app.close()
