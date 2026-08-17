import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdistiller.engine.cfg import load_config_dict
p=argparse.ArgumentParser(); p.add_argument('--cfg', required=True); args=p.parse_args()
print(yaml.safe_dump(load_config_dict(args.cfg), allow_unicode=True, sort_keys=False))
