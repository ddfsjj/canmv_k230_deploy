"""Short local entry for the K230 + VQ deploy GUI."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from vq_deploy_gradio import main


if __name__ == "__main__":
    main()
