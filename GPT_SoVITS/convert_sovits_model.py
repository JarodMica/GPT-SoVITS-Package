import argparse
import torch
import sys
from collections import OrderedDict

##
# If your 'utils' package is in a different location, adjust sys.path or imports accordingly.
# Make sure you can import `HParams` from the location used in your original SoVITS code.
##
try:
    from utils import HParams
    # Needed to safely unpickle references to HParams
    torch.serialization.add_safe_globals([HParams])
except ImportError:
    print("Could not import HParams from 'utils'. Make sure 'utils' is in your PYTHONPATH.")
    sys.exit(1)


def hparams_to_dict(hps_obj):
    """
    Convert an HParams instance (or a nested structure) to a plain dictionary.
    Adjust this logic as needed based on how your HParams is structured.
    """
    # If HParams has a 'to_dict' method, you can simply do:
    # return hps_obj.to_dict()

    # Otherwise, suppose HParams is basically storing attributes in __dict__:
    # Filter out any private attributes or unwanted references:
    return {
        k: v
        for k, v in vars(hps_obj).items()
        if not k.startswith("_")  # skip private attrs, if desired
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert a SoVITS model checkpoint referencing HParams into a purely weights-based checkpoint."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the original SoVITS .pth file (referencing utils.HParams).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the new .pth file with no custom class references.",
    )
    args = parser.parse_args()

    # 1) Load the original checkpoint (weights_only=True to ensure restricted unpickling)
    print(f"Loading checkpoint from '{args.input}' ...")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)

    # 2) Check and convert config if it is an HParams instance
    old_config = checkpoint.get("config", None)
    if isinstance(old_config, HParams):
        print("Detected `utils.HParams` in config. Converting to plain dictionary.")
        new_config = hparams_to_dict(old_config)
    else:
        print("No HParams instance found in `config`. Using config as-is (or None).")
        new_config = old_config

    # 3) Build a new checkpoint dict
    #    (You can keep "info" if you want, or remove it if it references custom data.)
    new_checkpoint = OrderedDict()
    new_checkpoint["weight"] = checkpoint["weight"]  # All your model's layers
    new_checkpoint["config"] = new_config            # Plain dictionary instead of HParams
    new_checkpoint["info"] = checkpoint.get("info", "")

    # 4) Save the new checkpoint
    print(f"Saving converted checkpoint to '{args.output}' ...")
    torch.save(new_checkpoint, args.output)
    print("Done! Your new checkpoint no longer references `utils.HParams`.")


if __name__ == "__main__":
    main()
