import argparse
from pathlib import Path
from typing import Any
import torch
import yaml
from addict import Dict as Adict
import onnx
from onnxsim import simplify
import io
from models.retinaface_module import RetinaFaceModule


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg(
        "-p",
        "--pretrain_path",
        type=Path,
        help="Path to the pretrained.",
        required=True,
    )
    arg("--cpu", action="store_true", default=True, help="Use cpu inference")
    arg("--input_shape", type=tuple, default=(640, 640), help="Network input shape")

    arg("--onnxsim", action="store_true", default=True, help="Use cpu inference")

    return parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print("Missing keys:{}".format(len(missing_keys)))
    print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    print("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print("Loading pretrained model from {}".format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def main() -> None:
    args = get_args()

    with args.config_path.open() as f:
        config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    module = RetinaFaceModule(config)

    model = load_model(module.model, args.pretrain_path, args.cpu)
    model.eval()
    print("Finished loading model!")

    # Export to ONNX
    onnx_path = str(args.pretrain_path).replace("pth", "onnx")
    input_names = ["input"]
    output_names = ["bboxes", "confs", "lmks"]
    input_shapes = {input_names[0]: [1, 3, *args.input_shape]}
    onnx_bytes = io.BytesIO()
    zero_input = torch.zeros(*input_shapes[input_names[0]])
    dynamic_axes = {input_names[0]: {0: "batch"}}
    for _, name in enumerate(output_names):
        dynamic_axes[name] = dynamic_axes[input_names[0]]
    extra_args = {
        "opset_version": 10,
        "verbose": False,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }

    torch.onnx.export(model, zero_input, onnx_bytes, **extra_args)
    with open(onnx_path, "wb") as out:
        out.write(onnx_bytes.getvalue())

    # Optimization ONNX model
    if args.onnxsim:
        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(onnx_path)
        model_simp, check = simplify(onnx_model, test_input_shapes=input_shapes)

        assert check, "Simplified ONNX model could not be validated"

        onnx.save(model_simp, onnx_path)

        print("Generated simplified onnx model named {}".format(onnx_path))


if __name__ == "__main__":
    main()
