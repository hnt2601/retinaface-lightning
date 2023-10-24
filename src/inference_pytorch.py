import argparse
from pathlib import Path
from typing import Any
import torch
import yaml
from addict import Dict as Adict
import torch.backends.cudnn as cudnn
import cv2
import time
import numpy as np
from models.retinaface_module import RetinaFaceModule
from utils.postprocess import DecodePostProcess, np_decode, np_decode_landm, preprocess


torch.set_grad_enabled(False)


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
    cudnn.benchmark = True

    args = get_args()

    with args.config_path.open() as f:
        config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    module = RetinaFaceModule(config)

    model = load_model(module.model, args.pretrain_path, args.cpu)
    model.eval()
    print("Finished loading model!")
    device = torch.device("cpu" if args.cpu else "cuda")
    model = model.to(device)

    priors = module.prior_box
    dpp = DecodePostProcess(args.input_shape)

    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, image = cap.read()
        image = cv2.resize(image, args.input_shape)

        # Check if the frame was captured successfully
        if not ret:
            print("Failed to capture frame")
            break

        inp, resize = preprocess(image, net_inshape=args.input_shape)

        # Create infer request and do inference synchronously
        tic = time.perf_counter()
        inp_tensor = torch.from_numpy(inp).unsqueeze(0)
        # inp_tensor = inp_tensor.to(device)
        boxes, scores, landms = model(inp_tensor)  # forward pass
        toc = time.perf_counter()
        print("Model forward time: {:.4f}ms".format((toc - tic) * 1000))

        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        landms = landms.detach().cpu().numpy()

        boxes = np_decode(boxes, priors)
        landms = np_decode_landm(landms, priors)

        toc2 = time.perf_counter()
        print("After decode time: {:.4f}ms".format((toc2 - tic) * 1000))

        for box, landm, score in zip(boxes, landms, scores):
            dets = dpp(score, box, landm, resize)

            for b in dets:
                startX, startY, endX, endY = b[:4].astype(np.int32)
                lmk = b[5:15].reshape((5, 2))

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

                lmk = lmk.astype(np.int32)
                cv2.circle(image, tuple(lmk[0]), 1, (0, 0, 255), 2)
                cv2.circle(image, tuple(lmk[1]), 1, (0, 255, 255), 2)
                cv2.circle(image, tuple(lmk[2]), 1, (255, 0, 255), 2)
                cv2.circle(image, tuple(lmk[3]), 1, (0, 255, 0), 2)
                cv2.circle(image, tuple(lmk[4]), 1, (255, 0, 0), 2)

                conf = "{:.4f}".format(b[4])
                cx = startX
                cy = startY + 12
                cv2.putText(
                    image, conf, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
                )

        cv2.imshow("Webcam Feed", image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
