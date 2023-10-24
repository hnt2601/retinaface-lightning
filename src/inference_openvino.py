import argparse
from pathlib import Path
from typing import Any
from openvino.runtime import Core
import cv2
import time
import os
import numpy as np
from utils.postprocess import DecodePostProcess, np_decode, np_decode_landm, preprocess

pwd = os.getcwd()


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-p",
        "--model_path",
        type=Path,
        help="Path to the pretrained.",
        required=True,
    )
    arg("--input_shape", type=tuple, default=(640, 640), help="Network input shape")

    return parser.parse_args()


def main() -> None:
    args = get_args()

    cache_path = Path("cache/model_cache")
    cache_path.mkdir(exist_ok=True)
    # Enable caching for OpenVINO Runtime. To disable caching set enable_caching = False
    enable_caching = True
    config_dict = {"CACHE_DIR": str(cache_path)}

    # Load your model here
    model_xml = str(args.model_path)

    # Initialize OpenVINO's Inference Engine
    core = Core()
    # Loading model to the device
    compiled_model = core.compile_model(model_xml, "AUTO", config=config_dict)

    priors = np.load(
        os.path.join(
            pwd, f"weights/prior_{args.input_shape[0]}_{args.input_shape[1]}.npy"
        )
    )

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
        inp = np.expand_dims(inp, axis=0)

        # Create infer request and do inference synchronously
        tic = time.perf_counter()
        results = compiled_model(inp)
        toc = time.perf_counter()
        print("Model forward time: {:.4f}ms".format((toc - tic) * 1000))

        boxes, scores, landms = list(results.values())

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
