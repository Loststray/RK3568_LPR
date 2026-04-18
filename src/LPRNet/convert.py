import argparse
import inspect
import os
import torch

from model.LPRNet import build_lprnet
from model.STNet import build_STNet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Export LPRNet PyTorch model to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(BASE_DIR, "weights", "Final_LPRNet_model.pth"),
        help="Path to .pth weights file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(BASE_DIR, "weights", "LPRNet.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument("--img_w", type=int, default=94, help="Input image width")
    parser.add_argument("--img_h", type=int, default=24, help="Input image height")
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--lpr_max_len", type=int, default=8, help="LPR max length")
    parser.add_argument("--class_num", type=int, default=68, help="Number of classes")
    parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate")
    parser.add_argument("--model",default='LPRNet',help="model to convert")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for export",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="Enable dynamic batch axis in ONNX",
    )
    return parser.parse_args()


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def load_model(mdl, weights_path, lpr_max_len, class_num, dropout_rate, device):
    if mdl == "LPRNet":
        model = build_lprnet(
        lpr_max_len=lpr_max_len,
        phase=False,
        class_num=class_num,
        dropout_rate=dropout_rate,
        )
    else:
        model = build_STNet(False)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError("Weights file not found: {}".format(weights_path))

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def export_onnx(model, output_path, batch_size, img_h, img_w, opset, dynamic_batch, device):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dummy_input = torch.randn(batch_size, 3, img_h, img_w, device=device)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["images"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
    }

    # Force a single-file ONNX when the exporter supports this switch.
    try:
        export_sig = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
        export_sig = {}

    if "external_data" in export_sig:
        export_kwargs["external_data"] = False
    elif "use_external_data_format" in export_sig:
        export_kwargs["use_external_data_format"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            **export_kwargs,
        )


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested.")

    device = torch.device(args.device)
    weights_path = resolve_path(args.weights)
    output_path = resolve_path(args.output)

    model = load_model(
        mdl=args.model,
        weights_path=weights_path,
        lpr_max_len=args.lpr_max_len,
        class_num=args.class_num,
        dropout_rate=args.dropout_rate,
        device=device,
    )

    export_onnx(
        model=model,
        output_path=output_path,
        batch_size=args.batch_size,
        img_h=args.img_h,
        img_w=args.img_w,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        device=device,
    )

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("ONNX export finished but output file is invalid: {}".format(output_path))

    print("ONNX export success: {}".format(output_path))


if __name__ == "__main__":
    main()
