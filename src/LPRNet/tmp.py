import argparse
import os

import torch

from model.LPRNet import build_lprnet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LPRNet .pth weights to .pt format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(BASE_DIR, "weights", "Final_LPRNet_model.pth"),
        help="Input .pth checkpoint path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(BASE_DIR, "weights", "Final_LPRNet_model.pt"),
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="torchscript",
        choices=["state_dict", "torchscript"],
        help="Conversion mode: save state_dict or export TorchScript module.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used to load/export model.",
    )
    parser.add_argument("--img_w", type=int, default=94, help="Input image width.")
    parser.add_argument("--img_h", type=int, default=24, help="Input image height.")
    parser.add_argument("--lpr_max_len", type=int, default=8, help="LPR max length.")
    parser.add_argument("--class_num", type=int, default=68, help="Class count.")
    parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate.")
    return parser.parse_args()


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def _strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict(ckpt):
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    if not isinstance(ckpt, dict):
        raise TypeError("Unsupported checkpoint type: {}".format(type(ckpt)))

    for key in ("state_dict", "model_state_dict", "model", "net"):
        val = ckpt.get(key)
        if isinstance(val, dict):
            return _strip_module_prefix(val)
    return _strip_module_prefix(ckpt)


def convert_to_state_dict_pt(input_path, output_path, device):
    ckpt = torch.load(input_path, map_location=device)
    state_dict = extract_state_dict(ckpt)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(state_dict, output_path)


def convert_to_torchscript(
    input_path,
    output_path,
    device,
    lpr_max_len,
    class_num,
    dropout_rate,
    img_h,
    img_w,
):
    model = build_lprnet(
        lpr_max_len=lpr_max_len,
        phase=False,
        class_num=class_num,
        dropout_rate=dropout_rate,
    )
    ckpt = torch.load(input_path, map_location=device)
    state_dict = extract_state_dict(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    example_input = torch.randn(1, 3, img_h, img_w, device=device)
    with torch.no_grad():
        scripted = torch.jit.trace(model, example_input)
        scripted = torch.jit.freeze(scripted)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scripted.save(output_path)


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested.")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    device = torch.device(args.device)

    if not os.path.isfile(input_path):
        raise FileNotFoundError("Input checkpoint not found: {}".format(input_path))

    if args.mode == "state_dict":
        convert_to_state_dict_pt(
            input_path=input_path,
            output_path=output_path,
            device=device,
        )
    else:
        convert_to_torchscript(
            input_path=input_path,
            output_path=output_path,
            device=device,
            lpr_max_len=args.lpr_max_len,
            class_num=args.class_num,
            dropout_rate=args.dropout_rate,
            img_h=args.img_h,
            img_w=args.img_w,
        )

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Conversion finished but output file is invalid: {}".format(output_path))

    print("Convert success ({}): {}".format(args.mode, output_path))


if __name__ == "__main__":
    main()
