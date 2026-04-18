from pathlib import Path


DATASET_DIR = Path("/home/fiatiustitia/RK3568_LPR/src/dataset/CCPD_test")
OUTPUT_FILE = Path("/home/fiatiustitia/RK3568_LPR/src/RKNN/dataset.txt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_image_paths(dataset_dir: Path,max_size = 100) -> list[Path]:
    image_paths: list[Path] = []

    for path in dataset_dir.rglob("*"):
        if len(image_paths) >= max_size:
            break
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path.resolve())

    return sorted(image_paths, key=str)


def write_dataset_file(image_paths: list[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for image_path in image_paths:
            file.write(f"{image_path}\n")


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    image_paths = collect_image_paths(DATASET_DIR)
    write_dataset_file(image_paths, OUTPUT_FILE)
    print(f"Saved {len(image_paths)} image paths to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
