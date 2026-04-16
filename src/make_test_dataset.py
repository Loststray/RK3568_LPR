import random
import shutil
from pathlib import Path

r'''
\begin{table}
  \begin{tabular}{|c|c|c|c|}
    \hline
    名称 & 原样本数量 & 抽取数量 & 备注
    \hline
    CCPD_base & 20000 & 3000 & 正常数据
    CCPD_challenge & 10000 & 1000 & 比较困难的数据
    CCPD_db & 20000 & 250 & 光线过强或过弱
    CCPD_fn & 20000 & 250 & 距离过近或过远
    CCPD_rotate & 10000 & 250 & 车牌有较大水平倾斜
    CCPD_weather & 10000 & 250 & 雨天、大雾天数据
  \end{tabular}
  \caption{车牌检测数据集}
  \label{tab:dataset}
\end{table}
'''


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_PLAN = {
    "CCPD_base": 3000,
    "CCPD_challenge": 1000,
    "CCPD_db": 250,
    "CCPD_fn": 250,
    "CCPD_rotate": 250,
    "CCPD_weather": 250,
}


def collect_image_files(root_dir):
    image_files = []
    if not root_dir.exists():
        return image_files

    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(path)

    return image_files


def build_unique_target_path(dst_dir, src_path):
    target_path = dst_dir / src_path.name
    if not target_path.exists():
        return target_path

    index = 1
    while True:
        stem = f"{src_path.stem}_{index:03d}"
        candidate = dst_dir / f"{stem}{src_path.suffix.lower()}"
        if not candidate.exists():
            return candidate
        index += 1


def resolve_subset_dir(ccpd2019_root, subset_name):
    suffix = subset_name.split("_", maxsplit=1)[-1]
    candidates = [
        subset_name,
        subset_name.lower(),
        f"ccpd_{suffix}",
        f"CCPD_{suffix}",
    ]

    for candidate in candidates:
        subset_dir = ccpd2019_root / candidate
        if subset_dir.exists() and subset_dir.is_dir():
            return subset_dir
    return None


def main():
    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "dataset"
    ccpd2019_root = dataset_root / "CCPD2019"
    target_root = dataset_root / "CCPD_test"
    target_root.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for subset_name, sample_count in SAMPLE_PLAN.items():
        subset_dir = resolve_subset_dir(ccpd2019_root, subset_name)
        if subset_dir is None:
            print(f"跳过 {subset_name}: 未找到目录")
            continue

        all_images = collect_image_files(subset_dir)
        if not all_images:
            print(f"跳过 {subset_name}: 未找到图片")
            continue

        random.shuffle(all_images)
        selected_images = all_images[: min(sample_count, len(all_images))]

        subset_target_dir = target_root / subset_name
        subset_target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in selected_images:
            target_path = build_unique_target_path(subset_target_dir, image_path)
            shutil.copy2(image_path, target_path)

        total_copied += len(selected_images)
        print(
            f"{subset_name}: 共 {len(all_images)} 张，抽取 {len(selected_images)} 张 -> {subset_target_dir}"
        )

    print(f"总计抽取 {total_copied} 张图片，输出目录: {target_root}")


if __name__ == "__main__":
    main()
