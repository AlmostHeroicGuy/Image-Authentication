import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from modifiedResnet10 import ResNet10


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "hybrid_encoder_epoch_100.pth"

VIS_ROOT = Path("visualization_set")
EVAL_ROOT = Path("eval_set")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 128
BATCH_SIZE = 512  # Adjust based on GPU memory
SEED = 69

EXPECTED_EVAL_IMAGES = 5000
NUM_RANDOM_EXAMPLES = 5
NUM_RN_CASES = 10


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPERS
# ============================================================

def extract_epoch_name(checkpoint_name):
    match = re.search(r"epoch_0*(\d+)", checkpoint_name)

    if match:
        return match.group(1)

    return "latest"


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [64, 64])
    return TF.to_tensor(img)


def load_image_batch(paths):
    images = [load_image(path) for path in paths]
    return torch.stack(images, dim=0)


def batched_paths(paths, batch_size):
    for start in range(0, len(paths), batch_size):
        yield start, paths[start:start + batch_size]


@torch.inference_mode()
def embed_tensor_batch(model, x):
    emb = model(x)
    return F.normalize(emb, dim=-1)


@torch.inference_mode()
def compute_path_embeddings(paths, model, label):
    if not paths:
        raise FileNotFoundError(f"No PNG images found for {label}")

    embeddings = []
    total = len(paths)

    for start, batch_paths in batched_paths(paths, BATCH_SIZE):
        print(f"Processing {label}: {start}/{total}")

        x = load_image_batch(batch_paths).to(DEVICE)
        embeddings.append(embed_tensor_batch(model, x).cpu())

    print(f"Processing {label}: {total}/{total}")

    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def compute_folder_embeddings(folder, model):
    paths = sorted(Path(folder).glob("*.png"))
    return compute_path_embeddings(paths, model, str(folder)), paths


def compute_matrix(A, B):
    return (A @ B.T).numpy().astype(np.float32, copy=False)


def compute_stats(M):
    diag = np.diag(M)
    offdiag_mask = ~np.eye(M.shape[0], dtype=bool)
    offdiag = M[offdiag_mask]

    return {
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_max": float(diag.max()),
        "diag_min": float(diag.min()),
        "offdiag_mean": float(offdiag.mean()),
        "offdiag_std": float(offdiag.std()),
        "offdiag_max": float(offdiag.max()),
        "offdiag_min": float(offdiag.min()),
        "gap": float(diag.mean() - offdiag.mean()),
    }


def threshold_analysis(diag_rp, diag_rn):
    labels = np.concatenate([
        np.ones(len(diag_rp), dtype=np.int64),
        np.zeros(len(diag_rn), dtype=np.int64),
    ])

    scores = np.concatenate([diag_rp, diag_rn])
    thresholds = np.linspace(-1, 1, 2001)

    best_acc = -1.0
    best_threshold = 0.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int64)
        acc = float((preds == labels).mean())

        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)

    return best_threshold, best_acc


def save_heatmap(M, title, path):
    plt.figure(figsize=(8, 7))
    plt.imshow(M, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Sample index")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_similarity_histogram(diag_rp, diag_rn, path):
    plt.figure(figsize=(8, 6))
    plt.hist(diag_rp, bins=50, alpha=0.6, label="RP diagonal positives")
    plt.hist(diag_rn, bins=50, alpha=0.6, label="RN diagonal negatives")
    plt.legend()
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Diagonal Similarity Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_matrix_stats(fp, name, M):
    stats = compute_stats(M)
    diag = np.diag(M)
    n = M.shape[0]

    fp.write("\n")
    fp.write("=" * 70 + "\n")
    fp.write(f"{name}\n")
    fp.write("=" * 70 + "\n")

    fp.write("\nMATCHED PAIRS (DIAGONAL)\n")
    fp.write("-" * 40 + "\n")
    fp.write(f"Mean : {stats['diag_mean']:.6f}\n")
    fp.write(f"Std  : {stats['diag_std']:.6f}\n")
    fp.write(f"Max  : {stats['diag_max']:.6f}\n")
    fp.write(f"Min  : {stats['diag_min']:.6f}\n")

    fp.write("\nRANDOM DIAGONAL EXAMPLES\n")
    for idx in random.sample(range(n), min(NUM_RANDOM_EXAMPLES, n)):
        fp.write(f"({idx:04d},{idx:04d}) -> {diag[idx]:.6f}\n")

    fp.write("\nUNMATCHED PAIRS (OFF-DIAGONAL)\n")
    fp.write("-" * 40 + "\n")
    fp.write(f"Mean : {stats['offdiag_mean']:.6f}\n")
    fp.write(f"Std  : {stats['offdiag_std']:.6f}\n")
    fp.write(f"Max  : {stats['offdiag_max']:.6f}\n")
    fp.write(f"Min  : {stats['offdiag_min']:.6f}\n")

    fp.write("\nRANDOM OFF-DIAGONAL EXAMPLES\n")
    count = 0
    while count < min(NUM_RANDOM_EXAMPLES, n * (n - 1)):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

        if i == j:
            continue

        fp.write(f"({i:04d},{j:04d}) -> {M[i, j]:.6f}\n")
        count += 1

    fp.write("\n")
    fp.write(f"Gap (diag - offdiag): {stats['gap']:.6f}\n")


def write_rn_cases(fp, diag_rn, real_paths):
    hardest = np.argsort(diag_rn)[-NUM_RN_CASES:][::-1]
    easiest = np.argsort(diag_rn)[:NUM_RN_CASES]

    fp.write("\n")
    fp.write("=" * 70 + "\n")
    fp.write("REAL-NEGATIVE HARDEST / EASIEST CASES\n")
    fp.write("=" * 70 + "\n")

    fp.write("\nHARDEST 10 RN CASES (highest real-negative similarity)\n")
    for idx in hardest:
        fp.write(f"{idx:04d} -> {diag_rn[idx]:.6f} | {real_paths[idx]}\n")

    fp.write("\nEASIEST 10 RN CASES (lowest real-negative similarity)\n")
    for idx in easiest:
        fp.write(f"{idx:04d} -> {diag_rn[idx]:.6f} | {real_paths[idx]}\n")


def write_threshold_results(fp, best_threshold, best_acc):
    fp.write("\n")
    fp.write("=" * 70 + "\n")
    fp.write("THRESHOLD ANALYSIS\n")
    fp.write("=" * 70 + "\n")
    fp.write("Positive scores: diagonal RP similarities\n")
    fp.write("Negative scores: diagonal RN similarities\n")
    fp.write("Threshold sweep: 2001 points from -1 to 1\n")
    fp.write(f"Best Threshold : {best_threshold:.6f}\n")
    fp.write(f"Best Accuracy  : {100 * best_acc:.2f}%\n")


def get_png_paths(folder):
    return sorted(Path(folder).glob("*.png"))


def validate_aligned_image_folders(root, folder_names, expected_count=None):
    paths_by_name = {}
    filenames_by_name = {}

    for folder_name in folder_names:
        folder = root / folder_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing required folder: {folder}\n"
                f"Please run 'python make_eval_set.py' first to generate eval_set/"
            )

        paths = get_png_paths(folder)

        if not paths:
            raise FileNotFoundError(f"No PNG images found in {folder}")

        paths_by_name[folder_name] = paths
        filenames_by_name[folder_name] = [path.name for path in paths]

    reference_name = folder_names[0]
    reference_filenames = filenames_by_name[reference_name]

    for folder_name in folder_names[1:]:
        if filenames_by_name[folder_name] != reference_filenames:
            raise RuntimeError(
                f"Misaligned image filenames between {root / reference_name} "
                f"and {root / folder_name}"
            )

    if expected_count is not None and len(reference_filenames) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} aligned images in {root}, "
            f"got {len(reference_filenames)}"
        )

    return paths_by_name


def validate_inputs():
    """Validate that precomputed eval_set and visualization_set folders exist."""
    eval_paths = validate_aligned_image_folders(
        EVAL_ROOT,
        ["real", "positive", "negative", "masks"],
        expected_count=EXPECTED_EVAL_IMAGES,
    )

    required_vis_paths = [
        VIS_ROOT / "real",
        VIS_ROOT / "positive",
        VIS_ROOT / "negative",
        VIS_ROOT / "visualization_indices.npy",
    ]

    for path in required_vis_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required visualization input: {path}")

    return eval_paths


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    eval_paths = validate_inputs()

    print("Loading model...")
    model = ResNet10(embedding_dim=EMBEDDING_DIM).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    try:
        model.load_state_dict(ckpt["model"])
    except Exception:
        model.load_state_dict(ckpt)

    model.eval()
    print("Checkpoint loaded.")

    epoch_name = extract_epoch_name(CHECKPOINT)
    outdir = Path(f"results_epoch_{epoch_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nLoading pre-computed evaluation set embeddings from folders...")
    print("(This requires eval_set/ to exist. Run 'python make_eval_set.py' first if needed.)")

    real_paths = eval_paths["real"]
    positive_paths = eval_paths["positive"]
    negative_paths = eval_paths["negative"]

    real_emb = compute_path_embeddings(real_paths, model, EVAL_ROOT / "real")
    positive_emb = compute_path_embeddings(positive_paths, model, EVAL_ROOT / "positive")
    negative_emb = compute_path_embeddings(negative_paths, model, EVAL_ROOT / "negative")

    torch.save(real_emb, outdir / "embeddings_real.pt")
    torch.save(positive_emb, outdir / "embeddings_positive.pt")
    torch.save(negative_emb, outdir / "embeddings_negative.pt")

    print("\nComputing full similarity matrices...")
    M_rp_full = compute_matrix(real_emb, positive_emb)
    M_rn_full = compute_matrix(real_emb, negative_emb)
    M_pn_full = compute_matrix(positive_emb, negative_emb)

    diag_rp_full = np.diag(M_rp_full)
    diag_rn_full = np.diag(M_rn_full)

    best_threshold, best_acc = threshold_analysis(diag_rp_full, diag_rn_full)

    save_similarity_histogram(
        diag_rp_full,
        diag_rn_full,
        outdir / "similarity_histogram.png",
    )

    print("\nComputing 200-image visualization matrices...")
    real_vis_emb, real_vis_paths = compute_folder_embeddings(VIS_ROOT / "real", model)
    positive_vis_emb, positive_vis_paths = compute_folder_embeddings(VIS_ROOT / "positive", model)
    negative_vis_emb, negative_vis_paths = compute_folder_embeddings(VIS_ROOT / "negative", model)

    if not (
        len(real_vis_paths) == len(positive_vis_paths) == len(negative_vis_paths) == 200
    ):
        raise RuntimeError(
            "Expected 200 aligned PNG images in each visualization_set folder."
        )

    M_rp_vis = compute_matrix(real_vis_emb, positive_vis_emb)
    M_rn_vis = compute_matrix(real_vis_emb, negative_vis_emb)
    M_pn_vis = compute_matrix(positive_vis_emb, negative_vis_emb)

    save_heatmap(
        M_rp_vis,
        "Real vs Positive",
        outdir / "heatmap_real_positive.png",
    )
    save_heatmap(
        M_rn_vis,
        "Real vs Negative",
        outdir / "heatmap_real_negative.png",
    )
    save_heatmap(
        M_pn_vis,
        "Positive vs Negative",
        outdir / "heatmap_positive_negative.png",
    )

    np.savez_compressed(
        outdir / "similarity_matrices.npz",
        M_rp_full=M_rp_full,
        M_rn_full=M_rn_full,
        M_pn_full=M_pn_full,
        M_rp_vis=M_rp_vis,
        M_rn_vis=M_rn_vis,
        M_pn_vis=M_pn_vis,
    )

    with open(outdir / "statistics.txt", "w", encoding="utf-8") as fp:
        fp.write(f"Checkpoint: {CHECKPOINT}\n")
        fp.write(f"Checkpoint Epoch: {epoch_name}\n")
        fp.write(f"Device: {DEVICE}\n")
        fp.write(f"Full quantitative samples: {len(real_paths)}\n")
        fp.write("Full matrices are computed from eval_set/ (pre-generated, reproducible).\n")
        fp.write("Visualization matrices use only visualization_set's 200 aligned samples.\n")

        write_matrix_stats(fp, "REAL vs POSITIVE (FULL)", M_rp_full)
        write_matrix_stats(fp, "REAL vs NEGATIVE (FULL)", M_rn_full)
        write_matrix_stats(fp, "POSITIVE vs NEGATIVE (FULL)", M_pn_full)
        write_rn_cases(fp, diag_rn_full, real_paths)
        write_threshold_results(fp, best_threshold, best_acc)

    print()
    print("=" * 70)
    print("Evaluation complete")
    print(f"Results saved to: {outdir}")
    print("=" * 70)
    print()
    print("To inspect hardest failures with masks:")
    print(f"  - Check results in {outdir}/statistics.txt")
    print(f"  - View hardest case images with: eval_set/real/, eval_set/negative/, eval_set/masks/")
    print(f"  - Example: Open eval_set/real/00123.png, eval_set/negative/00123.png, eval_set/masks/00123.png")



if __name__ == "__main__":
    main()
