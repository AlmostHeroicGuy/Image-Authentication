"""
Evaluate a trained forensic encoder using aligned validation and test images.

Required directory layout::

    validation_set/{real,positive,negative}/*.png
    eval_set/{real,positive,negative,masks}/*.png
    visualization_set/{real,positive,negative}/*.png
    visualization_set/visualization_indices.npy

Filenames must be aligned across the folders in each set. The validation set
is used only to choose the cosine-similarity threshold. That threshold is then
frozen and applied once to the held-out evaluation set. The checkpoint records
whether images are processed at 224x224 or 256x256. Preprocessing preserves
aspect ratio by resizing the shorter side and then taking a center crop.

Full similarity matrices require O(N^2) memory and disk space. There is no
fixed evaluation-set size, but very large sets may require a chunked approach.
"""

import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from architecture import ResNet50
from processing.preprocessing import preprocess_image


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = r"C:\Users\tusha\Deep\Asymmetric_Contrastive_Learning\Realistic_Prototype_Resnet50Based\model_checkpoints\hybrid_encoder_epoch_200.pth"

VIS_ROOT = Path("visualization_set")
VAL_ROOT = Path("validation_set")
EVAL_ROOT = Path("eval_set")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 2048
IMAGE_SIZE = 224  # Fallback for older checkpoints without image-size metadata.
BATCH_SIZE = 256  # Adjust based on GPU memory
SEED = 69

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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# HELPERS
# ============================================================

def extract_epoch_name(checkpoint_name):
    match = re.search(r"epoch_0*(\d+)", checkpoint_name)

    if match:
        return match.group(1)

    return "latest"


def load_image(path, image_size):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return preprocess_image(img, image_size)


def load_image_batch(paths, image_size):
    images = [load_image(path, image_size) for path in paths]
    return torch.stack(images, dim=0)


def batched_paths(paths, batch_size):
    for start in range(0, len(paths), batch_size):
        yield start, paths[start:start + batch_size]


@torch.inference_mode()
def embed_tensor_batch(model, x):
    emb = model(x)
    return F.normalize(emb, dim=-1)


@torch.inference_mode()
def compute_path_embeddings(paths, model, label, image_size):
    if not paths:
        raise FileNotFoundError(f"No PNG images found for {label}")

    embeddings = []
    total = len(paths)

    for start, batch_paths in batched_paths(paths, BATCH_SIZE):
        print(f"Processing {label}: {start}/{total}")

        x = load_image_batch(batch_paths, image_size).to(DEVICE)
        embeddings.append(embed_tensor_batch(model, x).cpu())

    print(f"Processing {label}: {total}/{total}")

    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def compute_folder_embeddings(folder, model, image_size):
    paths = sorted(Path(folder).glob("*.png"))
    return compute_path_embeddings(paths, model, str(folder), image_size), paths


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

    scores = np.concatenate([diag_rp, diag_rn]).astype(np.float64, copy=False)
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


def classification_metrics(diag_rp, diag_rn, threshold):
    """Compute held-out binary metrics at one frozen similarity threshold."""
    labels = np.concatenate([
        np.ones(len(diag_rp), dtype=np.int64),
        np.zeros(len(diag_rn), dtype=np.int64),
    ])
    scores = np.concatenate([diag_rp, diag_rn]).astype(np.float64, copy=False)
    predictions = (scores >= threshold).astype(np.int64)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    def safe_divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    return {
        "threshold": float(threshold),
        "accuracy": safe_divide(tp + tn, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1_score": safe_divide(2 * precision * recall, precision + recall),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_negative_rate": safe_divide(fn, fn + tp),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def roc_analysis(diag_rp, diag_rn):
    """Return ROC points, area under the curve, and approximate EER."""
    labels = np.concatenate([
        np.ones(len(diag_rp), dtype=np.int64),
        np.zeros(len(diag_rn), dtype=np.int64),
    ])
    scores = np.concatenate([diag_rp, diag_rn]).astype(np.float64, copy=False)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    threshold_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]

    tpr = np.r_[0.0, true_positives[threshold_ends] / np.sum(labels == 1)]
    fpr = np.r_[0.0, false_positives[threshold_ends] / np.sum(labels == 0)]
    thresholds = np.r_[np.inf, sorted_scores[threshold_ends]]

    auc = float(np.trapz(tpr, fpr))
    fnr = 1.0 - tpr
    difference = fpr - fnr
    crossings = np.flatnonzero(difference[:-1] * difference[1:] <= 0)

    if len(crossings):
        left = int(crossings[0])
        right = left + 1
        denominator = difference[right] - difference[left]
        weight = 0.0 if denominator == 0 else -difference[left] / denominator
        eer = float(fpr[left] + weight * (fpr[right] - fpr[left]))
        if np.isfinite(thresholds[left]) and np.isfinite(thresholds[right]):
            eer_threshold = float(
                thresholds[left]
                + weight * (thresholds[right] - thresholds[left])
            )
        else:
            eer_threshold = float(thresholds[right])
    else:
        eer_index = int(np.argmin(np.abs(difference)))
        eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
        eer_threshold = float(thresholds[eer_index])

    return fpr, tpr, thresholds, auc, eer, eer_threshold


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


def save_roc_curve(fpr, tpr, auc, eer, path):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (EER = {eer:.4f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_confusion_matrix(metrics, path):
    matrix = np.array([
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")

    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > midpoint else "black"
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=color,
            )

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


def write_threshold_results(
    fp,
    selected_threshold,
    validation_accuracy,
    metrics,
    roc_auc,
    eer,
    eer_threshold,
):
    fp.write("\n")
    fp.write("=" * 70 + "\n")
    fp.write("VALIDATION THRESHOLD SELECTION\n")
    fp.write("=" * 70 + "\n")
    fp.write("Threshold selected only on validation_set/.\n")
    fp.write("Threshold sweep: 2001 points from -1 to 1\n")
    fp.write(f"Selected Threshold   : {selected_threshold:.6f}\n")
    fp.write(f"Validation Accuracy  : {100 * validation_accuracy:.2f}%\n")

    fp.write("\n")
    fp.write("=" * 70 + "\n")
    fp.write("HELD-OUT EVALUATION METRICS\n")
    fp.write("=" * 70 + "\n")
    fp.write(f"Accuracy             : {100 * metrics['accuracy']:.2f}%\n")
    fp.write(f"Precision            : {metrics['precision']:.6f}\n")
    fp.write(f"Recall               : {metrics['recall']:.6f}\n")
    fp.write(f"F1-score             : {metrics['f1_score']:.6f}\n")
    fp.write(f"False Positive Rate  : {metrics['false_positive_rate']:.6f}\n")
    fp.write(f"False Negative Rate  : {metrics['false_negative_rate']:.6f}\n")
    fp.write(f"ROC AUC              : {roc_auc:.6f}\n")
    fp.write(f"Equal Error Rate     : {eer:.6f}\n")
    fp.write(f"EER Threshold        : {eer_threshold:.6f}\n")

    fp.write("\nCONFUSION MATRIX COUNTS\n")
    fp.write("-" * 40 + "\n")
    fp.write(f"True Positive  : {metrics['true_positive']}\n")
    fp.write(f"True Negative  : {metrics['true_negative']}\n")
    fp.write(f"False Positive : {metrics['false_positive']}\n")
    fp.write(f"False Negative : {metrics['false_negative']}\n")


def get_png_paths(folder):
    return sorted(Path(folder).glob("*.png"))


def validate_aligned_image_folders(
    root,
    folder_names,
    expected_count=None,
    minimum_count=2,
):
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

    if len(reference_filenames) < minimum_count:
        raise RuntimeError(
            f"Expected at least {minimum_count} aligned images in {root}, "
            f"got {len(reference_filenames)}"
        )

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
    """Validate precomputed validation, evaluation, and visualization inputs."""
    validation_paths = validate_aligned_image_folders(
        VAL_ROOT,
        ["real", "positive", "negative"],
    )
    eval_paths = validate_aligned_image_folders(
        EVAL_ROOT,
        ["real", "positive", "negative", "masks"],
    )

    visualization_paths = validate_aligned_image_folders(
        VIS_ROOT,
        ["real", "positive", "negative"],
        expected_count=200,
    )
    visualization_indices = VIS_ROOT / "visualization_indices.npy"
    if not visualization_indices.exists():
        raise FileNotFoundError(
            f"Missing required visualization input: {visualization_indices}"
        )

    return validation_paths, eval_paths, visualization_paths


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    validation_paths, eval_paths, visualization_paths = validate_inputs()

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    image_size = ckpt.get("image_size", IMAGE_SIZE)
    if image_size not in (224, 256):
        raise ValueError(
            f"Checkpoint image_size must be 224 or 256, got {image_size}."
        )

    print("Loading model...")
    model = ResNet50().to(DEVICE)

    try:
        model.load_state_dict(ckpt["model"])
    except Exception:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Checkpoint loaded (input size: {image_size}x{image_size}).")

    epoch_name = extract_epoch_name(CHECKPOINT)
    outdir = Path(f"results_epoch_{epoch_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nSelecting threshold from validation_set/...")
    validation_real_emb = compute_path_embeddings(
        validation_paths["real"], model, VAL_ROOT / "real", image_size
    )
    validation_positive_emb = compute_path_embeddings(
        validation_paths["positive"], model, VAL_ROOT / "positive", image_size
    )
    validation_negative_emb = compute_path_embeddings(
        validation_paths["negative"], model, VAL_ROOT / "negative", image_size
    )
    validation_rp = np.sum(
        validation_real_emb.numpy() * validation_positive_emb.numpy(), axis=1
    )
    validation_rn = np.sum(
        validation_real_emb.numpy() * validation_negative_emb.numpy(), axis=1
    )
    selected_threshold, validation_accuracy = threshold_analysis(
        validation_rp, validation_rn
    )
    print(
        f"Frozen threshold: {selected_threshold:.6f} "
        f"(validation accuracy: {100 * validation_accuracy:.2f}%)"
    )

    print("\nLoading pre-computed evaluation set embeddings from folders...")
    print("(This requires eval_set/ to exist. Run 'python make_eval_set.py' first if needed.)")

    real_paths = eval_paths["real"]
    positive_paths = eval_paths["positive"]
    negative_paths = eval_paths["negative"]

    real_emb = compute_path_embeddings(real_paths, model, EVAL_ROOT / "real", image_size)
    positive_emb = compute_path_embeddings(
        positive_paths, model, EVAL_ROOT / "positive", image_size
    )
    negative_emb = compute_path_embeddings(
        negative_paths, model, EVAL_ROOT / "negative", image_size
    )

    torch.save(real_emb, outdir / "embeddings_real.pt")
    torch.save(positive_emb, outdir / "embeddings_positive.pt")
    torch.save(negative_emb, outdir / "embeddings_negative.pt")

    print("\nComputing full similarity matrices...")
    M_rp_full = compute_matrix(real_emb, positive_emb)
    M_rn_full = compute_matrix(real_emb, negative_emb)
    M_pn_full = compute_matrix(positive_emb, negative_emb)

    diag_rp_full = np.diag(M_rp_full)
    diag_rn_full = np.diag(M_rn_full)

    metrics = classification_metrics(
        diag_rp_full, diag_rn_full, selected_threshold
    )
    fpr, tpr, roc_thresholds, roc_auc, eer, eer_threshold = roc_analysis(
        diag_rp_full, diag_rn_full
    )

    save_similarity_histogram(
        diag_rp_full,
        diag_rn_full,
        outdir / "similarity_histogram.png",
    )
    save_roc_curve(
        fpr,
        tpr,
        roc_auc,
        eer,
        outdir / "roc_curve.png",
    )
    save_confusion_matrix(metrics, outdir / "confusion_matrix.png")

    metrics_output = {
        "selected_threshold": float(selected_threshold),
        "validation_accuracy": float(validation_accuracy),
        "roc_auc": roc_auc,
        "equal_error_rate": eer,
        "eer_threshold": eer_threshold,
        **metrics,
    }
    with open(outdir / "evaluation_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics_output, fp, indent=2)

    np.savez_compressed(
        outdir / "roc_data.npz",
        false_positive_rate=fpr,
        true_positive_rate=tpr,
        thresholds=roc_thresholds,
    )

    print("\nComputing 200-image visualization matrices...")
    real_vis_paths = visualization_paths["real"]
    positive_vis_paths = visualization_paths["positive"]
    negative_vis_paths = visualization_paths["negative"]
    real_vis_emb = compute_path_embeddings(
        real_vis_paths, model, VIS_ROOT / "real", image_size
    )
    positive_vis_emb = compute_path_embeddings(
        positive_vis_paths, model, VIS_ROOT / "positive", image_size
    )
    negative_vis_emb = compute_path_embeddings(
        negative_vis_paths, model, VIS_ROOT / "negative", image_size
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
        fp.write(f"Image size: {image_size}x{image_size}\n")
        fp.write(f"Validation samples: {len(validation_paths['real'])}\n")
        fp.write(f"Full quantitative samples: {len(real_paths)}\n")
        fp.write("Full matrices are computed from eval_set/ (pre-generated, reproducible).\n")
        fp.write("Visualization matrices use only visualization_set's 200 aligned samples.\n")

        write_matrix_stats(fp, "REAL vs POSITIVE (FULL)", M_rp_full)
        write_matrix_stats(fp, "REAL vs NEGATIVE (FULL)", M_rn_full)
        write_matrix_stats(fp, "POSITIVE vs NEGATIVE (FULL)", M_pn_full)
        write_rn_cases(fp, diag_rn_full, real_paths)
        write_threshold_results(
            fp,
            selected_threshold,
            validation_accuracy,
            metrics,
            roc_auc,
            eer,
            eer_threshold,
        )

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
