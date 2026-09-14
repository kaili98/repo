"""Dev-only training script - NOT bundled into the frozen exe.

Trains two small SVM classifiers (equation-scale and options-scale) on the
real, hand-verified digit/operator glyphs accumulated in images/digits/eq
and images/digits/opt, and saves them as images/digits/eq_model.joblib and
images/digits/opt_model.joblib for ArithmeticPuzzleSolver to load at runtime.

Re-run this whenever new template PNGs are added to those folders.
"""
import glob
import os

import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from bot.arithmetic_puzzle import _LABEL_FOR_PREFIX, _canonicalize
from bot.digit_features import hog_features


def load_labeled_glyphs(subdir):
    d = os.path.join("images", "digits", subdir)
    items = []
    for filepath in sorted(glob.glob(os.path.join(d, "*.png"))):
        base = os.path.splitext(os.path.basename(filepath))[0]
        prefix = base.rsplit("_", 1)[0]
        label = _LABEL_FOR_PREFIX.get(prefix, prefix)
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        items.append((gray, label, filepath))
    return items


def augment(gray):
    """Generate plausible variants of one real glyph crop, targeting the
    exact kind of variance repeatedly observed between real captures of the
    same digit: different pixel widths (a "3" seen at both 7px and 10px
    wide), slight blur/softness, and stroke-weight differences."""
    variants = [gray]
    h, w = gray.shape[:2]

    for dw in (-3, -2, -1, 1, 2, 3):
        new_w = max(1, w + dw)
        variants.append(cv2.resize(gray, (new_w, h), interpolation=cv2.INTER_LINEAR))
    for dh in (-1, 1):
        new_h = max(1, h + dh)
        variants.append(cv2.resize(gray, (w, new_h), interpolation=cv2.INTER_LINEAR))

    variants.append(cv2.GaussianBlur(gray, (3, 3), 0.5))

    kernel = np.ones((2, 2), np.uint8)
    variants.append(cv2.dilate(gray, kernel, iterations=1))
    variants.append(cv2.erode(gray, kernel, iterations=1))

    return variants


def build_dataset(subdir, augmented=True):
    X, y, groups = [], [], []
    for gray, label, filepath in load_labeled_glyphs(subdir):
        samples = augment(gray) if augmented else [gray]
        for variant in samples:
            canon = _canonicalize(variant)
            X.append(hog_features(canon))
            y.append(label)
            groups.append(filepath)  # which real source glyph this came from
    return np.array(X), np.array(y), np.array(groups)


def train_and_save(subdir):
    X, y, groups = build_dataset(subdir)
    n_classes = len(set(y))
    print(f"[{subdir}] {X.shape[0]} samples ({len(set(groups))} real sources), {n_classes} classes")

    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    clf.fit(X, y)

    out_path = os.path.join("images", "digits", f"{subdir}_model.joblib")
    joblib.dump(clf, out_path)
    print(f"[{subdir}] saved -> {out_path}")
    return clf


def leave_one_source_out(subdir):
    """For each real (non-augmented) source glyph, train on everything else
    (augmented) and check whether the held-out real glyph is still
    classified correctly - the fair apples-to-apples comparison against the
    earlier template-matching leave-one-out audit."""
    items = load_labeled_glyphs(subdir)
    correct = 0
    failures = []
    for held_gray, held_label, held_path in items:
        X, y = [], []
        for gray, label, filepath in items:
            if filepath == held_path:
                continue
            for variant in augment(gray):
                canon = _canonicalize(variant)
                X.append(hog_features(canon))
                y.append(label)
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        clf.fit(np.array(X), np.array(y))

        held_canon = _canonicalize(held_gray)
        held_feat = hog_features(held_canon).reshape(1, -1)
        pred = clf.predict(held_feat)[0]
        proba = clf.predict_proba(held_feat)[0]
        top_prob = proba.max()

        ok = pred == held_label
        correct += ok
        if not ok:
            failures.append((held_path, held_label, pred, top_prob))

    print(f"[{subdir}] leave-one-source-out: {correct}/{len(items)} correct")
    for f in failures:
        print("  FAIL", f)
    return correct, len(items), failures


if __name__ == "__main__":
    for subdir in ("eq", "opt"):
        leave_one_source_out(subdir)
        print()

    for subdir in ("eq", "opt"):
        train_and_save(subdir)
