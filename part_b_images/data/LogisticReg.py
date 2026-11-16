
import sys
import time
import numpy as np

IMG = 64
N_CLASSES = 5
SEED = 42



def load_flowers(img_size=IMG):
    import tensorflow_datasets as tfds
    import tensorflow as tf

    ds, info = tfds.load("tf_flowers", split="train", as_supervised=True, with_info=True)
    names = info.features["label"].names          # real label order from the dataset
    images, labels = [], []
    for img, lab in tfds.as_numpy(ds):
        img = tf.image.resize(img, (img_size, img_size), antialias=True).numpy()
        images.append(np.clip(img, 0, 255).astype(np.uint8))
        labels.append(int(lab))
    return np.stack(images), np.array(labels), names


def load_synthetic(n=1500, img_size=IMG, seed=0):
    """Fake data: each class has a different dominant color. Only for testing the code."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    base = rng.uniform(40, 215, (N_CLASSES, 3))
    X = base[y][:, None, None, :] + rng.normal(0, 40, (n, img_size, img_size, 3))
    return np.clip(X, 0, 255).astype(np.uint8), y, [f"class{i}" for i in range(N_CLASSES)]


def split_data(X, y, seed=SEED, frac=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_tr = int(frac[0] * len(X))
    n_va = int(frac[1] * len(X))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def augment_flip(X, y):
    """Add horizontally flipped copies (doubles the training set)."""
    return np.concatenate([X, X[:, :, ::-1, :]]), np.concatenate([y, y])



def rgb_to_hsv(img):
    """img: (...,3) float in [0,1] -> h,s,v each in [0,1]."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = img.max(axis=-1)
    minc = img.min(axis=-1)
    delta = maxc - minc
    safe = np.where(delta == 0, 1.0, delta)

    h = np.zeros_like(maxc)
    h = np.where(maxc == r, ((g - b) / safe) % 6, h)
    h = np.where(maxc == g, (b - r) / safe + 2, h)
    h = np.where(maxc == b, (r - g) / safe + 4, h)
    h = np.where(delta == 0, 0.0, h) / 6.0
    s = np.where(maxc == 0, 0.0, delta / np.where(maxc == 0, 1.0, maxc))
    return h, s, maxc


def hsv_joint_hist(h, s, v, hb=8, sb=3, vb=3):
    hi = np.minimum((h * hb).astype(int), hb - 1)
    si = np.minimum((s * sb).astype(int), sb - 1)
    vi = np.minimum((v * vb).astype(int), vb - 1)
    idx = (hi * sb + si) * vb + vi
    hist = np.bincount(idx.ravel(), minlength=hb * sb * vb).astype(np.float64)
    return hist / hist.sum()


def color_features(img):
    """Joint HSV histogram for the whole image + 4 quadrants + center."""
    f = img.astype(np.float64) / 255.0
    h, s, v = rgb_to_hsv(f)
    H, W = h.shape
    hh, hw = H // 2, W // 2
    regions = [
        (slice(0, H), slice(0, W)),            # whole image
        (slice(0, hh), slice(0, hw)),          # top-left
        (slice(0, hh), slice(hw, W)),          # top-right
        (slice(hh, H), slice(0, hw)),          # bottom-left
        (slice(hh, H), slice(hw, W)),          # bottom-right
        (slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4)),  # center
    ]
    feats = [hsv_joint_hist(h[r, c], s[r, c], v[r, c]) for r, c in regions]
    out = np.concatenate(feats)
    return np.sqrt(out)  # Hellinger / sqrt trick: makes histograms friendlier for linear models


def hog_features(img, cell=16, bins=9):
    """Simplified HOG: unsigned gradient orientation histogram per cell."""
    g = img.astype(np.float64) / 255.0
    gray = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)               # [0, pi)
    bidx = np.minimum((ang / np.pi * bins).astype(int), bins - 1)
    H, W = gray.shape
    feats = []
    for i in range(0, H - cell + 1, cell):
        for j in range(0, W - cell + 1, cell):
            hist = np.bincount(bidx[i:i + cell, j:j + cell].ravel(),
                               weights=mag[i:i + cell, j:j + cell].ravel(),
                               minlength=bins)
            feats.append(hist / (np.linalg.norm(hist) + 1e-6))
    return np.concatenate(feats)


def extract_features(images):
    return np.stack([np.concatenate([color_features(im), hog_features(im)]) for im in images])


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu, sd



def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_fn(X, Y, W, b, l2):
    P = softmax(X @ W + b)
    ce = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
    return ce + 0.5 * l2 * np.sum(W * W)


def predict(X, W, b):
    return np.argmax(X @ W + b, axis=1)


def accuracy(X, y, W, b):
    return float((predict(X, W, b) == y).mean())


def gradient_check(seed=0):
    """Compare analytic gradient with numerical gradient on a tiny problem."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(8, 5)); y = rng.integers(0, 3, 8)
    Y = np.eye(3)[y]; W = rng.normal(size=(5, 3)) * 0.1; b = np.zeros(3); l2 = 0.01
    P = softmax(X @ W + b)
    gW = X.T @ (P - Y) / len(X) + l2 * W
    num = np.zeros_like(W); eps = 1e-5
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += eps
            Wm = W.copy(); Wm[i, j] -= eps
            num[i, j] = (loss_fn(X, Y, Wp, b, l2) - loss_fn(X, Y, Wm, b, l2)) / (2 * eps)
    err = np.abs(gW - num).max()
    print(f"gradient check max error: {err:.2e}  ({'OK' if err < 1e-6 else 'PROBLEM'})")


def train_softmax(Xtr, ytr, Xva, yva, lr=0.05, l2=1e-3, epochs=60, batch=128,
                  momentum=0.9, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    W = rng.normal(0, 0.01, (d, N_CLASSES)); b = np.zeros(N_CLASSES)
    vW = np.zeros_like(W); vb = np.zeros_like(b)
    Ytr = np.eye(N_CLASSES)[ytr]

    best = {"acc": -1, "W": W.copy(), "b": b.copy(), "epoch": 0}
    hist = {"loss": [], "train_acc": [], "val_acc": []}

    for ep in range(epochs):
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))   # cosine decay
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            P = softmax(Xtr[j] @ W + b)
            G = (P - Ytr[j]) / len(j)
            gW = Xtr[j].T @ G + l2 * W
            gb = G.sum(axis=0)
            vW = momentum * vW - lr_t * gW
            vb = momentum * vb - lr_t * gb
            W += vW
            b += vb

        tr_loss = loss_fn(Xtr, Ytr, W, b, l2)
        tr_acc = accuracy(Xtr, ytr, W, b)
        va_acc = accuracy(Xva, yva, W, b)
        hist["loss"].append(tr_loss); hist["train_acc"].append(tr_acc); hist["val_acc"].append(va_acc)
        if va_acc > best["acc"]:
            best = {"acc": va_acc, "W": W.copy(), "b": b.copy(), "epoch": ep}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} | loss {tr_loss:.4f} | train {tr_acc:.3f} | val {va_acc:.3f}")
    return best, hist



def confusion_matrix(y_true, y_pred, k=N_CLASSES):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm



def main(synthetic=False):
    np.random.seed(SEED)
    t0 = time.time()

    print("Loading data...")
    X, y, names = load_synthetic() if synthetic else load_flowers()
    print(f"  images: {X.shape}, classes: {np.bincount(y)}")

    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_data(X, y)
    Xtr, ytr = augment_flip(Xtr, ytr)
    print(f"  train (with flip): {len(Xtr)} | val: {len(Xva)} | test: {len(Xte)}")

    print("Extracting features...")
    Ftr, Fva, Fte = extract_features(Xtr), extract_features(Xva), extract_features(Xte)
    print(f"  feature dim: {Ftr.shape[1]}  ({time.time() - t0:.1f}s)")

    mu, sd = standardize_fit(Ftr)          # statistics from TRAIN only
    Ftr, Fva, Fte = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd

    gradient_check()

    print("Grid search on validation set...")
    best_cfg, best_run = None, None
    for lr in (0.02, 0.05, 0.1):
        for l2 in (1e-4, 1e-3, 1e-2):
            run, _ = train_softmax(Ftr, ytr, Fva, yva, lr=lr, l2=l2, epochs=40)
            print(f"  lr={lr:<5} l2={l2:<7} -> best val acc {run['acc']:.3f}")
            if best_run is None or run["acc"] > best_run["acc"]:
                best_cfg, best_run = (lr, l2), run

    lr, l2 = best_cfg
    print(f"\nBest config: lr={lr}, l2={l2}. Retraining longer...")
    final, hist = train_softmax(Ftr, ytr, Fva, yva, lr=lr, l2=l2, epochs=120, verbose=True)
    W, b = final["W"], final["b"]

    # ---- final test evaluation (done once) ----
    test_acc = accuracy(Fte, yte, W, b)
    print(f"\nBest val acc: {final['acc']:.3f} (epoch {final['epoch']})")
    print(f"TEST accuracy: {test_acc:.3f}")

    cm = confusion_matrix(yte, predict(Fte, W, b))
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("            " + " ".join(f"{n[:6]:>7}" for n in names))
    for n, row in zip(names, cm):
        print(f"{n:>11} " + " ".join(f"{v:7d}" for v in row))
    print("\nPer-class recall:")
    for i, (n, row) in enumerate(zip(names, cm)):
        print(f"  {n:<11} {row[i] / max(row.sum(), 1):.3f}")

    np.savez("softmax_flowers_model.npz", W=W, b=b, mu=mu, sd=sd)
    print(f"\nSaved model to softmax_flowers_model.npz  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(synthetic="--synthetic" in sys.argv)