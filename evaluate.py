import os
from app import (
    model_image_text,
    model_video_deepfake,
    model_face_image,
    model_voice_clone,
    ALLOWED_IMAGE_EXT,
    ALLOWED_DOC_EXT,
    ALLOWED_VIDEO_EXT,
    ALLOWED_AUDIO_EXT,
)
from sklearn.metrics import classification_report

def iter_files(root, allowed_ext):
    for fname in os.listdir(root):
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            continue
        ext = fname.rsplit(".", 1)[-1].lower()
        if ext in allowed_ext:
            yield path, ext

def bin_label_from_verdict(verdict: str) -> int:
    """
    Map model verdict -> binary label:
    0 = real, 1 = manipulated/suspicious
    """
    if verdict is None:
        return 1
    v = verdict.lower()
    if v == "real":
        return 0
    # treat suspicious/manipulated/inconclusive as manipulated-ish
    return 1

def eval_image_text(root="data/image_text"):
    print("=== Evaluating IMAGE TEXT detector ===")
    y_true = []
    y_pred = []

    for cls_name, true_label in [("real", 0), ("manipulated", 1)]:
        folder = os.path.join(root, cls_name)
        if not os.path.isdir(folder):
            continue

        for path, ext in iter_files(folder, ALLOWED_IMAGE_EXT | ALLOWED_DOC_EXT):
            file_type = "image" if ext in ALLOWED_IMAGE_EXT else "document"
            res = model_image_text(path, file_type)
            if not res.get("success"):
                print("  [skip] error on", path, "->", res.get("error"))
                continue
            v = res.get("verdict")
            pred_label = bin_label_from_verdict(v)

            y_true.append(true_label)
            y_pred.append(pred_label)

    return y_true, y_pred

def eval_video(root="data/video"):
    print("=== Evaluating VIDEO detector ===")
    y_true, y_pred = [], []

    for cls_name, true_label in [("real", 0), ("manipulated", 1)]:
        folder = os.path.join(root, cls_name)
        if not os.path.isdir(folder):
            continue

        for path, ext in iter_files(folder, ALLOWED_VIDEO_EXT):
            res = model_video_deepfake(path)
            if not res.get("success"):
                print("  [skip] error on", path, "->", res.get("error"))
                continue
            pred_label = bin_label_from_verdict(res.get("verdict"))
            y_true.append(true_label)
            y_pred.append(pred_label)

    return y_true, y_pred

def eval_face(root="data/face_image"):
    print("=== Evaluating FACE IMAGE detector ===")
    y_true, y_pred = [], []

    for cls_name, true_label in [("real", 0), ("manipulated", 1)]:
        folder = os.path.join(root, cls_name)
        if not os.path.isdir(folder):
            continue

        for path, ext in iter_files(folder, ALLOWED_IMAGE_EXT):
            res = model_face_image(path)
            if not res.get("success"):
                print("  [skip] error on", path, "->", res.get("error"))
                continue
            pred_label = bin_label_from_verdict(res.get("verdict"))
            y_true.append(true_label)
            y_pred.append(pred_label)

    return y_true, y_pred

def eval_voice(root="data/voice"):
    print("=== Evaluating VOICE detector ===")
    y_true, y_pred = [], []

    for cls_name, true_label in [("real", 0), ("manipulated", 1)]:
        folder = os.path.join(root, cls_name)
        if not os.path.isdir(folder):
            continue

        for path, ext in iter_files(folder, ALLOWED_AUDIO_EXT):
            res = model_voice_clone(path)
            if not res.get("success"):
                print("  [skip] error on", path, "->", res.get("error"))
                continue
            pred_label = bin_label_from_verdict(res.get("verdict"))
            y_true.append(true_label)
            y_pred.append(pred_label)

    return y_true, y_pred

def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)

if __name__ == "__main__":
    if __name__ == "__main__":
        for name, fn in [
            ("Image Text", eval_image_text),
            ("Video", eval_video),
            ("Face", eval_face),
            ("Voice", eval_voice),
        ]:
            y_true, y_pred = fn()
            acc = accuracy(y_true, y_pred)
            print(f"\n===== {name} =====")
            print(f"Accuracy: {acc:.2%}  (n={len(y_true)})")

            # ---- ADD THIS BLOCK HERE ----
            if len(y_true) > 0:
                from sklearn.metrics import classification_report
                print(classification_report(
                    y_true, y_pred,
                    target_names=["real", "manipulated"]
                ))
            # -----------------------------
