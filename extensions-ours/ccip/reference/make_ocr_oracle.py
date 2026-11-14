# Oracle data for the /v1/vision/detect "ocr" task: runs dghs-imgutils itself (imgutils.ocr.ocr with its
# defaults, the film pipeline's reference) and saves per image everything ocr_oracle_test compares:
#   rgb        decoded RGB8 pixels
#   det_input  the detector input (padded to 64, CLIP-normalised)
#   heat       the onnxruntime heat map cropped to the image (post-processing is checked on it alone)
#   crop_<k>   the recognizer input image of detector box k (crop, 90 degree turn, height 48)
# and in manifest.json the detector boxes (_get_text_points: corners + score) and the ocr() results
# (rectangle, text, score), plus imgutils' own time per image (models loaded before timing).
# Written with AI assistance (Claude Code).
#
# Run in a venv with dghs-imgutils 0.19.0 (+ onnxruntime, safetensors). The ONNX files are read from a local
# copy of deepghs/paddleocr (--paddleocr <dir with det/ and rec/>) or from the HF cache:
#   python -I make_ocr_oracle.py <out_dir> --paddleocr <dir> --render            built-in test images
#   python -I make_ocr_oracle.py <out_dir> --paddleocr <dir> k1.png k2.png ...   own keyframes
# --render draws German and English name plates, signs and screen texts with PIL (fonts: Arial/Segoe UI
# on Windows, DejaVu elsewhere) and stores them as PNG next to the manifest.
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from safetensors.numpy import save_file

import imgutils
import imgutils.ocr.detect as ocr_detect
import imgutils.ocr.recognize as ocr_recognize
from imgutils.data import load_image
from imgutils.ocr import ocr

DET = "ch_PP-OCRv4_det"
REC = "ch_PP-OCRv4_rec"


def use_local_files(root):
    def download(repo, filename, **_):
        assert repo == "deepghs/paddleocr", repo
        path = os.path.join(root, *filename.split("/"))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path

    ocr_detect.hf_hub_download = download
    ocr_recognize.hf_hub_download = download


def font(size, bold=False):
    names = (["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"])
    for name in names:
        for d in (os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
                  "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def mono(size):
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        for d in (os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
                  "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return ImageFont.truetype(p, size)
    return font(size)


def noise_background(w, h, base, seed, amount=18):
    rng = np.random.default_rng(seed)
    arr = np.empty((h, w, 3), dtype=np.int16)
    arr[:] = np.array(base, dtype=np.int16)
    yy = np.linspace(-1, 1, h)[:, None, None]
    arr = arr + (yy * 25).astype(np.int16) + rng.integers(-amount, amount + 1, size=(h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def paste_rotated(canvas, layer, angle, pos):
    rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
    canvas.paste(rotated, pos, rotated)


def render_images(out_dir):
    paths = []

    # 1. German door name plate on a wall, 1344x768.
    img = noise_background(1344, 768, (182, 170, 150), 1)
    plate = Image.new("RGBA", (620, 260), (0, 0, 0, 0))
    d = ImageDraw.Draw(plate)
    d.rounded_rectangle((0, 0, 619, 259), 18, fill=(214, 186, 112, 255), outline=(120, 96, 40, 255), width=6)
    d.text((40, 30), "Dr. med. Anna Müller", font=font(46, True), fill=(30, 26, 20, 255))
    d.text((40, 100), "Praxis für Allgemeinmedizin", font=font(34), fill=(40, 34, 26, 255))
    d.text((40, 160), "Sprechzeiten: Mo-Fr 8-12 Uhr", font=font(30), fill=(40, 34, 26, 255))
    paste_rotated(img, plate, 0, (360, 220))
    paths.append(("nameplate_de.png", img))

    # 2. English conference name tag, slightly rotated, 960x540.
    img = noise_background(960, 540, (60, 90, 140), 2)
    tag = Image.new("RGBA", (420, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(tag)
    d.rectangle((0, 0, 419, 299), fill=(250, 250, 250, 255))
    d.rectangle((0, 0, 419, 90), fill=(200, 30, 40, 255))
    d.text((40, 12), "HELLO", font=font(48, True), fill=(255, 255, 255, 255))
    d.text((210, 40), "my name is", font=font(28), fill=(255, 255, 255, 255))
    d.text((40, 150), "John Smith", font=font(54, True), fill=(20, 20, 20, 255))
    d.text((40, 230), "Sales Manager", font=font(28), fill=(80, 80, 80, 255))
    paste_rotated(img, tag, 7, (250, 90))
    paths.append(("nametag_en.png", img))

    # 3. Terminal screen with status lines, 1344x768.
    img = Image.new("RGB", (1344, 768), (12, 16, 20))
    d = ImageDraw.Draw(img)
    lines = [
        ("ERROR 404: Page not found", (255, 90, 90)),
        ("System status: OK", (120, 230, 120)),
        ("Download 75% complete", (220, 220, 220)),
        ("Datei wird geladen, bitte warten", (220, 220, 220)),
        ("user@server:~$ ls -la /var/log", (200, 200, 120)),
    ]
    for i, (text, color) in enumerate(lines):
        d.text((60, 60 + i * 70), text, font=mono(36), fill=color)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    paths.append(("screen_terminal.png", img))

    # 4. Street and exit signs, rotated, odd size 1000x563.
    img = noise_background(1000, 563, (110, 120, 105), 4, 25)
    sign = Image.new("RGBA", (480, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(sign)
    d.rectangle((0, 0, 479, 119), fill=(20, 60, 150, 255), outline=(255, 255, 255, 255), width=5)
    d.text((30, 25), "Bahnhofstraße 12", font=font(56, True), fill=(255, 255, 255, 255))
    paste_rotated(img, sign, -12, (60, 70))
    exit_sign = Image.new("RGBA", (300, 110), (0, 0, 0, 0))
    d = ImageDraw.Draw(exit_sign)
    d.rectangle((0, 0, 299, 109), fill=(10, 140, 60, 255))
    d.text((40, 18), "EXIT", font=font(64, True), fill=(255, 255, 255, 255))
    paste_rotated(img, exit_sign, 18, (600, 330))
    d = ImageDraw.Draw(img)
    d.text((80, 470), "Welcome to Berlin", font=font(40), fill=(250, 250, 240))
    paths.append(("signs_rotated.png", img))

    # 5. Vertical banner (tall boxes: the 90 degree path), 640x480.
    img = noise_background(640, 480, (230, 225, 210), 5, 10)
    banner = Image.new("RGBA", (420, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(banner)
    d.rectangle((0, 0, 419, 89), fill=(240, 200, 40, 255))
    d.text((24, 14), "OPEN 24 HOURS", font=font(52, True), fill=(20, 20, 20, 255))
    paste_rotated(img, banner, 90, (40, 30))
    d = ImageDraw.Draw(img)
    d.text((200, 60), "Kiosk am Markt", font=font(44, True), fill=(30, 30, 90))
    d.text((200, 140), "Zeitungen, Kaffee, Snacks", font=font(30), fill=(30, 30, 30))
    d.text((200, 400), "Tel. 030 1234567", font=font(30), fill=(60, 30, 30))
    paths.append(("banner_vertical.png", img))

    # 6. Small TV caption and a monitor with UI labels, 1280x720.
    img = noise_background(1280, 720, (40, 44, 52), 6, 12)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 600, 1279, 719), fill=(0, 0, 0))
    d.text((40, 625), "BREAKING NEWS: Storm warning for the coast", font=font(44, True), fill=(255, 255, 255))
    d.rectangle((700, 80, 1220, 420), fill=(235, 238, 242), outline=(90, 90, 90), width=4)
    d.text((730, 110), "Login", font=font(40, True), fill=(20, 20, 20))
    d.text((730, 190), "Benutzername", font=font(28), fill=(50, 50, 50))
    d.text((730, 260), "Passwort vergessen?", font=font(28), fill=(30, 80, 200))
    d.text((730, 340), "Anmelden", font=font(30, True), fill=(20, 120, 40))
    d.text((60, 120), "Kapitel 3", font=font(60, True), fill=(250, 220, 120))
    d.text((60, 210), "Die Rückkehr", font=font(48), fill=(240, 240, 240))
    paths.append(("tv_screen.png", img))

    out = []
    for name, im in paths:
        p = os.path.join(out_dir, name)
        im.save(p)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--paddleocr", help="local copy of deepghs/paddleocr (det/, rec/)")
    ap.add_argument("--render", action="store_true", help="render the built-in test images")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.paddleocr:
        use_local_files(args.paddleocr)
    images = list(args.images)
    if args.render:
        images += render_images(args.out_dir)
    if not images:
        raise SystemExit("no images (give paths or --render)")
    try:
        from imgutils.config.meta import __VERSION__ as version
    except ImportError:
        version = getattr(imgutils, "__version__", None)
    manifest = {"imgutils": version, "det_model": DET, "rec_model": REC, "images": []}
    session = ocr_detect._open_ocr_detection_model(DET)
    ocr(Image.new("RGB", (64, 64), (255, 255, 255)))  # load both models before timing
    for i, path in enumerate(images):
        img = load_image(path, mode="RGB")
        rgb = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        # Detector input and heat map exactly as ocr/detect.py _get_text_points builds them.
        width, height = img.size
        pw = width + (-width) % 64
        ph = height + (-height) % 64
        x = np.array(img).transpose((2, 0, 1)).astype(np.float32) / 255.0
        x = np.pad(x[None, ...], ((0, 0), (0, 0), (0, ph - height), (0, pw - width)))
        x = ocr_detect._normalize(x).astype(np.float32)
        heat = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: x})[0][0][0]
        heat = np.ascontiguousarray(heat[:height, :width])
        tensors = {"rgb": rgb, "det_input": np.ascontiguousarray(x[0]), "heat": heat}
        boxes = []
        for k, (points, score) in enumerate(ocr_detect._get_text_points(img, DET)):
            x0, y0 = int(points[:, 0].min()), int(points[:, 1].min())
            x1, y1 = int(points[:, 0].max()), int(points[:, 1].max())
            boxes.append({"points": [[int(a), int(b)] for a, b in points], "score": float(score),
                          "box": [x0, y0, x1, y1]})
            area = img.crop((x0, y0, x1, y1))
            if y1 - y0 >= (x1 - x0) * 1.5:
                area = area.rotate(90)
            if area.width > 0 and area.height > 0:
                r = 48 / area.height
                area = area.resize((int(round(area.width * r)), int(round(area.height * r))))
                tensors[f"crop_{k}"] = np.ascontiguousarray(np.asarray(area, dtype=np.uint8))
        t0 = time.perf_counter()
        results = ocr(img)
        seconds = time.perf_counter() - t0
        name = f"ocr_{i}.safetensors"
        save_file(tensors, os.path.join(args.out_dir, name))
        manifest["images"].append({
            "source": os.path.basename(path), "file": name, "width": width, "height": height,
            "imgutils_seconds": seconds, "boxes": boxes,
            "ocr": [{"box": [int(v) for v in box], "text": text, "score": float(score)}
                    for box, text, score in results],
        })
        print(os.path.basename(path), f"{len(boxes)} boxes, {seconds:.3f} s:",
              [(t["text"], round(t["score"], 3)) for t in manifest["images"][-1]["ocr"]])
    data = json.dumps(manifest, indent=1, ensure_ascii=False).encode("utf-8")
    with open(os.path.join(args.out_dir, "manifest.json"), "wb") as f:
        f.write(data)


if __name__ == "__main__":
    main()
