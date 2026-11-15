# Oracle data for the "ocr" recognizer "latin_v5" (PP-OCRv5 latin mobile, the community ONNX export
# monkt/paddleocr-onnx languages/latin, converted by convert_ocr.py). Draws German text with known answers
# (umlauts, ß, capitals) and saves for ocr_latin_test:
#   crop_<k>    a single text line as the recognizer sees it (height 48, Pillow BICUBIC like imgutils), RGB8
#   logits_<k>  onnxruntime's output on it ((x/255 - 0.5)/0.5, NCHW): the native graph must match it
#   page_<k>    a sign / page with several lines (RGB8) for the whole pipeline (v4 detector + recognizer)
# and in manifest.json the true texts and onnxruntime's greedy CTC text (repeats merged, PaddleOCR's
# CTCLabelDecode). Written with AI assistance (Claude Code).
#
# Run in a venv with onnxruntime, numpy, pillow, safetensors (python -I), Windows fonts or DejaVu:
#   python -I make_ocr_latin_oracle.py <languages/latin dir with rec.onnx + dict.txt> <out_dir>
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from safetensors.numpy import save_file

LINES = [
    "Bahnhofstraße 12", "Größe: 1,85 m", "Übergröße und Ärger", "Fußgängerzone München",
    "Grüße aus Köln", "Schöne Grüße, Ihr Jürgen", "Die Straße ist gesperrt.", "Weißwurst mit süßem Senf",
    "Äpfel, Birnen und Pflaumen", "Ökostrom für alle Haushalte", "Überweisung bis zum 15. März",
    "Dr. med. Anna Müller", "Gemäß § 3 Abs. 2", "Fröhliche Weihnachten!", "Prüfung bestanden",
    "Maßnahmen zur Lärmminderung", "Rathausplatz 7, 50667 Köln", "Kühlschrank 249,99 €",
    "Zugspitze 2962 m ü. NN", "Bäckerei Schäfer", "ÄRZTEHAUS AM SCHLOSS", "GRÜNE WIESE",
    "ÖFFENTLICHE TOILETTE", "ÜBERHOLVERBOT", "Das Mädchen läuft über die Brücke.", "Er aß Spätzle mit Soße.",
    "Fährhafen Kiel", "Wir schließen um 20 Uhr.", "Gültig bis 31.12.2026", "Notausgang",
]
PAGES = [
    ["Bäckerei Schäfer", "Täglich frische Brötchen", "Öffnungszeiten 7 bis 18 Uhr"],
    ["Fußgängerzone", "Größere Straßenbahn", "Übergang gesperrt"],
    ["Dr. med. Jürgen Weiß", "Facharzt für Hals, Nase, Ohren", "Sprechstunde nach Vereinbarung"],
    ["ÄRZTEHAUS AM SCHLOSS", "Müllerstraße 5", "Grüße aus Düsseldorf"],
]
FONT_NAMES = ["arial.ttf", "times.ttf", "calibri.ttf", "verdana.ttf", "georgia.ttf", "segoeui.ttf"]


def font(index, size):
    for name in (FONT_NAMES[index % len(FONT_NAMES)], "arial.ttf", "DejaVuSans.ttf"):
        for base in ("C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/", ""):
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                pass
    raise SystemExit("no TrueType font found")


def line_image(text, k):
    size = [22, 30, 40][k % 3]
    f = font(k, size)
    x0, y0, x1, y1 = f.getbbox(text)
    pad = max(4, size // 4)
    dark = k % 5 == 0
    fg, bg = ((235, 235, 235), (30, 40, 60)) if dark else ((10, 10, 10), (250, 248, 240))
    im = Image.new("RGB", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), bg)
    ImageDraw.Draw(im).text((pad - x0, pad - y0), text, font=f, fill=fg)
    if k % 4 == 3:
        im = im.filter(ImageFilter.GaussianBlur(0.6))
    r = 48 / im.height
    return im.resize((int(round(im.width * r)), int(round(im.height * r))), Image.BICUBIC)


def page_image(lines, k):
    im = Image.new("RGB", (900, 420), [(250, 250, 250), (20, 60, 30), (245, 240, 220), (30, 30, 90)][k % 4])
    fg = (20, 20, 20) if k % 2 == 0 else (240, 240, 240)
    d = ImageDraw.Draw(im)
    y = 40
    for i, text in enumerate(lines):
        f = font(k + i, [46, 36, 30][i % 3])
        d.text((50 + 20 * i, y), text, font=f, fill=fg)
        y += 120
    return im


def decode(logits, vocab):
    idx = logits.argmax(-1)[0]
    out, prev = [], -1
    for k in idx:
        if k != 0 and k != prev:
            out.append(vocab[k])
        prev = k
    return "".join(out)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_ocr_latin_oracle.py <latin dir> <out_dir>")
    import onnxruntime as ort
    src, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(src, "dict.txt"), encoding="utf-8") as f:
        vocab = ["<blank>"] + [line.strip() for line in f] + [" "]
    sess = ort.InferenceSession(os.path.join(src, "rec.onnx"), providers=["CPUExecutionProvider"])
    tensors, lines = {}, []
    for k, text in enumerate(LINES):
        im = line_image(text, k)
        rgb = np.asarray(im, dtype=np.uint8)
        x = ((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        y = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x)})[0]
        assert y.shape[-1] == len(vocab), (y.shape, len(vocab))
        tensors[f"crop_{k}"] = np.ascontiguousarray(rgb)
        tensors[f"logits_{k}"] = np.ascontiguousarray(y.astype(np.float32))
        lines.append({"text": text, "ort_text": decode(y, vocab).strip()})
    pages = []
    for k, page in enumerate(PAGES):
        tensors[f"page_{k}"] = np.ascontiguousarray(np.asarray(page_image(page, k), dtype=np.uint8))
        pages.append({"lines": page})
    save_file(tensors, os.path.join(out, "ocr_latin.safetensors"))
    manifest = {"source": "monkt/paddleocr-onnx languages/latin", "file": "ocr_latin.safetensors",
                "lines": lines, "pages": pages}
    with open(os.path.join(out, "manifest.json"), "wb") as f:
        f.write((json.dumps(manifest, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    right = sum(e["text"] == e["ort_text"] for e in lines)
    print(f"{len(lines)} lines ({right} read right by onnxruntime), {len(pages)} pages -> {out}")


if __name__ == "__main__":
    main()
