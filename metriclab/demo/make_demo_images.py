#!/usr/bin/env python3
"""Genera una coleccion de imagenes de prueba con la MISMA estructura que
Caltech-101: una carpeta por categoria con archivos de imagen dentro.

Para que sirve: permite probar todo el flujo (extraccion -> indice ->
clasificacion -> reporte) hoy mismo, sin depender de la descarga. Cuando se
tenga Caltech-101, basta con apuntar el extractor a su carpeta
101_ObjectCategories; no hay que cambiar nada mas.

ADVERTENCIA para el informe: estas imagenes son sinteticas y sus categorias
son faciles de distinguir. Los resultados que producen validan que el flujo
funciona, NO predicen el rendimiento sobre fotografias reales.

Las categorias tienen tamanos distintos a proposito, para reproducir el
desbalance de Caltech-101 y que la evaluacion estratificada se pruebe de
verdad.

Uso:  python3 make_demo_images.py salida/ [--seed 1]
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 128


def jitter(rgb, rng, amount=28):
    return tuple(int(max(0, min(255, c + rng.randint(-amount, amount)))) for c in rgb)


def background(rng, base, noise=18):
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    top = np.array(jitter(base, rng, 20), dtype=np.float32)
    bottom = np.array(jitter(base, rng, 20), dtype=np.float32) * rng.uniform(0.7, 1.1)
    for y in range(SIZE):
        t = y / (SIZE - 1)
        arr[y, :, :] = top * (1 - t) + bottom * t
    arr += np.random.default_rng(rng.randint(0, 10**9)).normal(0, noise, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def cat_sol(rng):
    img = background(rng, (70, 130, 210))
    d = ImageDraw.Draw(img)
    r = rng.randint(18, 34)
    cx, cy = rng.randint(r + 4, SIZE - r - 4), rng.randint(r + 4, SIZE // 2 + 10)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=jitter((250, 210, 40), rng))
    return img


def cat_hoja(rng):
    img = background(rng, (225, 220, 200))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(1, 3)):
        w, h = rng.randint(30, 60), rng.randint(14, 28)
        cx, cy = rng.randint(30, SIZE - 30), rng.randint(30, SIZE - 30)
        leaf = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(leaf).ellipse([0, 0, w - 1, h - 1], fill=jitter((50, 150, 50), rng) + (255,))
        leaf = leaf.rotate(rng.randint(0, 180), expand=True)
        img.paste(leaf, (cx - leaf.width // 2, cy - leaf.height // 2), leaf)
    return img


def cat_ladrillo(rng):
    img = background(rng, (200, 200, 200), noise=8)
    d = ImageDraw.Draw(img)
    bw, bh = rng.randint(22, 30), rng.randint(10, 14)
    color = (175, 65, 45)
    for row, y in enumerate(range(0, SIZE, bh + 2)):
        off = (bw // 2) if row % 2 else 0
        for x in range(-off, SIZE, bw + 2):
            d.rectangle([x, y, x + bw, y + bh], fill=jitter(color, rng, 22))
    return img


def cat_mar(rng):
    img = background(rng, (30, 90, 150), noise=10)
    d = ImageDraw.Draw(img)
    amp, freq = rng.uniform(3, 7), rng.uniform(0.08, 0.16)
    for y0 in range(rng.randint(5, 15), SIZE, rng.randint(10, 16)):
        pts = [(x, y0 + amp * math.sin(freq * x + rng.random() * 6)) for x in range(0, SIZE, 4)]
        d.line(pts, fill=jitter((190, 225, 245), rng, 15), width=2)
    return img


def cat_nieve(rng):
    img = background(rng, (215, 220, 230), noise=12)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(60, 140)):
        x, y, r = rng.randint(0, SIZE), rng.randint(0, SIZE), rng.randint(1, 3)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def cat_flor(rng):
    img = background(rng, (90, 150, 70))
    d = ImageDraw.Draw(img)
    cx, cy = rng.randint(35, SIZE - 35), rng.randint(35, SIZE - 35)
    pet = rng.randint(5, 8)
    pr = rng.randint(12, 20)
    color = jitter((235, 100, 170), rng)
    for k in range(pet):
        a = 2 * math.pi * k / pet
        px, py = cx + pr * math.cos(a), cy + pr * math.sin(a)
        d.ellipse([px - 10, py - 10, px + 10, py + 10], fill=color)
    d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=jitter((250, 220, 60), rng))
    return img


def cat_noche(rng):
    img = background(rng, (15, 18, 40), noise=6)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(25, 60)):
        x, y = rng.randint(0, SIZE), rng.randint(0, SIZE)
        d.point((x, y), fill=(250, 250, 220))
    if rng.random() < 0.7:
        r = rng.randint(8, 14)
        x, y = rng.randint(20, SIZE - 20), rng.randint(15, 50)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 235, 210))
    return img


def cat_arbol(rng):
    img = background(rng, (160, 200, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, SIZE - 22, SIZE, SIZE], fill=jitter((95, 140, 60), rng))
    for _ in range(rng.randint(1, 2)):
        x = rng.randint(25, SIZE - 25)
        tw = rng.randint(6, 10)
        d.rectangle([x - tw // 2, SIZE - 60, x + tw // 2, SIZE - 20], fill=jitter((110, 75, 40), rng))
        r = rng.randint(18, 26)
        d.ellipse([x - r, SIZE - 60 - 2 * r + 8, x + r, SIZE - 52], fill=jitter((40, 115, 45), rng))
    return img


# Tamanos desiguales a proposito (desbalance tipo Caltech-101).
CATEGORIES = {
    "sol": (cat_sol, 70),
    "hoja": (cat_hoja, 55),
    "ladrillo": (cat_ladrillo, 90),
    "mar": (cat_mar, 60),
    "nieve": (cat_nieve, 40),
    "flor": (cat_flor, 75),
    "noche": (cat_noche, 45),
    "arbol": (cat_arbol, 65),
}


def degrade(img, rng, other):
    """Variacion tipo fotografia real: cambio de tono e iluminacion, desenfoque
    y un trozo de OTRA categoria dentro de la imagen (fondo desordenado,
    objetos ajenos). Sin esto las categorias sinteticas se separan al 100 % y
    la comparacion entre algoritmos no muestra nada."""
    # 1) trozo de otra categoria (desorden / oclusion)
    if rng.random() < 0.6:
        s = rng.randint(SIZE // 3, SIZE // 2 + 10)
        x0, y0 = rng.randint(0, SIZE - s), rng.randint(0, SIZE - s)
        patch = other.crop((x0, y0, x0 + s, y0 + s))
        img = img.copy()
        img.paste(patch, (rng.randint(0, SIZE - s), rng.randint(0, SIZE - s)))
    # 2) tono e iluminacion
    hsv = np.array(img.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-22, 22)) % 256
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.6, 1.25), 0, 255)
    img = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    # 3) desenfoque
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--facil", action="store_true",
                    help="sin variacion realista (categorias trivialmente separables)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.out)
    names = list(CATEGORIES)
    total = 0
    for name, (fn, count) in CATEGORIES.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = fn(rng)
            if not args.facil:
                other_name = rng.choice([n for n in names if n != name])
                img = degrade(img, rng, CATEGORIES[other_name][0](rng))
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(folder / f"image_{i + 1:04d}.jpg", quality=90)
            total += 1
    print(f"{total} imagenes en {len(CATEGORIES)} categorias -> {root}")


if __name__ == "__main__":
    main()
