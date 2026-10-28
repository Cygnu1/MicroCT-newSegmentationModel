#!/usr/bin/env python3
"""Normaliza volumes TIFF inteiros em uint16 usando percentis por amostra, sem máscaras.

Dependências: pip install numpy tifffile
Uso: edite a seção CONFIGURAÇÃO e execute python normalizar_duas_amostras.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

# ===================== CONFIGURAÇÃO =====================
AMOSTRAS = {
    "P4_post": Path("D:\Alan\Doutorado\Rochas\data\MC3_1_P4\post"),
    "P4_pre": Path("D:\Alan\Doutorado\Rochas\data\MC3_1_P4\pre"),
}
PASTA_SAIDA = Path("C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data\\mc3_1_P4\\post\\normalized")
NORMALIZATION_MODE = "p2_p98"  # p1_p99 | p2_p98 | p05_p995 | p5_p95
MODOS = {
    "p1_p99": (1.0, 99.0),
    "p2_p98": (2.0, 98.0),
    "p05_p995": (0.5, 99.5),
    "p5_p95": (5.0, 95.0),
}
VALOR_MINIMO_VALIDO = 0  # percentis calculados apenas onde I > 0
PRESERVAR_FUNDO_ZERO = True
SOBRESCREVER = False
ESCALA_UINT16 = 65535
# =======================================================


def listar_tiffs(pasta: Path) -> list[Path]:
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
    arquivos = sorted(p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    if not arquivos:
        raise ValueError(f"Nenhum TIFF encontrado em: {pasta}")
    return arquivos


def obter_imagem(caminho: Path) -> np.ndarray:
    img = tifffile.imread(caminho)
    if img.ndim != 2:
        raise ValueError(f"Esperada fatia TIFF 2D em {caminho}, recebida shape={img.shape}")
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError(f"Esperado TIFF inteiro (preferencialmente uint16): {caminho} ({img.dtype})")
    if np.any(img < 0):
        raise ValueError(f"Intensidades negativas em {caminho}; ajuste a estratégia de histograma.")
    if img.max() > 65535:
        raise ValueError(f"Intensidade acima de 65535 em {caminho}; ajuste a estratégia de histograma.")
    return img


def calcular_histograma(arquivos: list[Path]) -> tuple[np.ndarray, int]:
    """Histograma inteiro de 16 bits: ~512 KiB, independente do volume."""
    hist = np.zeros(65536, dtype=np.int64)
    total = 0
    for indice, caminho in enumerate(arquivos, 1):
        img = obter_imagem(caminho)
        validos = img[img > VALOR_MINIMO_VALIDO]
        if validos.size:
            hist += np.bincount(validos.astype(np.int32), minlength=65536)
            total += int(validos.size)
        if indice % 100 == 0 or indice == len(arquivos):
            print(f"    Histograma: {indice}/{len(arquivos)} fatias")
    if total == 0:
        raise ValueError("A amostra não contém pixels válidos (> 0).")
    return hist, total


def percentil_exato_histograma(hist: np.ndarray, total: int, percentil: float) -> float:
    """Reproduz interpolação linear padrão de np.percentile para dados inteiros."""
    acumulado = np.cumsum(hist)
    posicao = (total - 1) * (percentil / 100.0)
    indice_inferior = int(np.floor(posicao))
    indice_superior = int(np.ceil(posicao))
    valor_inferior = int(np.searchsorted(acumulado, indice_inferior + 1, side="left"))
    valor_superior = int(np.searchsorted(acumulado, indice_superior + 1, side="left"))
    return float(valor_inferior + (posicao - indice_inferior) * (valor_superior - valor_inferior))


def salvar_amostra(nome: str, arquivos: list[Path], saida: Path, minimo: float, maximo: float) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    for indice, origem in enumerate(arquivos, 1):
        destino = saida / origem.name
        if destino.exists() and not SOBRESCREVER:
            raise FileExistsError(f"Arquivo já existe (SOBRESCREVER=False): {destino}")
        original = obter_imagem(origem)
        img = original.astype(np.float32)
        img = np.clip((img - minimo) / (maximo - minimo), 0.0, 1.0)
        img = np.rint(img * ESCALA_UINT16).astype(np.uint16)
        if PRESERVAR_FUNDO_ZERO:
            img[original <= VALOR_MINIMO_VALIDO] = 0
        tifffile.imwrite(destino, img, photometric="minisblack")
        if indice % 100 == 0 or indice == len(arquivos):
            print(f"    Salvando: {indice}/{len(arquivos)} fatias")


def main() -> None:
    if NORMALIZATION_MODE not in MODOS:
        raise ValueError(f"Modo inválido: {NORMALIZATION_MODE}. Opções: {list(MODOS)}")
    if len(AMOSTRAS) != 2:
        raise ValueError("Configure exatamente duas amostras em AMOSTRAS.")
    pbaixo, palto = MODOS[NORMALIZATION_MODE]
    raiz = PASTA_SAIDA / NORMALIZATION_MODE
    raiz.mkdir(parents=True, exist_ok=True)
    resultados = {
        "normalization_mode": NORMALIZATION_MODE,
        "percentiles": {"low": pbaixo, "high": palto},
        "scope": "per_volume_independent",
        "valid_pixel_rule": f"I > {VALOR_MINIMO_VALIDO}",
        "output_dtype": "uint16",
        "output_range": [0, 65535],
        "decode_to_float01": "image.astype(np.float32) / 65535.0",
        "quantization": "round(clip((I - clip_min) / (clip_max - clip_min), 0, 1) * 65535)",
        "preserve_background_zero": PRESERVAR_FUNDO_ZERO,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples": {},
    }
    for nome, pasta in AMOSTRAS.items():
        print(f"\n[{nome}] Lendo: {pasta}")
        arquivos = listar_tiffs(pasta)
        hist, total = calcular_histograma(arquivos)
        minimo = percentil_exato_histograma(hist, total, pbaixo)
        maximo = percentil_exato_histograma(hist, total, palto)
        if maximo <= minimo:
            raise ValueError(f"Intervalo inválido para {nome}: [{minimo}, {maximo}]")
        destino = raiz / nome
        print(f"[{nome}] {len(arquivos)} fatias, {total} pixels válidos; P{pbaixo}={minimo:.3f}, P{palto}={maximo:.3f}")
        salvar_amostra(nome, arquivos, destino, minimo, maximo)
        resultados["samples"][nome] = {
            "source_dir": str(pasta.resolve()),
            "output_dir": str(destino.resolve()),
            "num_slices": len(arquivos),
            "num_valid_pixels": total,
            "clip_min": minimo,
            "clip_max": maximo,
        }
    metadados = raiz / "normalization_metadata.json"
    with metadados.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nConcluído! Resultados: {raiz.resolve()}\nMetadados: {metadados.resolve()}")


if __name__ == "__main__":
    main()
