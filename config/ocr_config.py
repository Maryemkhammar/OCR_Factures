from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TesseractConfig:
    cmd:  str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    lang: str = "fra"
    psm:  int = 3
    oem:  int = 3

    @property
    def config_string(self) -> str:
        return f"--oem {self.oem} --psm {self.psm}"


@dataclass
class EasyOCRConfig:
    enabled:   bool = True
    languages: list = field(default_factory=lambda: ["fr", "en"])
    gpu:       bool = False
    verbose:   bool = False


@dataclass
class PreprocessConfig:
    dpi:                int   = 300
    thr_blue:           int   = 10
    thr_red:            int   = 37
    sat_min:            int   = 30
    val_min:            int   = 60
    dilate_iter:        int   = 2
    inpaint_radius:     int   = 2
    median_blur:        int   = 3
    deskew_min_angle:   float = 0.5
    min_area_ratio:     float = 0.05
    min_width_ratio:    float = 0.35
    min_height_ratio:   float = 0.20


@dataclass
class PathConfig:
    data_raw:       Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    data_output:    Path = Path("data/output")
    logs:           Path = Path("logs")

    def create_all(self):
        for p in [self.data_raw, self.data_processed,
                  self.data_output, self.logs]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class OCRConfig:
    tesseract:  TesseractConfig  = field(default_factory=TesseractConfig)
    easyocr:    EasyOCRConfig    = field(default_factory=EasyOCRConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    paths:      PathConfig       = field(default_factory=PathConfig)


# ✅ LIGNE OBLIGATOIRE — c'est elle qu'on importe partout
config = OCRConfig()