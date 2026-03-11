"""
configuration OCR 
"""
import os 
from dataclasses import dataclass,field  
from pathlib import Path 


#____TESSERACT_____
@dataclass 
class TesseractConfig:
    cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    #langue 
    lang: str="fra"
    psm: int = 6
    oem : int =3 
    @property
    def config_string(self) -> str:
        """ """
        return f"--oem {self.oem} --psm {self.psm}"
#______EASYOCR_____
@dataclass 
class EasyOCRConfig:
    languages:list=field(default_factory=lambda :{"fr","en"})
    
#____PREPROCESS Image ____
@dataclass
class PreprocessConfig:
    dpi: int =300 
    #remove stylo
    thr_blue:int   = 10
    thr_red:int    = 37
    sat_min:int    = 30
    val_min:int    = 60
    #nettoyage mask 
    diliate_iter: int = 2
    inpaint_radius:int = 2 
    
    #debruitage
    median_blur: int   = 3 
    
    #deskew_min_angle
    deskew_min_angle: float = 0.5
    
    #crop zone contenu ()
    min_area_ration : float =0.05
    min_width_ratio : float =0.35
    min_height_ratio : float = 0.20 
     
     
    


