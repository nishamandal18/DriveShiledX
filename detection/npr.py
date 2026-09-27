from __future__ import annotations

import re
from typing import List, Optional, Tuple

import cv2

# Indian plate (e.g. MH01AB1234). Kept for FORMAT BONUS scoring, not as a hard filter.
INDIAN_PLATE_REGEX = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}")
# General plate: a run of 5–10 alphanumerics with at least one letter and one digit.
GENERAL_PLATE_REGEX = re.compile(r"[A-Z0-9]{5,10}")


class NumberPlateRecognizer:
    """OCR-based number-plate recognizer. Uses EasyOCR if available; otherwise disabled.
    Locates the plate region inside the vehicle box (Haar cascade + heuristic fallback),
    deblurs/sharpens it, then OCRs several enhanced variants and picks the best read."""

    def __init__(self, use_gpu: bool = False) -> None:
        self.reader = None
        self.enabled = False
        self._cascade = None
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=use_gpu)
            self.enabled = True
        except Exception:
            self.reader = None
            self.enabled = False
        # Load the bundled plate Haar cascade (helps locate the actual plate rectangle).
        try:
            path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                self._cascade = c
        except Exception:
            self._cascade = None

    def _vehicle_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        v = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        return v if v.size else None

    def _plate_regions(self, vehicle) -> List:
        """Return candidate plate crops: cascade detections first, then a heuristic
        lower-centre crop as a fallback."""
        regions = []
        if vehicle is None or vehicle.size == 0:
            return regions
        gray = cv2.cvtColor(vehicle, cv2.COLOR_BGR2GRAY)
        if self._cascade is not None:
            try:
                found = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 14))
                for (x, y, w, h) in found:
                    pad = int(h * 0.15)
                    crop = vehicle[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
                    if crop.size:
                        regions.append(crop)
            except Exception:
                pass
        # Heuristic fallback: plates usually sit in the lower-centre of the vehicle box.
        h, w = vehicle.shape[:2]
        crop = vehicle[int(h * 0.55): min(h, int(h * 0.95)), int(w * 0.10): min(w, int(w * 0.90))]
        if crop.size:
            regions.append(crop)
        return regions

    def _enhance(self, crop):
        """Deblur + sharpen a (possibly motion-blurred / low-res) plate crop so OCR
        can read it. Returns a list of processed single-channel variants to try."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        h, w = gray.shape[:2]
        if max(h, w) < 240:
            scale = 240.0 / max(1, max(h, w))
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        den = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        eq = clahe.apply(den)
        blur = cv2.GaussianBlur(eq, (0, 0), 3)
        sharp = cv2.addWeighted(eq, 1.6, blur, -0.6, 0)
        th = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 9)
        return [sharp, eq, th]

    @staticmethod
    def _looks_like_plate(s: str) -> bool:
        return bool(s) and len(s) >= 5 and any(c.isalpha() for c in s) and any(c.isdigit() for c in s)

    def read_plate(self, frame, bbox) -> Tuple[Optional[str], float]:
        if not self.enabled:
            return None, 0.0
        vehicle = self._vehicle_crop(frame, bbox)
        if vehicle is None:
            return None, 0.0
        best_text = None
        best_conf = 0.0
        for region in self._plate_regions(vehicle):
            for variant in self._enhance(region):
                for _, text, conf in self.reader.readtext(variant, detail=1):
                    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
                    if len(cleaned) < 5:
                        continue
                    # Prefer an Indian-format match, then a general plate pattern.
                    m = INDIAN_PLATE_REGEX.search(cleaned)
                    if m:
                        candidate, fmt_bonus = m.group(0), 0.25
                    else:
                        gm = GENERAL_PLATE_REGEX.search(cleaned)
                        candidate = gm.group(0) if gm else cleaned
                        fmt_bonus = 0.0
                    if not self._looks_like_plate(candidate):
                        continue
                    score = float(conf) + fmt_bonus
                    if score > best_conf:
                        best_text, best_conf = candidate, score
        return best_text, min(best_conf, 1.0)
