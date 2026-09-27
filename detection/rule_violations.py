"""
DriveShieldX — Layer-2 rule-violation detectors.

Detects:
    - No helmet
    - Triple riding / three-seater
    - No seatbelt
    - License plate localisation

Important:
    twowheeler_best.pt is NOT assumed to detect triple_riding.
    Triple riding is determined by counting COCO 'person' detections.

Seatbelt detection:
    The seatbelt model is applied to a PERSON crop associated with
    the detected car, rather than directly to the entire vehicle bbox.

Temporal confirmation prevents the same tracker from generating
the same violation repeatedly.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


# ---------------------------------------------------------------------------
# MODEL DIRECTORY
# ---------------------------------------------------------------------------

MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
)


# ---------------------------------------------------------------------------
# FINES
# ---------------------------------------------------------------------------

DEFAULT_FINES = {
    "no_helmet": 500,
    "three_seater": 1000,
    "no_seatbelt": 1000,
}


# ---------------------------------------------------------------------------
# MODEL CLASS → RULE MAPPING
# ---------------------------------------------------------------------------

_POSITIVE = {
    "no_helmet": {
        "no_helmet",
        "no helmet",
        "no-helmet",
        "without_helmet",
        "without helmet",
        "unhelmeted",
    },

    "no_seatbelt": {
        "no_seatbelt",
        "no seatbelt",
        "no-seatbelt",
        "without_seatbelt",
        "without seatbelt",
        "unbelted",
    },

    "three_seater": {
        "triple_riding",
        "triple riding",
        "three_seater",
        "three seater",
    },
}


# ---------------------------------------------------------------------------
# YOLO LOADER
# ---------------------------------------------------------------------------

def _load(name: str):
    """Load a YOLO model safely."""

    path = MODEL_DIR / name

    if not path.exists():
        print(
            f"[RuleDetector] Model not found: {path}"
        )
        return None

    try:
        from ultralytics import YOLO

        model = YOLO(str(path))

        print(
            f"[RuleDetector] Loaded: {name}"
        )

        return model

    except Exception as exc:
        print(
            f"[RuleDetector] Failed loading "
            f"{name}: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# RULE DETECTOR
# ---------------------------------------------------------------------------

class RuleViolationDetector:
    """
    Helmet / triple-riding / seatbelt / plate detector.

    Triple riding:
        YOLOv8n person detection is used to count people
        associated with a motorcycle.

    Seatbelt:
        The car is first associated with a person.
        The seatbelt model is then run on that PERSON crop.

    Temporal confirmation:
        A rule must be detected in consecutive frames before
        it becomes a confirmed violation.
    """

    # Number of consecutive frames required.
    CONFIRM_FRAMES = 2

    # Person confidence for triple-riding/person association.
    PERSON_CONF = 0.30

    # Three or more people associated with a motorcycle.
    TRIPLE_RIDING_COUNT = 3

    # Minimum no-seatbelt confidence.
    NO_SEATBELT_CONF = 0.50

    # Seatbelt model input size.
    SEATBELT_IMGSZ = 640

    def __init__(self) -> None:

        # ---------------------------------------------------------------
        # SPECIALIST MODELS
        # ---------------------------------------------------------------

        self.helmet_model = _load(
            "helmet_yolov8.pt"
        )

        self.seatbelt_model = _load(
            "seatbelt_yolov8.pt"
        )

        self.twowheeler_model = _load(
            "twowheeler_best.pt"
        )

        self.plate_model = _load(
            "plate_best_v2.pt"
        )

        # ---------------------------------------------------------------
        # GENERAL COCO MODEL
        #
        # Class 0 = person
        # ---------------------------------------------------------------

        self.person_model = _load(
            "yolov8n.pt"
        )

        # ---------------------------------------------------------------
        # TEMPORAL STATE
        # ---------------------------------------------------------------

        self._streak: Dict[
            str,
            Dict[str, int]
        ] = defaultdict(
            lambda: defaultdict(int)
        )

        self._confirmed: Dict[
            str,
            set
        ] = defaultdict(set)

    # -------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------

    def loaded(self) -> Dict[str, bool]:
        """Return model-loading status."""

        return {
            "helmet": (
                self.helmet_model
                is not None
            ),
            "seatbelt": (
                self.seatbelt_model
                is not None
            ),
            "twowheeler": (
                self.twowheeler_model
                is not None
            ),
            "plate": (
                self.plate_model
                is not None
            ),
            "person": (
                self.person_model
                is not None
            ),
        }

    # -------------------------------------------------------------------
    # GENERIC SPECIALIST MODEL CHECK
    # -------------------------------------------------------------------

    def _has(
        self,
        model,
        frame,
        box,
        rule: str,
    ) -> bool:
        """
        Check whether a specialist model detects a violation class.
        """

        if model is None:
            return False

        try:

            x1, y1, x2, y2 = map(
                int,
                box
            )

            h, w = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return False

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                return False

            results = model(
                crop,
                verbose=False,
                conf=0.15,
                imgsz=384,
            )

            if not results:
                return False

            result = results[0]

            names = (
                getattr(
                    model,
                    "names",
                    {}
                )
                or {}
            )

            targets = _POSITIVE.get(
                rule,
                set()
            )

            for detection in result.boxes:

                cls_id = int(
                    detection.cls[0]
                )

                confidence = float(
                    detection.conf[0]
                )

                label = str(
                    names.get(
                        cls_id,
                        ""
                    )
                ).strip().lower()

                if (
                    label in targets
                    and confidence >= 0.15
                ):
                    return True

            return False

        except Exception as exc:

            print(
                f"[RuleDetector] "
                f"_has({rule}) failed: {exc}"
            )

            return False

    # -------------------------------------------------------------------
    # PERSON COUNTING
    # -------------------------------------------------------------------

    def _count_people_inside(
        self,
        frame,
        vehicle_box,
    ) -> int:
        """
        Count people associated with a motorcycle.

        The motorcycle ROI is expanded because the rider's body/head
        can extend considerably above the motorcycle bbox.
        """

        if self.person_model is None:
            return 0

        try:

            x1, y1, x2, y2 = map(
                int,
                vehicle_box
            )

            h, w = frame.shape[:2]

            pad_x = 40
            pad_top = 120
            pad_bottom = 40

            rx1 = max(
                0,
                x1 - pad_x
            )

            ry1 = max(
                0,
                y1 - pad_top
            )

            rx2 = min(
                w,
                x2 + pad_x
            )

            ry2 = min(
                h,
                y2 + pad_bottom
            )

            if rx2 <= rx1 or ry2 <= ry1:
                return 0

            crop = frame[
                ry1:ry2,
                rx1:rx2
            ]

            if crop.size == 0:
                return 0

            results = self.person_model(
                crop,
                verbose=False,
                conf=self.PERSON_CONF,
                imgsz=640,
            )

            if not results:
                return 0

            result = results[0]

            count = 0

            for detection in result.boxes:

                cls_id = int(
                    detection.cls[0]
                )

                if cls_id != 0:
                    continue

                confidence = float(
                    detection.conf[0]
                )

                if confidence < self.PERSON_CONF:
                    continue

                px1, py1, px2, py2 = map(
                    int,
                    detection.xyxy[0].tolist()
                )

                px1 += rx1
                py1 += ry1
                px2 += rx1
                py2 += ry1

                center_x = (
                    px1 + px2
                ) / 2.0

                bottom_y = float(py2)

                if (
                    rx1 <= center_x <= rx2
                    and ry1 <= bottom_y <= ry2
                ):
                    count += 1

            return count

        except Exception as exc:

            print(
                "[RuleDetector] "
                f"Person counting failed: {exc}"
            )

            return 0

    # -------------------------------------------------------------------
    # TRIPLE RIDING
    # -------------------------------------------------------------------

    def _is_triple_riding(
        self,
        frame,
        vehicle_box,
        person_boxes=None,
    ) -> bool:
        """
        Determine whether a motorcycle has >= 3 riders.
        """

        # ---------------------------------------------------------------
        # METHOD 1 — COCO PERSON DETECTOR
        # ---------------------------------------------------------------

        person_count = (
            self._count_people_inside(
                frame,
                vehicle_box,
            )
        )

        if (
            person_count
            >= self.TRIPLE_RIDING_COUNT
        ):
            return True

        # ---------------------------------------------------------------
        # METHOD 2 — PERSON BOXES FROM PIPELINE
        # ---------------------------------------------------------------

        if person_boxes:

            try:

                count = self._riders_inside(
                    vehicle_box,
                    person_boxes,
                )

                if (
                    count
                    >= self.TRIPLE_RIDING_COUNT
                ):
                    return True

            except Exception:
                pass

        # ---------------------------------------------------------------
        # METHOD 3 — SPECIALIST MODEL CLASS
        # ---------------------------------------------------------------

        if (
            self.twowheeler_model
            is not None
        ):

            try:

                names = (
                    getattr(
                        self.twowheeler_model,
                        "names",
                        {}
                    )
                    or {}
                )

                has_triple_class = any(
                    str(name)
                    .strip()
                    .lower()
                    in _POSITIVE[
                        "three_seater"
                    ]
                    for name in names.values()
                )

                if has_triple_class:

                    if self._has(
                        self.twowheeler_model,
                        frame,
                        vehicle_box,
                        "three_seater",
                    ):
                        return True

            except Exception:
                pass

        return False

    # -------------------------------------------------------------------
    # COMPATIBILITY PERSON-BOX COUNTER
    # -------------------------------------------------------------------

    def _riders_inside(
        self,
        box,
        person_boxes,
    ) -> int:
        """Count person boxes inside an expanded bike bbox."""

        if not person_boxes:
            return 0

        try:

            x1, y1, x2, y2 = map(
                int,
                box
            )

            exp_x1 = x1 - 40
            exp_y1 = y1 - 120
            exp_x2 = x2 + 40
            exp_y2 = y2 + 40

            count = 0

            for person in person_boxes:

                if len(person) < 4:
                    continue

                px1, py1, px2, py2 = map(
                    float,
                    person[:4]
                )

                center_x = (
                    px1 + px2
                ) / 2.0

                bottom_y = py2

                if (
                    exp_x1 <= center_x <= exp_x2
                    and exp_y1 <= bottom_y <= exp_y2
                ):
                    count += 1

            return count

        except Exception:
            return 0

    # -------------------------------------------------------------------
    # CAR PERSON ASSOCIATION
    # -------------------------------------------------------------------

    def _find_person_for_car(
        self,
        frame,
        vehicle_box,
        person_boxes=None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the most likely driver/person associated with a car.

        Priority:
            1. Person boxes already supplied by the pipeline.
            2. Fresh YOLOv8n person detection inside the car.

        Returns:
            (x1, y1, x2, y2)
            or None.
        """

        try:

            h, w = frame.shape[:2]

            vx1, vy1, vx2, vy2 = map(
                int,
                vehicle_box
            )

            vx1 = max(
                0,
                vx1
            )

            vy1 = max(
                0,
                vy1
            )

            vx2 = min(
                w,
                vx2
            )

            vy2 = min(
                h,
                vy2
            )

            if (
                vx2 <= vx1
                or vy2 <= vy1
            ):
                return None

            # -----------------------------------------------------------
            # METHOD 1 — USE PERSON BOXES FROM PIPELINE
            # -----------------------------------------------------------

            best_person = None
            best_score = -1.0

            if person_boxes:

                for person in person_boxes:

                    if len(person) < 4:
                        continue

                    px1, py1, px2, py2 = map(
                        int,
                        person[:4]
                    )

                    px1 = max(
                        0,
                        px1
                    )

                    py1 = max(
                        0,
                        py1
                    )

                    px2 = min(
                        w,
                        px2
                    )

                    py2 = min(
                        h,
                        py2
                    )

                    if (
                        px2 <= px1
                        or py2 <= py1
                    ):
                        continue

                    center_x = (
                        px1 + px2
                    ) / 2.0

                    center_y = (
                        py1 + py2
                    ) / 2.0

                    # ---------------------------------------------------
                    # Person center should lie inside vehicle.
                    # ---------------------------------------------------

                    if not (
                        vx1 <= center_x <= vx2
                        and vy1 <= center_y <= vy2
                    ):
                        continue

                    area = (
                        px2 - px1
                    ) * (
                        py2 - py1
                    )

                    # Prefer larger associated person.
                    if area > best_score:
                        best_score = area
                        best_person = (
                            px1,
                            py1,
                            px2,
                            py2,
                        )

            if best_person is not None:
                return best_person

            # -----------------------------------------------------------
            # METHOD 2 — DETECT PERSON DIRECTLY
            # -----------------------------------------------------------

            if self.person_model is None:
                return None

            crop = frame[
                vy1:vy2,
                vx1:vx2
            ]

            if crop.size == 0:
                return None

            results = self.person_model(
                crop,
                verbose=False,
                conf=0.25,
                imgsz=640,
            )

            if not results:
                return None

            result = results[0]

            best_person = None
            best_area = 0

            for detection in result.boxes:

                cls_id = int(
                    detection.cls[0]
                )

                if cls_id != 0:
                    continue

                confidence = float(
                    detection.conf[0]
                )

                if confidence < 0.25:
                    continue

                px1, py1, px2, py2 = map(
                    int,
                    detection.xyxy[0].tolist()
                )

                px1 += vx1
                py1 += vy1
                px2 += vx1
                py2 += vy1

                px1 = max(
                    0,
                    px1
                )

                py1 = max(
                    0,
                    py1
                )

                px2 = min(
                    w,
                    px2
                )

                py2 = min(
                    h,
                    py2
                )

                area = (
                    max(
                        0,
                        px2 - px1
                    )
                    *
                    max(
                        0,
                        py2 - py1
                    )
                )

                if area > best_area:

                    best_area = area

                    best_person = (
                        px1,
                        py1,
                        px2,
                        py2,
                    )

            return best_person

        except Exception as exc:

            print(
                "[RuleDetector] "
                f"Person/car association failed: {exc}"
            )

            return None

    # -------------------------------------------------------------------
    # SEATBELT CHECK
    # -------------------------------------------------------------------

    def _check_car_seatbelt(
        self,
        frame,
        vehicle_box,
        person_boxes=None,
    ) -> bool:
        """
        Detect no-seatbelt on the person/driver associated with a car.

        IMPORTANT:
            seatbelt_yolov8.pt is run on the PERSON crop, not the
            entire vehicle crop.
        """

        if self.seatbelt_model is None:
            return False

        try:

            person_box = (
                self._find_person_for_car(
                    frame=frame,
                    vehicle_box=vehicle_box,
                    person_boxes=person_boxes,
                )
            )

            if person_box is None:

                print(
                    "[RuleDetector] "
                    "No person associated with car."
                )

                return False

            px1, py1, px2, py2 = (
                map(
                    int,
                    person_box
                )
            )

            h, w = frame.shape[:2]

            px1 = max(
                0,
                px1
            )

            py1 = max(
                0,
                py1
            )

            px2 = min(
                w,
                px2
            )

            py2 = min(
                h,
                py2
            )

            if (
                px2 <= px1
                or py2 <= py1
            ):
                return False

            # -----------------------------------------------------------
            # PERSON CROP
            # -----------------------------------------------------------

            crop = frame[
                py1:py2,
                px1:px2
            ]

            if crop.size == 0:
                return False

            # -----------------------------------------------------------
            # SEATBELT MODEL
            # -----------------------------------------------------------

            results = self.seatbelt_model(
                crop,
                verbose=False,
                conf=0.15,
                imgsz=self.SEATBELT_IMGSZ,
            )

            if not results:
                return False

            result = results[0]

            names = (
                getattr(
                    self.seatbelt_model,
                    "names",
                    {}
                )
                or {}
            )

            best_no_seatbelt_conf = 0.0
            best_seatbelt_conf = 0.0

            # -----------------------------------------------------------
            # CLASS SCORES
            # -----------------------------------------------------------

            for detection in result.boxes:

                cls_id = int(
                    detection.cls[0]
                )

                confidence = float(
                    detection.conf[0]
                )

                label = str(
                    names.get(
                        cls_id,
                        ""
                    )
                ).strip().lower()

                if label in {
                    "no_seatbelt",
                    "no seatbelt",
                    "no-seatbelt",
                    "without_seatbelt",
                    "without seatbelt",
                    "unbelted",
                }:

                    best_no_seatbelt_conf = max(
                        best_no_seatbelt_conf,
                        confidence,
                    )

                elif label in {
                    "seatbelt",
                    "seat belt",
                    "belted",
                }:

                    best_seatbelt_conf = max(
                        best_seatbelt_conf,
                        confidence,
                    )

            # -----------------------------------------------------------
            # DEBUG OUTPUT
            # -----------------------------------------------------------

            print(
                "[RuleDetector] Seatbelt:"
                f" no_seatbelt="
                f"{best_no_seatbelt_conf:.3f}"
                f" seatbelt="
                f"{best_seatbelt_conf:.3f}"
            )

            # -----------------------------------------------------------
            # DECISION
            #
            # A no-seatbelt violation requires:
            #
            #   no_seatbelt >= 0.50
            #
            # AND
            #
            #   no_seatbelt >= seatbelt
            #
            # This prevents a weak no-seatbelt detection from
            # overriding a stronger seatbelt detection.
            # -----------------------------------------------------------

            if (
                best_no_seatbelt_conf
                >= self.NO_SEATBELT_CONF
                and
                best_no_seatbelt_conf
                >= best_seatbelt_conf
            ):

                print(
                    "[RuleDetector] "
                    "NO SEATBELT CONFIRMED"
                )

                return True

            return False

        except Exception as exc:

            print(
                "[RuleDetector] "
                f"Car seatbelt check failed: {exc}"
            )

            return False

    # -------------------------------------------------------------------
    # LICENSE PLATE LOCALISATION
    # -------------------------------------------------------------------

    def locate_plate(
        self,
        frame,
        vehicle_box,
    ) -> Optional[
        Tuple[list, float]
    ]:
        """
        Return plate bbox and confidence.

        OCR remains handled by NumberPlateRecognizer.
        """

        if self.plate_model is None:
            return None

        try:

            x1, y1, x2, y2 = map(
                int,
                vehicle_box
            )

            h, w = frame.shape[:2]

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                w,
                x2
            )

            y2 = min(
                h,
                y2
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                return None

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                return None

            results = self.plate_model(
                crop,
                verbose=False,
                conf=0.30,
                imgsz=320,
            )

            if not results:
                return None

            result = results[0]

            best = None
            best_conf = 0.0

            for detection in result.boxes:

                confidence = float(
                    detection.conf[0]
                )

                if confidence <= best_conf:
                    continue

                px1, py1, px2, py2 = map(
                    int,
                    detection.xyxy[0].tolist()
                )

                best = [
                    x1 + px1,
                    y1 + py1,
                    x1 + px2,
                    y1 + py2,
                ]

                best_conf = confidence

            if best is None:
                return None

            return (
                best,
                best_conf,
            )

        except Exception as exc:

            print(
                "[RuleDetector] "
                f"Plate localisation failed: {exc}"
            )

            return None

    # -------------------------------------------------------------------
    # MAIN OBSERVE FUNCTION
    # -------------------------------------------------------------------

    def observe(
        self,
        tracker_id: str,
        vehicle_type: str,
        frame,
        vehicle_box,
        person_boxes=None,
    ) -> List[str]:
        """
        Check rules for one tracked vehicle.

        Returns only newly confirmed violations.

        Example:

            frame 1 → no seatbelt
            frame 2 → no seatbelt
            frame 3 → already confirmed → nothing

        Result:

            ["no_seatbelt"]
        """

        confirmed_now: List[str] = []

        checks: Dict[str, bool] = {}

        vt = (
            vehicle_type or ""
        ).strip().lower()

        # ---------------------------------------------------------------
        # VEHICLE TYPE NORMALISATION
        # ---------------------------------------------------------------

        is_bike = vt in {
            "bike",
            "motorcycle",
            "motorbike",
            "two-wheeler",
            "two_wheeler",
        }

        is_car = vt in {
            "car",
            "sedan",
            "hatchback",
            "suv",
        }

        # ---------------------------------------------------------------
        # MOTORCYCLE RULES
        # ---------------------------------------------------------------

        if is_bike:

            checks[
                "three_seater"
            ] = self._is_triple_riding(
                frame,
                vehicle_box,
                person_boxes=person_boxes,
            )

            checks[
                "no_helmet"
            ] = (
                self._has(
                    self.twowheeler_model,
                    frame,
                    vehicle_box,
                    "no_helmet",
                )
                or
                self._has(
                    self.helmet_model,
                    frame,
                    vehicle_box,
                    "no_helmet",
                )
            )

        # ---------------------------------------------------------------
        # CAR RULES
        # ---------------------------------------------------------------

        if is_car:

            checks[
                "no_seatbelt"
            ] = self._check_car_seatbelt(
                frame=frame,
                vehicle_box=vehicle_box,
                person_boxes=person_boxes,
            )

        # ---------------------------------------------------------------
        # TEMPORAL CONFIRMATION
        # ---------------------------------------------------------------

        state = self._streak[
            tracker_id
        ]

        already = self._confirmed[
            tracker_id
        ]

        for rule, hit in checks.items():

            # Already reported for this tracker.
            if rule in already:
                continue

            if hit:

                state[rule] += 1

            else:

                state[rule] = 0

            # -----------------------------------------------------------
            # Confirm after consecutive frames.
            # -----------------------------------------------------------

            if (
                state[rule]
                >= self.CONFIRM_FRAMES
            ):

                already.add(rule)

                confirmed_now.append(
                    rule
                )

        return confirmed_now

    # -------------------------------------------------------------------
    # RESET TRACKER
    # -------------------------------------------------------------------

    def reset_tracker(
        self,
        tracker_id: str,
    ) -> None:
        """Clear temporal state for a tracker."""

        self._streak.pop(
            tracker_id,
            None,
        )

        self._confirmed.pop(
            tracker_id,
            None,
        )

    # -------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Dict[str, list]]:
        """Diagnostic information for dashboard."""

        return {
            tracker_id: {
                "confirmed": sorted(
                    rules
                )
            }
            for tracker_id, rules
            in self._confirmed.items()
        }


# ===========================================================================
# UPI QR HELPERS
# ===========================================================================

def upi_intent(
    vpa: str,
    payee: str,
    amount: int,
    tid: str,
    note: str = "",
) -> str:

    from urllib.parse import urlencode

    txn = (
        f"DSX-{tid}"
    )[:35]

    params = {
        "pa": vpa.strip(),
        "pn": payee.strip(),
        "am": str(
            max(
                0,
                int(amount)
            )
        ),
        "cu": "INR",
        "tn": (
            note
            or f"E-Challan #{tid}"
        )[:60],
        "tr": txn,
    }

    return (
        "upi://pay?"
        + urlencode(params)
    )


def upi_qr_png(
    data: str,
) -> bytes:

    import io
    import qrcode

    img = qrcode.make(
        data
    )

    buf = io.BytesIO()

    img.save(
        buf,
        format="PNG",
    )

    return buf.getvalue()