# Testing — Vision stack (`vision.py`)

This document describes how to verify the Week 1 gate-proxy detector: HSV masking, contour filtering, and CLI output.

## Scope

| Layer | What is verified |
|--------|-------------------|
| **CLI** | Argument parsing, image load, demo path, exit codes, optional save path |
| **Pipeline** | `detect_rectangle` returns a dict with expected keys or `None` when nothing qualifies |
| **Visual** | `annotate_image` draws box and crosshair; saved PNG opens and looks aligned with the gate |

## Environment

1. Use the same Python interpreter for installs and runs (avoid `ModuleNotFoundError: cv2`).

   ```bash
   python -m pip install opencv-python numpy
   ```

2. On macOS with a Homebrew “externally managed” Python, create a venv first, activate it, then install the requirements above inside the venv.

3. Supported input formats are whatever `cv2.imread` accepts (commonly PNG, JPEG).

## Manual test cases

Execute all commands from the repository root unless noted otherwise.

### TC-01 — Help / usage

**Command:** `python vision.py` (no `image`, no `--demo`)

**Expected:** Help text printed; process exits with code **1**.

---

### TC-02 — Demo smoke (synthetic gate)

**Command:** `python vision.py --demo`

**Expected:**

- Exit code **0**.
- Stdout includes `[demo] Generating synthetic gate image`, `Image size: 640×480`, `Target colour: red`, and a framed summary block with `Bounding Box`, `Center`, `Area`, `Vertices`.
- File **`demo_detection_output.png`** is created (or overwritten) in the current working directory.
- Open the PNG: green rectangle and red crosshair should sit on the large red “gate” frame (not only on the inner gray cutout).

**Note:** `generate_demo_image()` adds random noise. Exact `Bounding Box` numbers may differ between runs; the center should remain close to the geometric middle of the detected frame (typically near **(320, 240)** for the default 640×480 demo).

---

### TC-03 — Demo with explicit save path

**Command:** `python vision.py --demo --save /tmp/vision_demo_out.png`

**Expected:** Same behavior as TC-02, but the annotated image is written to the path given by `--save` (adjust path for your OS).

---

### TC-04 — Real image, default colour

**Prerequisites:** A test image containing a bright **red** rectangular region similar to a gate, readable at a known path.

**Command:** `python vision.py path/to/image.png --color red --save out.png`

**Expected:**

- Exit **0** if a detection is printed; stdout shows bounding box and center.
- If the scene has no qualifying contour, stdout shows `No rectangle detected.` and exit **0** (by design).
- `out.png` exists when `--save` is passed; visually confirm the overlay matches the intended gate.

---

### TC-05 — Real image, other HSV targets

Repeat TC-04 with `--color green`, `blue`, `yellow`, and `orange` on images where the dominant gate colour matches. Expect either a reasonable box or `No rectangle detected.` depending on lighting and saturation.

---

### TC-06 — Missing / invalid file

**Command:** `python vision.py /nonexistent/file.png`

**Expected:** Message like `Error: could not read image at ...` on stderr; exit code **1**.

## API-level checks (optional)

These are useful for regression testing without a GUI. They are **not** implemented in this repo by default; add `pytest` if you want automation.

| Case | Idea |
|------|------|
| `create_hsv_mask` | Unknown colour raises `ValueError`. |
| `create_hsv_mask` | For a solid BGR patch matching a range, mask has dtype `uint8` and non-zero pixels. |
| `detect_rectangle` | `None` input raises `ValueError`. |
| `detect_rectangle` | On `generate_demo_image()` with `color="red"`, result is not `None`; keys `bounding_box`, `center`, `area`, `contour`, `approx_poly` present. |
| Stability | Because the demo image includes random noise, either fix `numpy.random.seed` in tests or assert loose tolerances on box position/size. |

Suggested layout: `tests/test_vision.py` importing functions from `vision` (ensure the package path is on `PYTHONPATH` or run `pytest` from the repo root with a simple `conftest` if you convert the tree to a package).

## Quick release checklist

- [ ] TC-01, TC-02, TC-06 pass on the target Python version.
- [ ] At least one real-world image (TC-04) passes visual review for your mission colours.
- [ ] Document any known limitations (motion blur, white balance, non-gate red objects) in deliverable notes if relevant.

## References

- Source: `vision.py` — CLI in `main()`, core logic in `detect_rectangle`, `create_hsv_mask`, `annotate_image`, `generate_demo_image`.
