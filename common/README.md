# common – shared Python code

| File | What |
|---|---|
| `sstvml.py` | folders of the package, the two C tools, picture preparation (any picture → full 320×256), the SSTV channel simulation, picture comparison (PSNR), near-duplicate fingerprints |
| `scoring.py` | scores denoisers on a test set (steps 4 and 6) |
| `html_page.py` | the HTML pages of steps 5 and 6 |
| `model_file.py` | reads and writes the model file (`SSTVDNN1`), the format read by `src/sstv/sstv_dnn.c` |

The scripts of the numbered folders import these files; they are not meant
to be run by themselves.
