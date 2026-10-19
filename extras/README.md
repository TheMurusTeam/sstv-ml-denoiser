# extras – tools for special kinds of pictures

Not steps of the package: tools to prepare particular pictures before
using them in step 1 (test sets) or step 2 (training data).

## `crop_log11dx_qsl.py` – eQSL cards from LOG11DX

The eQSL cards of LOG11DX (log11dx.com) are the sender's picture on the
left and a light panel with the QSO data on the right. For the denoiser only
the picture is useful, so this script keeps the left part, at its original
size (steps 1 and 2 bring it to 320×256 afterwards).

    python3 extras/crop_log11dx_qsl.py FOLDER_OF_CARDS OUTPUT_FOLDER

- reads PNG, JPG and WEBP, subfolders included;
- writes one PNG per card, with the same name, plus `_check.jpg`: the first
  cards with the kept part outlined in magenta, to check the cuts;
- the panel is searched from the right, so light areas of the picture (the
  white of a flag, a sky) are not mistaken for it;
- pictures without a light panel on the right are reported and skipped.

Then use `OUTPUT_FOLDER` as `--pictures` in step 2, or copy the pictures into
`1_test_set/my_pictures/` for your test set. The cards are other people's
pictures: use them for your own training and tests, do not publish them.
