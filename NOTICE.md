# License notice

## Code

| Part | Author | License |
|---|---|---|
| `src/sstv/` – SSTV encoder, decoder, VIS detector, line sync, classic denoiser and ML denoiser, the same files as in the 1KO125 firmware for the Xiegu X6100 | Hany El Imam | LGPL 2.1 or later |
| `src/sstv_channel_sim.c`, `src/sstv_denoise_tool.c`, `src/stubs/`, all the Python scripts, the launchers and the documentation | Hany El Imam | LGPL 2.1 or later |
| `src/third_party/stb_image.h`, `stb_image_write.h` | Sean Barrett and contributors | public domain or MIT (the choice is written at the end of each file) |
| `2_training_data/fonts/` – the fonts used to draw text pictures | their authors, listed in `2_training_data/fonts/FONTS.md` | SIL Open Font License 1.1 (`2_training_data/fonts/OFL.txt`) |

The full text of the GNU Lesser General Public License version 2.1 is in
`LICENSE`.

## Model

`models/sstv_dnn_20x6.bin`, the trained network of the 1KO125 firmware, is
distributed under the same terms as the code (LGPL 2.1 or later).

## Pictures

The pictures of `1_test_set/reference_pictures/` have their own credits and
licenses, listed in `1_test_set/reference_pictures/CREDITS.md`.

## Warranty

This program comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law. See `LICENSE`, sections 15 and 16.

## Earlier versions

Version 1.0 of this package was released under the GNU GPL version 3 or
later, with an additional attribution term; copies of that version keep
those terms. Later versions are under the LGPL 2.1 or later.
