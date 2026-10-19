# License notice

## Code

| Part | Author | License |
|---|---|---|
| `src/sstv/` – SSTV encoder, decoder, VIS detector, line sync, classic denoiser and ML denoiser, the same files as in the 1KO125 firmware for the Xiegu X6100 | Hany El Imam | GPL 3.0 or later, with the additional term below |
| `src/sstv_channel_sim.c`, `src/sstv_denoise_tool.c`, `src/stubs/`, all the Python scripts, the launchers and the documentation | Hany El Imam | GPL 3.0 or later, with the additional term below |
| `src/third_party/stb_image.h`, `stb_image_write.h` | Sean Barrett and contributors | public domain or MIT (the choice is written at the end of each file) |
| `2_training_data/fonts/` – the fonts used to draw text pictures | their authors, listed in `2_training_data/fonts/FONTS.md` | SIL Open Font License 1.1 (`2_training_data/fonts/OFL.txt`) |

The full text of the GNU General Public License version 3 is in `LICENSE`.

## Additional term (GNU GPL v3, section 7(b))

> Any work that contains these files, or a work based on them, must preserve
> the author attribution "Hany El Imam" and the credits of the WEFAX,
> NAVTEX/SITOR-B, DSC, SSTV and FT8 CB applications in the Appropriate Legal
> Notices displayed by its user interface. In 1KO125 these notices are shown
> in the Info window of each application.

For this package, which has no graphical interface, this means: keep this
file and the author attribution in the README and in the source headers.

## Model

`models/sstv_dnn_20x6.bin`, the trained network of the 1KO125 firmware, is
distributed under the same terms as the code (GPL 3.0 or later).

## Pictures

The pictures of `1_test_set/reference_pictures/` have their own credits and
licenses, listed in `1_test_set/reference_pictures/CREDITS.md`.

## Warranty

This program comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law. See `LICENSE`, sections 15 and 16.
