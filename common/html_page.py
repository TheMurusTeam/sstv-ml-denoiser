# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
"""
The HTML pages of steps 5 and 6: one self-contained page (no internet
needed) next to a folder of pictures, with relative links, so the whole
output folder can be zipped or moved.

In the page:
  - the zoom buttons enlarge every picture (pixels stay sharp, to see the
    noise and the detail as they are);
  - pressing and holding the mouse (or a finger) on a cleaned picture shows
    the received picture in its place: what the denoiser removed is what
    changes.
"""
import html

STYLE = """
:root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#deded9; --accent:#2f6fdb;
        --good:#1f8a4c; --zoom:2; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161616; --fg:#ececea; --muted:#9b9b96; --card:#1f1f1f; --line:#333331; --accent:#7aa7ff;
          --good:#5fcf8f; color-scheme: dark; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--fg);
       font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 100%; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:17px; margin:32px 0 8px; }
p.sub { color:var(--muted); margin:0 0 16px; max-width:70ch; overflow-wrap:anywhere; }
.bar { position:sticky; top:0; z-index:2; background:var(--bg); padding:8px 0; display:flex; gap:8px;
       align-items:center; flex-wrap:wrap; border-bottom:1px solid var(--line); margin-bottom:12px; }
.bar button { font:inherit; padding:4px 12px; border-radius:6px; border:1px solid var(--line);
              background:var(--card); color:var(--fg); cursor:pointer; }
.bar button.on { border-color:var(--accent); color:var(--accent); }
.bar span { color:var(--muted); margin-left:8px; }
.scroll { overflow-x:auto; }
table.scores { border-collapse:collapse; background:var(--card); border:1px solid var(--line); }
table.scores th, table.scores td { padding:4px 10px; border-bottom:1px solid var(--line); text-align:right;
                                   font-variant-numeric: tabular-nums; white-space:nowrap; }
table.scores th:first-child, table.scores td:first-child { text-align:left; }
table.scores tr.all td { font-weight:600; }
td.best { color:var(--good); font-weight:600; }
.row { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px;
       margin:0 0 12px; }
.row .name { font-weight:600; }
.row .info { color:var(--muted); font-size:13px; }
.pics { display:flex; gap:10px; overflow-x:auto; padding-top:8px; }
figure { margin:0; flex:none; }
figcaption { font-size:13px; color:var(--muted); margin-bottom:4px; }
figure img { display:block; width:calc(320px * var(--zoom)); height:auto; image-rendering:pixelated;
             border:1px solid var(--line); background:#000; }
figure img.hold { cursor:pointer; }
@media (max-width: 700px) { figure img { width:calc(160px * var(--zoom)); } }
"""

SCRIPT = """
function zoom(z) {
  document.documentElement.style.setProperty('--zoom', z);
  document.querySelectorAll('.bar button').forEach(b => b.classList.toggle('on', b.dataset.z == z));
  try { localStorage.setItem('sstv_zoom', z); } catch (e) {}
}
document.querySelectorAll('.bar button').forEach(b => b.onclick = () => zoom(b.dataset.z));
let z0 = DEFAULT_ZOOM; try { z0 = localStorage.getItem('sstv_zoom') || DEFAULT_ZOOM; } catch (e) {}
zoom(z0);
document.querySelectorAll('img.hold').forEach(img => {
  const own = img.src, rx = img.dataset.rx;
  const show = e => { e.preventDefault(); img.src = rx; };
  const back = () => { img.src = own; };
  img.addEventListener('mousedown', show); img.addEventListener('touchstart', show, {passive:false});
  ['mouseup', 'mouseleave', 'touchend', 'touchcancel'].forEach(t => img.addEventListener(t, back));
});
"""


def esc(s):
    return html.escape(str(s))


def figure(caption, src, received=None):
    """One picture with its caption; with received, press-and-hold shows it"""
    if received:
        return ('<figure><figcaption>%s</figcaption><img class="hold" src="%s" data-rx="%s" loading="lazy" '
                'alt="%s" title="press and hold: the received picture"></figure>'
                % (esc(caption), esc(src), esc(received), esc(caption)))
    return '<figure><figcaption>%s</figcaption><img src="%s" loading="lazy" alt="%s"></figure>' % (
        esc(caption), esc(src), esc(caption))


def row(name, info, figures):
    return ('<div class="row"><div class="name">%s</div><div class="info">%s</div><div class="pics">%s</div></div>'
            % (esc(name), esc(info), "".join(figures)))


def scores_table(result, title):
    """The table of scoring.score_set() results; the best gain of each row in green"""
    import scoring
    names = result["names"]
    out = ['<h2>%s</h2><div class="scroll"><table class="scores"><tr><th>noise</th><th>pictures</th>'
           '<th>received dB</th>%s</tr>' % (esc(title), "".join("<th>%s</th>" % esc(n) for n in names))]
    for case in sorted(result["cases"], key=scoring.case_order):
        r = result["cases"][case]
        best = max(r[n] for n in names)
        cells = "".join('<td class="%s">%+.2f</td>' % ("best" if r[n] == best else "", r[n]) for n in names)
        out.append('<tr class="%s"><td>%s</td><td>%d</td><td>%.2f</td>%s</tr>'
                   % ("all" if case.startswith("ALL") else "", esc(case), result["count"][case], r["received"], cells))
    out.append("</table></div>")
    if result["clean"]:
        out.append('<p class="sub">Clean pictures left alone (PSNR dB, higher = less changed; above about 40 dB '
                   'the change cannot be seen): %s</p>'
                   % ", ".join("%s mean %.1f, worst %.1f" % (esc(n), m, w) for n, (m, w) in result["clean"].items()))
    return "".join(out)


def write_page(path, title, intro, body, columns=2):
    """columns: pictures per row; with more than 2 the page starts at zoom 1x"""
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title><style>%s</style></head>
<body><main>
<h1>%s</h1><p class="sub">%s</p>
<div class="bar">zoom <button data-z="1">1x</button><button data-z="2">2x</button><button data-z="3">3x</button>
<span>press and hold a cleaned picture to see the received one</span></div>
%s
</main><script>%s</script></body></html>
""" % (esc(title), STYLE, esc(title), intro, body, SCRIPT.replace("DEFAULT_ZOOM", "2" if columns <= 2 else "1"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
