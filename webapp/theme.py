"""
Visual theme: glassmorphism panels floating over an animated night sky, with a
lightweight particle field that scatters away from the cursor.

Purely cosmetic. Nothing here touches data, features, or predictions - app.py
still computes every number exactly as before, this module only restyles how
it's presented. Every piece degrades independently and silently: if the
background component fails to render (old browser, blocked iframe, whatever),
the CSS still applies and the app is fully usable with a plain gradient behind
it; if even the CSS injection fails, inject_theme() swallows the error rather
than take the dashboard down over a styling problem.

WHY A COMPONENT, NOT JUST st.markdown
--------------------------------------
st.markdown(unsafe_allow_html=True) sets innerHTML under the hood, and browsers
never execute <script> tags inserted that way - true regardless of framework.
CSS in a <style> tag DOES apply through that path, which is why the panel/tile
styling below is plain st.markdown, but the animated canvas needs a real parsed
HTML document to run its JS at all, so it goes through
streamlit.components.v1.html (an actual iframe), and the CSS block below
targets that iframe from the parent page to stretch it into a fixed full-screen
backdrop rather than the small inline box components.html defaults to.
"""
import streamlit as st
import streamlit.components.v1 as components


CSS = """
<style>
:root {
  /* Dark, mostly-opaque glass - readability over transparency. An earlier,
     lighter/more-see-through fill let bright bits of the animated sky bleed
     through inconsistently, dimming contrast under text wherever a star or
     nebula happened to sit. This fill is dark enough on its own that legibility
     no longer depends on what is animating behind it. */
  --glass: rgba(16,10,28,0.62);
  --glass-strong: rgba(16,10,28,0.80);
  --glass-border: rgba(255,255,255,0.14);
  --accent: #7dd3fc;
  --accent2: #e9a6ff;
  --text: #f3f2fa;
  --text-dim: #b7b3cc;
}

/* ---- sky backdrop on the real app surface ----
   Black/purple night palette - no blue-navy base. The aurora blobs and the
   base gradient are BOTH part of this element's own background (not a
   separate positioned layer) so there is no stacking-context ambiguity to get
   wrong: an element's own background is always painted, full stop, before any
   question of z-index arises. The radial layers animate purely via
   background-position, enough motion to read as a slowly drifting sky without
   a second element. */
[data-testid="stAppViewContainer"] {
  background-image:
    radial-gradient(circle at 30% 25%, rgba(147,51,234,.28), transparent 42%),
    radial-gradient(circle at 72% 38%, rgba(88,28,135,.30), transparent 46%),
    radial-gradient(circle at 50% 82%, rgba(219,39,119,.14), transparent 46%),
    radial-gradient(ellipse 90% 55% at 12% -8%, rgba(76,29,149,.40) 0%, transparent 60%),
    radial-gradient(ellipse 90% 55% at 88% 0%,   rgba(30,10,50,.45) 0%, transparent 60%),
    linear-gradient(180deg, #050308 0%, #0d0716 40%, #170a28 75%, #1c0a30 100%) !important;
  background-size: 160% 160%, 170% 170%, 150% 150%, 100% 100%, 100% 100%, 100% 100%;
  background-repeat: no-repeat !important;
  background-attachment: fixed !important;
  animation: sky-drift 30s ease-in-out infinite alternate;
}
@keyframes sky-drift {
  0%   { background-position: 0% 0%,   100% 20%, 50% 100%, 0 0, 0 0, 0 0; }
  50%  { background-position: 15% 10%, 80% 35%,  60% 85%,  0 0, 0 0, 0 0; }
  100% { background-position: 5% 20%,  90% 10%,  40% 95%,  0 0, 0 0, 0 0; }
}
[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }

html, body, p, li, label, [data-testid="stMarkdownContainer"] {
  color: var(--text);
  font-family: -apple-system, "Segoe UI", Roboto, Inter, sans-serif;
}
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-dim) !important; }

/* System font stacks only - no external @import. A prior version pulled
   Space Grotesk/Inter from Google Fonts via @import, which is render-blocking:
   a viewer with any latency or failure reaching fonts.googleapis.com (flaky
   network, corporate/regional blocking, an ad/privacy blocker) could stall the
   whole page's first paint behind it. The distinct look comes from weight and
   letter-spacing here, not a particular typeface. */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif !important;
  letter-spacing: -0.01em;
}
h1 {
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent !important;
  font-weight: 700 !important;
}

/* ---- background particle iframe -> fixed, full-viewport, click-through ----
   Verified against the real DOM: components.v1.html renders as
   <iframe data-testid="stIFrame"> inside a <div data-testid="stElementContainer">.
   Both selectors are covered (container AND iframe itself, both !important)
   so this survives whichever the current Streamlit version keys layout on.

   z-index is 0, deliberately NOT negative. stAppViewContainer never becomes a
   stacking context of its own (no position/transform/opacity on it), so a
   negative z-index here would be compared at the ROOT stacking context - where
   it loses to stAppViewContainer's own (fully opaque) background under the
   ordinary CSS painting-order rules, hiding the whole layer behind it
   completely. Non-negative + this being the first content injected in main()
   (so it is earliest in DOM order) is what keeps it behind the later glass
   panels without needing a negative value at all. */
div[data-testid="stElementContainer"]:has(> iframe[data-testid="stIFrame"]) {
  position: fixed !important; inset: 0 !important; z-index: 0 !important;
  pointer-events: none !important; width: 100vw !important; height: 100vh !important;
  max-width: none !important; overflow: hidden !important;
}
iframe[data-testid="stIFrame"] {
  position: fixed !important; inset: 0 !important; z-index: 0 !important;
  width: 100vw !important; height: 100vh !important;
  border: 0 !important; pointer-events: none !important;
}

/* ---- glass panels: st.container(border=True) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--glass) !important;
  backdrop-filter: blur(20px) saturate(150%);
  -webkit-backdrop-filter: blur(20px) saturate(150%);
  border: 1px solid var(--glass-border) !important;
  border-radius: 22px !important;
  box-shadow: 0 8px 32px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
  padding: 4px !important;
  margin-bottom: 1.4rem;
  transition: box-shadow .3s ease, transform .3s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  box-shadow: 0 14px 44px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.10);
}

/* ---- glass metric tiles (the NOW / +24h / +48h / +72h badges) ---- */
.glass-tile {
  position: relative; overflow: hidden;
  border-radius: 18px; padding: 18px 14px; text-align: center;
  border: 1px solid rgba(255,255,255,.22);
  box-shadow: 0 6px 22px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.18);
  backdrop-filter: blur(6px);
  transition: transform .22s ease, box-shadow .22s ease;
}
.glass-tile:hover { transform: translateY(-4px) scale(1.015); box-shadow: 0 12px 30px rgba(0,0,0,.45); }
.glass-tile .g-label { font-size: .8rem; font-weight: 700; opacity: .9; letter-spacing: .04em; text-transform: uppercase; text-shadow: 0 1px 3px rgba(0,0,0,.35); }
.glass-tile .g-value { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1.05; margin: 2px 0; text-shadow: 0 2px 6px rgba(0,0,0,.35); }
.glass-tile .g-cat { font-size: .84rem; font-weight: 800; text-shadow: 0 1px 3px rgba(0,0,0,.35); }
.glass-tile .g-sub { font-size: .74rem; opacity: .9; margin-top: 2px; text-shadow: 0 1px 2px rgba(0,0,0,.3); }

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 6px 22px rgba(220,40,60,.25), inset 0 1px 0 rgba(255,255,255,.18); }
  50%      { box-shadow: 0 6px 34px rgba(220,40,60,.55), inset 0 1px 0 rgba(255,255,255,.18); }
}
.glass-tile.g-hazard { animation: pulse-glow 2.4s ease-in-out infinite; }

/* ---- legend pills ---- */
.pill-row { margin: 4px 0 14px; display: flex; flex-wrap: wrap; gap: 6px; }
.pill {
  padding: 4px 12px; border-radius: 999px; font-size: .72rem; font-weight: 700;
  color: #10131f; border: 1px solid rgba(255,255,255,.35);
  box-shadow: 0 2px 8px rgba(0,0,0,.25);
}

/* ---- widgets ---- */
[data-testid="stExpander"] {
  background: var(--glass) !important; backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border) !important; border-radius: 18px !important;
  overflow: hidden;
}
[data-testid="stAlert"] {
  border-radius: 16px !important; backdrop-filter: blur(14px);
  border: 1px solid var(--glass-border) !important;
}
[data-testid="stMetric"] {
  background: var(--glass); border: 1px solid var(--glass-border);
  border-radius: 14px; padding: 10px 6px;
}
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; background: var(--glass-strong); }
div[role="radiogroup"] label {
  background: var(--glass); border: 1px solid var(--glass-border);
  border-radius: 999px !important; padding: 4px 14px !important; margin-right: 6px;
}
[data-baseweb="select"] > div {
  background: var(--glass) !important; border-color: var(--glass-border) !important;
  border-radius: 12px !important;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.18); border-radius: 8px; }
</style>
"""

# Small, dependency-free particle field: cool/violet motes drifting slowly,
# scattering from the cursor within a short radius, easing back afterwards.
# Themed loosely as haze/smog motes being "cleared" by the pointer - a nod to
# the app's subject without touching any real data.
_BACKGROUND_HTML = """
<!DOCTYPE html><html><head><style>
  html,body{margin:0;padding:0;overflow:hidden;background:transparent;}
  canvas{display:block;}
</style></head><body>
<canvas id="sky"></canvas>
<script>
(function () {
  var canvas = document.getElementById('sky');
  var ctx = canvas.getContext('2d');
  var W = window.innerWidth, H = window.innerHeight;
  canvas.width = W; canvas.height = H;
  window.addEventListener('resize', function () {
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W; canvas.height = H;
  });

  var N = Math.max(70, Math.min(160, Math.floor((W * H) / 7000)));
  var pts = [];
  for (var i = 0; i < N; i++) {
    var big = Math.random() < 0.22;
    pts.push({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.12, vy: (Math.random() - 0.5) * 0.12,
      r: big ? (Math.random() * 2.2 + 2.6) : (Math.random() * 1.4 + 1.1),
      tw: Math.random() * Math.PI * 2,
      hue: Math.random() < 0.4 ? '186,230,253' : (Math.random() < 0.7 ? '221,190,254' : '251,182,206')
    });
  }

  // Clouds: soft clusters of overlapping radial-gradient puffs, dark violet,
  // drifting slowly and independently of the cursor (unlike the star field
  // above, these are a passive far-background layer - real night clouds don't
  // dodge anything). Each puff's gradient is built ONCE, in cloud-local
  // coordinates; every frame just translates and fillRects it, which is cheap
  // - recreating gradients per frame would not be.
  var clouds = [];
  var CLOUD_N = Math.max(3, Math.min(6, Math.round(W / 480)));
  for (var ci = 0; ci < CLOUD_N; ci++) {
    var puffs = [];
    var puffCount = 4 + Math.floor(Math.random() * 4);
    var spread = 90 + Math.random() * 70;
    for (var pi = 0; pi < puffCount; pi++) {
      var dx = (Math.random() - 0.5) * spread * 1.7;
      var dy = (Math.random() - 0.5) * spread * 0.5;
      var rad = spread * (0.5 + Math.random() * 0.45);
      // Lighter, cooler-grey than the sky behind it, like moonlit cloud vapor -
      // a cloud tinted the same purple as its background has no contrast to
      // read by at all, which is what made the first pass invisible.
      var g = ctx.createRadialGradient(dx, dy, 0, dx, dy, rad);
      g.addColorStop(0,    'rgba(168,158,208,0.62)');
      g.addColorStop(0.55, 'rgba(120,105,165,0.34)');
      g.addColorStop(1,    'rgba(90,75,130,0)');
      puffs.push({ dx: dx, dy: dy, r: rad, grad: g });
    }
    clouds.push({
      x: Math.random() * W,
      y: H * (0.04 + Math.random() * 0.34),
      vx: 0.03 + Math.random() * 0.035,
      alpha: 0.6 + Math.random() * 0.3,
      puffs: puffs
    });
  }

  // Mouse position comes from the PARENT document: this iframe is forced
  // pointer-events:none (see theme CSS) so it never intercepts real clicks,
  // which means it also never receives mousemove directly. Same-origin
  // access to window.parent is what components.v1.html provides; if a
  // future Streamlit version sandboxes that away, this just throws once and
  // the animation quietly falls back to ambient drift below.
  var mouseX = -9999, mouseY = -9999, haveMouse = false;
  try {
    window.parent.document.addEventListener('mousemove', function (e) {
      mouseX = e.clientX; mouseY = e.clientY; haveMouse = true;
    });
    window.parent.document.addEventListener('mouseleave', function () {
      haveMouse = false;
    });
  } catch (err) { haveMouse = false; }

  var t = 0;
  function frame() {
    t += 0.016;
    ctx.clearRect(0, 0, W, H);
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      p.x += p.vx; p.y += p.vy;

      if (haveMouse) {
        var dx = p.x - mouseX, dy = p.y - mouseY;
        var d2 = dx * dx + dy * dy, R = 140;
        if (d2 < R * R) {
          var d = Math.sqrt(d2) || 1;
          var f = (1 - d / R) * 2.4;
          p.x += (dx / d) * f; p.y += (dy / d) * f;
        }
      } else {
        p.x += Math.sin(t * 0.3 + p.y * 0.01) * 0.25;
        p.y += Math.cos(t * 0.25 + p.x * 0.01) * 0.15;
      }

      if (p.x < -8) p.x = W + 8; if (p.x > W + 8) p.x = -8;
      if (p.y < -8) p.y = H + 8; if (p.y > H + 8) p.y = -8;

      var alpha = 0.72 + 0.28 * Math.sin(t * 0.8 + p.tw);
      ctx.save();
      // A modest, FIXED blur (not scaled with radius - a large blur spread
      // over a tiny circle dilutes per-pixel intensity to near-invisibility,
      // which is what made the first pass of this effect too subtle to read
      // as anything at normal viewing size).
      ctx.shadowBlur = 5;
      ctx.shadowColor = 'rgba(' + p.hue + ',1)';
      ctx.beginPath();
      ctx.fillStyle = 'rgba(' + p.hue + ',' + alpha + ')';
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Clouds draw AFTER stars so they visibly veil whatever star sits behind
    // them as they pass - the depth cue that sells "sky", not just "dots".
    for (var c = 0; c < clouds.length; c++) {
      var cl = clouds[c];
      cl.x += cl.vx;
      if (cl.x - 260 > W) cl.x = -260;
      ctx.save();
      ctx.translate(cl.x, cl.y);
      ctx.globalAlpha = cl.alpha;
      for (var pf = 0; pf < cl.puffs.length; pf++) {
        var puff = cl.puffs[pf];
        ctx.fillStyle = puff.grad;
        ctx.fillRect(puff.dx - puff.r, puff.dy - puff.r, puff.r * 2, puff.r * 2);
      }
      ctx.restore();
    }

    requestAnimationFrame(frame);
  }
  frame();
})();
</script>
</body></html>
"""


def inject_theme() -> None:
    """Call once near the top of the page. Never raises."""
    try:
        st.markdown(CSS, unsafe_allow_html=True)
    except Exception:
        pass
    try:
        components.html(_BACKGROUND_HTML, height=0, scrolling=False)
    except Exception:
        pass


def style_fig(fig):
    """
    Apply the dark/glass theme to a Plotly figure's layout only - trace colors
    and data are left alone, so this cannot change what a chart shows, only
    how the canvas around it looks.
    """
    try:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            # A darker, more opaque plot area than a first pass used - readable
            # data/gridlines regardless of what the animated sky is doing behind
            # the glass panel the chart sits in.
            plot_bgcolor="rgba(8,5,16,0.45)",
            font=dict(color="#f3f2fa", family="-apple-system, Segoe UI, Roboto, sans-serif", size=13),
            xaxis=dict(gridcolor="rgba(255,255,255,0.14)", zerolinecolor="rgba(255,255,255,0.16)",
                      tickfont=dict(color="#d8d4e8")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.14)", zerolinecolor="rgba(255,255,255,0.16)",
                      tickfont=dict(color="#d8d4e8")),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f3f2fa")),
            hoverlabel=dict(bgcolor="#1a0f2e", font_color="#f3f2fa",
                            bordercolor="rgba(255,255,255,0.22)"),
        )
    except Exception:
        pass
    return fig


def glass_tile(label: str, value: float, category: str, color: str, sub: str = "",
               hazard: bool = False) -> str:
    text = "#10131f" if category in ("Good", "Moderate", "Unhealthy for Sensitive Groups") else "#fff"
    hazard_class = " g-hazard" if hazard else ""
    return f"""
    <div class="glass-tile{hazard_class}" style="background:linear-gradient(160deg,{color}f0,{color}b8);color:{text};">
      <div class="g-label">{label}</div>
      <div class="g-value">{value:.0f}</div>
      <div class="g-cat">{category}</div>
      <div class="g-sub">{sub}</div>
    </div>"""


def glass_legend(items: dict) -> str:
    light_bg = ("Good", "Moderate", "Unhealthy for Sensitive Groups")
    chips = "".join(
        f'<span class="pill" style="background:{color};'
        f'color:{"#10131f" if name in light_bg else "#fff"};">{name}</span>'
        for name, color in items.items() if name != "Unknown"
    )
    return f'<div class="pill-row">{chips}</div>'
