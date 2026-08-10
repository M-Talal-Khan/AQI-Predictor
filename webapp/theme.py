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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --glass: rgba(255,255,255,0.07);
  --glass-strong: rgba(255,255,255,0.11);
  --glass-border: rgba(255,255,255,0.16);
  --accent: #7dd3fc;
  --accent2: #c084fc;
  --text: #eaeef7;
  --text-dim: #a3adc2;
}

/* ---- sky backdrop on the real app surface ----
   The aurora blobs and the base gradient are BOTH part of this element's own
   background (not a separate positioned layer) so there is no stacking-context
   ambiguity to get wrong: an element's own background is always painted, full
   stop, before any question of z-index arises. The three radial layers are
   animated purely via background-position, which is enough motion to read as
   a slowly drifting sky without needing a second element at all. */
[data-testid="stAppViewContainer"] {
  background-image:
    radial-gradient(circle at 30% 30%, rgba(125,211,252,.32), transparent 42%),
    radial-gradient(circle at 70% 40%, rgba(192,132,252,.28), transparent 44%),
    radial-gradient(circle at 50% 80%, rgba(244,114,182,.20), transparent 46%),
    radial-gradient(ellipse 90% 55% at 12% -8%,  rgba(120,60,200,.35) 0%, transparent 60%),
    radial-gradient(ellipse 90% 55% at 88% 0%,    rgba(30,120,190,.30) 0%, transparent 60%),
    linear-gradient(180deg, #0a0e1c 0%, #121a33 45%, #1b1440 100%) !important;
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
  font-family: 'Inter', -apple-system, sans-serif;
}
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-dim) !important; }

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Space Grotesk', 'Inter', sans-serif !important;
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
.glass-tile .g-label { font-size: .8rem; font-weight: 600; opacity: .85; letter-spacing: .04em; text-transform: uppercase; }
.glass-tile .g-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1.05; margin: 2px 0; }
.glass-tile .g-cat { font-size: .82rem; font-weight: 700; }
.glass-tile .g-sub { font-size: .72rem; opacity: .82; margin-top: 2px; }

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
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
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
            plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#eaeef7", family="Inter, sans-serif"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.09)", zerolinecolor="rgba(255,255,255,0.09)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.09)", zerolinecolor="rgba(255,255,255,0.09)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor="#1c2540", font_color="#eaeef7",
                            bordercolor="rgba(255,255,255,0.18)"),
        )
    except Exception:
        pass
    return fig


def glass_tile(label: str, value: float, category: str, color: str, sub: str = "",
               hazard: bool = False) -> str:
    text = "#10131f" if category in ("Good", "Moderate", "Unhealthy for Sensitive Groups") else "#fff"
    hazard_class = " g-hazard" if hazard else ""
    return f"""
    <div class="glass-tile{hazard_class}" style="background:linear-gradient(160deg,{color}cc,{color}88);color:{text};">
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
