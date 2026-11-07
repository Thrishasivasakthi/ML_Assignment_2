"""
components.layout - the design system: stylesheet, typography and page chrome.

Everything visual and shared lives here so that all six pages inherit one
vocabulary instead of each re-declaring colours and spacing.  The stylesheet
below is the single copy - pages must not embed their own <style> blocks.

Three typefaces by role:
    Georgia        display headlines
    Source Sans    body and widget text (set in .streamlit/config.toml)
    mono           every number, with tabular-nums so columns line up
"""

import streamlit as st

# ---------------------------------------------------------------- labels
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}

# One-line plain-English meaning of each metric, shown under its card.
METRIC_INFO = {
    "accuracy": "Share of rows classified correctly",
    "auc": "Ranking quality, threshold-free",
    "precision": "Of sessions flagged as buyers, the share that really were",
    "recall": "Of real buyers, how many we caught",
    "f1": "Harmonic mean of precision and recall",
    "mcc": "Correlation between prediction and truth",
}


CSS = """
<style>
/* ---------- primitives ---------------------------------------------- */
.s2s-eyebrow {
    font-size: .66rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: #9CA3AF;
}
.s2s-eyebrow.is-accent { color: #0F766E; }
.s2s-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.s2s-rule { border: 0; border-top: 1px solid #E5E7EB; margin: 1.15rem 0; }

/* ---------- masthead ------------------------------------------------- */
/* Display type: serif, deliberately.  Body and widget text are sans
   (set in .streamlit/config.toml) and every number is mono, so the page
   runs a three-typeface system - display / text / data - instead of
   Streamlit's single default face. */
.s2s-masthead h1 {
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    font-size: 2.8rem; font-weight: 700; letter-spacing: -.02em;
    color: #111827; margin: .55rem 0 .45rem; line-height: 1.06;
}
.s2s-lede {
    font-size: 1.02rem; color: #4B5563; max-width: 64ch;
    line-height: 1.6; margin: 0;
}
.s2s-stats { display: flex; flex-wrap: wrap; gap: 2.6rem; margin-top: .2rem; }
.s2s-stat .k {
    font-size: .62rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #9CA3AF;
}
.s2s-stat .v {
    font-size: 1.12rem; font-weight: 700; color: #111827; margin-top: .3rem;
    font-variant-numeric: tabular-nums;
}

/* ---------- section headers ----------------------------------------- */
.s2s-sec {
    display: flex; align-items: baseline; gap: .8rem;
    flex-wrap: wrap; margin-top: 2.4rem;
}
.s2s-sec .n {
    font-size: .78rem; font-weight: 700; color: #0F766E;
    font-family: ui-monospace, Menlo, monospace; letter-spacing: .04em;
}
.s2s-sec h2 {
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    font-size: 1.4rem; font-weight: 700; letter-spacing: -.01em;
    color: #111827; margin: 0;
}
.s2s-sec .sub { color: #6B7280; font-size: .88rem; font-style: italic; }

/* Inline chip.  NB: a raw code tag CANNOT be used in this app's HTML - Streamlit's
   markdown parser rewrites it into a full stCode BLOCK component (with a
   copy-to-clipboard button) that breaks the sentence across lines, and no
   amount of CSS on the inner element fixes it.  A span with our own class is
   the only reliable route to an inline chip. */
.s2s-code {
    display: inline;
    background: #FFFFFF; border: 1px solid #D1D5DB;
    border-radius: 4px; padding: 1px 5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .84em; color: #374151; white-space: nowrap;
}

/* ---------- empty state ---------------------------------------------- */
.s2s-empty {
    background: #F7F8FA; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 1.5rem 1.7rem; max-width: 62ch;
}
.s2s-empty p {
    font-size: .95rem; color: #4B5563; line-height: 1.65;
    margin: .6rem 0 0;
}

/* ---------- metric cards -------------------------------------------- */
.s2s-cards {
    display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 11px; margin: 1.1rem 0 .3rem;
}
.s2s-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 15px 14px 14px; box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.s2s-card .lab {
    font-size: .61rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: #6B7280;
}
.s2s-card .val {
    font-family: ui-monospace, Menlo, monospace; font-size: 1.5rem;
    font-weight: 600; color: #111827; margin-top: .6rem; letter-spacing: -.02em;
    font-variant-numeric: tabular-nums;
}
.s2s-card .dsc {
    font-size: .68rem; color: #9CA3AF; margin-top: .55rem; line-height: 1.45;
    min-height: 2.9em;
}
.s2s-baseline {
    font-size: .8rem; color: #6B7280; background: #F7F8FA;
    border: 1px solid #E5E7EB; border-left: 3px solid #0F766E;
    border-radius: 0 8px 8px 0; padding: .6rem .9rem; display: inline-block;
}

/* ---------- confusion matrix ---------------------------------------- */
.s2s-cm {
    display: grid; grid-template-columns: auto repeat(2, minmax(0, 1fr));
    gap: 8px;
}
.s2s-cm-h {
    font-size: .62rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #9CA3AF; text-align: center;
    display: flex; align-items: flex-end; justify-content: center;
    padding-bottom: 2px;
}
.s2s-cm-r {
    font-size: .74rem; font-weight: 600; color: #6B7280; text-align: right;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 6px; line-height: 1.25;
}
.s2s-cm-c {
    border-radius: 9px; padding: 17px 8px; text-align: center;
    border: 1px solid transparent;
}
.s2s-cm-c b {
    display: block; font-family: ui-monospace, Menlo, monospace;
    font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums;
}
.s2s-cm-c span {
    font-size: .6rem; letter-spacing: .1em; text-transform: uppercase;
    opacity: .75; display: block; margin-top: 3px;
}
.s2s-cm-ok  { background: #ECFDF5; border-color: #A7F3D0; color: #065F46; }
.s2s-cm-err { background: #FFF1F2; border-color: #FECDD3; color: #9F1239; }

/* ---------- tables --------------------------------------------------- */
.s2s-tbl { width: 100%; border-collapse: collapse; font-size: .87rem; }
.s2s-tbl th {
    font-size: .61rem; font-weight: 700; letter-spacing: .11em;
    text-transform: uppercase; color: #6B7280; padding: 0 8px 9px;
    border-bottom: 1px solid #E5E7EB; text-align: right; white-space: nowrap;
}
.s2s-tbl th.l, .s2s-tbl td.l { text-align: left; }
.s2s-tbl td {
    padding: 10px 8px; border-bottom: 1px solid #F3F4F6; text-align: right;
    font-variant-numeric: tabular-nums; color: #374151;
}
.s2s-tbl td.num { font-family: ui-monospace, Menlo, monospace; font-size: .84rem; }
.s2s-tbl tr.sel td { background: #F0FDFA; }
.s2s-tbl tr.sel td.l { color: #0F766E; font-weight: 700; }
.s2s-tbl td.best { color: #047857; font-weight: 700; }
.s2s-tbl td.worst { color: #BE123C; }

/* ---------- comparison bars ------------------------------------------ */
.s2s-bars { display: flex; flex-direction: column; gap: 9px; }
.s2s-bar {
    display: grid; grid-template-columns: 178px minmax(0, 1fr) 62px;
    align-items: center; gap: 14px;
}
.s2s-bar .nm { font-size: .85rem; color: #6B7280; }
.s2s-bar.on .nm { color: #111827; font-weight: 700; }
.s2s-bar .tk {
    height: 20px; background: #F3F4F6; border-radius: 5px; overflow: hidden;
}
.s2s-bar .fl { height: 100%; background: #D1D5DB; border-radius: 5px; }
.s2s-bar.on .fl { background: #0F766E; }
.s2s-bar .vl {
    font-family: ui-monospace, Menlo, monospace; font-size: .84rem;
    text-align: right; color: #374151; font-variant-numeric: tabular-nums;
}
.s2s-bar.on .vl { color: #0F766E; font-weight: 700; }

/* ---------- KPI tiles (overview) ------------------------------------- */
.s2s-kpis {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px; margin: 1.1rem 0 .3rem;
}
.s2s-kpi {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 16px 16px 15px; box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.s2s-kpi .k {
    font-size: .61rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: #6B7280;
}
.s2s-kpi .v {
    font-family: ui-monospace, Menlo, monospace; font-size: 1.6rem;
    font-weight: 600; color: #111827; margin-top: .55rem; letter-spacing: -.02em;
    font-variant-numeric: tabular-nums;
}
.s2s-kpi .v.text {
    font-family: Georgia, "Iowan Old Style", serif;
    font-size: 1.18rem; letter-spacing: -.01em;
}
.s2s-kpi .s { font-size: .7rem; color: #9CA3AF; margin-top: .45rem; }
.s2s-kpi.is-accent { background: #0F766E; border-color: #0F766E; }
.s2s-kpi.is-accent .k { color: #A7F3D0; }
.s2s-kpi.is-accent .v { color: #FFFFFF; }
.s2s-kpi.is-accent .s { color: #CCFBF1; }

/* ---------- sidebar -------------------------------------------------- */
/* Streamlit pins the rail with an inline `width: 256px`.  min-width and
   max-width are not in conflict with that - the used width is clamped to
   min(max(min, width), max) - but they are the only levers we have: a
   stylesheet `width` of ours would lose to an inline value outright, so the
   bounds are what is left.  Verified in 1.46: inline 256px, computed
   280px.  280px gives the uploader label and the classifier select a few
   pixels of air at the rail's edge without eating into the content area,
   which the .block-container rule below caps separately; the 300px ceiling
   stops a future Streamlit from widening the inline value and running away
   with the content area. */
section[data-testid="stSidebar"] {
    border-right: 1px solid #E5E7EB;
    min-width: 280px !important;
    max-width: 300px !important;
}
section[data-testid="stSidebar"] .s2s-eyebrow { margin-top: .2rem; }

/* Streamlit derives the sidebar label of the ENTRY script from its filename.
   The assignment brief requires that file to be named app.py, so the label is
   corrected here rather than by renaming the file.  The original text lives in
   a nested element with its own font-size, so both the anchor and every
   descendant are zeroed; ::after reprints it at normal size. Navigation itself
   is untouched - the link and its href are unchanged. */
[data-testid="stSidebarNav"] li:first-child a,
[data-testid="stSidebarNav"] li:first-child a * {
    font-size: 0 !important;
}
[data-testid="stSidebarNav"] li:first-child a::after {
    content: "Overview";
    font-size: .875rem !important;
}
div[data-testid="stButton"] > button {
    border-radius: 8px; border: 1px solid #E5E7EB; background: #FFFFFF;
    color: #374151; font-weight: 600; font-size: .83rem; padding: .45rem 1rem;
}
div[data-testid="stButton"] > button:hover {
    border-color: #0F766E; color: #0F766E; background: #F0FDFA;
}
div[data-testid="stFileUploader"] {
    background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 10px;
}

/* ---------- page chrome ----------------------------------------------
   Two separate problems, solved in one rule.

   VERTICAL.  Streamlit's own padding on this element is 6rem 1rem 10rem,
   declared on a hashed emotion class rather than on .block-container, so a
   page whose first element is always the masthead opens with 6rem of blank
   band above it.  2.6rem puts the header just under the toolbar.

   HORIZONTAL.  In the default centred layout Streamlit pins this element to
   736px (.st-emotion-cache-1w723zb in 1.46) and centres it, so the content
   never grows: measured against the main area it leaves 285px of blank
   margin down each side at a 1540px window and 475px at 1920px, always
   around a 704px column.  boot() now asks for layout="wide"; this rule
   decides what that actually means: a 1320px cap, as wide as the seven-
   column comparison table stays comfortably legible, and 3rem of side
   padding in place of Streamlit's 1rem.  After: 1540px -> 1154px of content
   behind a 48px gutter; 1920px -> 1224px behind 203px, which is 155px of
   centring plus that same 48px.

   !important because Streamlit declares its own padding on hashed emotion
   classes (.st-emotion-cache-zy6yx3 in 1.46), so nothing of ours would win
   on specificity - only on source order, which stops being true the moment
   Streamlit injects a <style> after ours.  Streamlit 1.46 renders exactly
   one .block-container and it lives inside stMain, so nothing else is
   affected today; the sidebar line below is a guard, kept because without
   it any Streamlit that DOES give the rail a block container would hand it
   this rule's 3rem side padding and quietly cost the rail 3rem of usable
   width. */
.block-container {
    padding: 2.6rem 3rem 3.5rem 3rem !important;
    max-width: 1320px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.7rem 1.5rem 2.5rem 1.5rem !important;
    max-width: 100% !important;
}

/* The uploader's own label is a Streamlit widget label, not one of our
   eyebrows, so it arrives in the default body face at body size and the
   rail reads as two typographies.  Restyled into the eyebrow vocabulary so
   "Dataset", "Test dataset (CSV)" and "Classifier" line up as one column of
   labels. */
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] [data-testid="stMarkdownLabel"] {
    font-size: .66rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: #9CA3AF; margin-bottom: .45rem;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
    margin-bottom: .3rem;
}
/* Streamlit's dropzone reserves a tall fixed band for the drag-and-drop
   hint.  In a rail that already carries a label above it and a status line
   below it, that band is most of the sidebar's empty space - tightened to
   the minimum the instructions still need. */
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
    min-height: 0; padding: .7rem .7rem .6rem;
}
section[data-testid="stSidebar"]
    div[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: .74rem; padding-top: .3rem;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin-bottom: .9rem;
}

/* ---------- responsive -------------------------------------------------
   With layout="wide" the content is no longer pinned to Streamlit's 736px,
   so the old arithmetic (viewport - 256 - 160) is wrong.  What a viewport
   actually receives is now:

       content = min(1320, viewport - 280 (sidebar) - 10 (scrollbar)) - 96 (padding)
               = min(1224, viewport - 386)

   giving 1920 -> 1224, 1440 -> 1054, 1280 -> 894, 1024 -> 638, 800 -> 414.

   The breakpoints below are keyed to that content width rather than to the
   viewport, because the content is what breaks:

     >= 1401   content >= 1015   -> 6 metric cards across, ~166px each
     <= 1400                       -> 3 cards across
     <= 1200                       -> the section-02 columns stack (see below)
     <= 1100                       -> compact table, matrix, bars and headline,
                                      for the genuinely narrow cases
     <=  900                       -> tighter table type and cell gutters, so
                                      the 7-column comparison table stays
                                      inside the text column
     <=  760                       -> 2 cards across
*/
@media (max-width: 1400px) {
    .s2s-cards { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .s2s-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
/* These columns stack at 1200 rather than at 1100, and that number is
   measured rather than chosen.  Side by side they split the content about
   46/54, and the classification report's table is 363px wide on its own; the
   right-hand column only reaches 363px once the content is ~765px, which
   this layout delivers at a viewport of ~1151.  1200 is the round number
   above that with ~38px of margin, and below it each column takes the whole
   content width, so the report cannot be clipped however narrow it gets. */
@media (max-width: 1200px) {
    [data-testid="stHorizontalBlock"] { row-gap: 1.75rem !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}
@media (max-width: 1100px) {
    /* The 7-column comparison table still measures ~457px at this smaller
       font, against 414px of content at an 800px viewport. */
    .s2s-tbl { font-size: .82rem; }
    .s2s-tbl td.num { font-size: .8rem; }

    .s2s-cm { gap: 5px; }
    .s2s-cm-c { padding: 13px 4px; }
    .s2s-cm-c b { font-size: 1.2rem; }
    .s2s-cm-c span { font-size: .55rem; letter-spacing: .06em; }
    .s2s-cm-r { font-size: .67rem; padding-right: 3px; }
    .s2s-cm-h { font-size: .57rem; letter-spacing: .06em; }

    .s2s-bar { grid-template-columns: 128px minmax(0, 1fr) 56px; gap: 9px; }
    .s2s-bar .nm { font-size: .79rem; }

    .s2s-masthead h1 { font-size: 2.2rem; }
}
/* Below a content width of ~843px the 7-column comparison table is wider
   than the text column: it measures 457px even at .82rem, and at an 800px
   viewport the content is only 414px.  Left alone the box runs 44px past the
   right edge of every other element on the page, into the block's side
   padding.  Tightening the type and the cell gutters here pulls it back to
   ~390px so all seven columns stay inside the column with everything else. */
@media (max-width: 900px) {
    .s2s-tbl { font-size: .76rem; }
    .s2s-tbl th { padding-left: 5px; padding-right: 5px; }
    .s2s-tbl td { padding-left: 5px; padding-right: 5px; }
    .s2s-tbl td.num { font-size: .74rem; }
}
@media (max-width: 760px) {
    .s2s-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .s2s-kpis { grid-template-columns: minmax(0, 1fr); }
    .s2s-stats { gap: 1.4rem; }
}
</style>
"""


def inject_css():
    """Install the design system.  Call once at the top of every page."""
    st.markdown(CSS, unsafe_allow_html=True)


def eyebrow(text, accent=False):
    cls = "s2s-eyebrow is-accent" if accent else "s2s-eyebrow"
    return f'<div class="{cls}">{text}</div>'


def rule():
    return '<hr class="s2s-rule">'


def section(number, title, subtitle=""):
    sub = f'<span class="sub">{subtitle}</span>' if subtitle else ""
    return (
        f'<div class="s2s-sec"><span class="n">{number}</span>'
        f"<h2>{title}</h2>{sub}</div>"
    )


def page_header(kicker, title, lede):
    """Editorial masthead: small accent kicker, serif headline, lede."""
    st.markdown(
        '<div class="s2s-masthead">'
        + eyebrow(kicker, accent=True)
        + f"<h1>{title}</h1>"
        + f'<p class="s2s-lede">{lede}</p>'
        + "</div>",
        unsafe_allow_html=True,
    )


def stat_strip(pairs):
    """A row of label/value facts under the masthead. pairs = [(k, v), ...]"""
    cells = "".join(
        f'<div class="s2s-stat"><div class="k">{k}</div>'
        f'<div class="v">{v}</div></div>'
        for k, v in pairs
    )
    st.markdown(f'<div class="s2s-stats">{cells}</div>', unsafe_allow_html=True)


def empty_state(body_html, heading="No dataset loaded"):
    """Panel shown when nothing has been uploaded yet."""
    st.markdown(
        rule()
        + '<div class="s2s-empty">'
        + eyebrow(heading)
        + f"<p>{body_html}</p>"
        + "</div>",
        unsafe_allow_html=True,
    )


def note(html):
    """The teal-edged context note used for baseline caveats."""
    st.markdown(f'<div class="s2s-baseline">{html}</div>', unsafe_allow_html=True)


def chip(text, wrap=False):
    """Inline monospace chip.  Never use a raw <code> tag - see CSS comment.

    wrap=True is for code longer than a few words: .s2s-code is nowrap by
    default so short chips never break mid-token, but that would overflow
    the column for something like a train_test_split(...) call.
    """
    style = ' style="white-space:normal"' if wrap else ""
    return f'<span class="s2s-code"{style}>{text}</span>'
