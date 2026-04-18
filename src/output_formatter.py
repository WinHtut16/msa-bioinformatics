"""
src/output_formatter.py
──────────────────────────────────────────────────────────────────────────────
Terminal output formatting for the MSA system.

Provides:
  - Colour-coded alignment display (conserved=green, mutation=yellow, gap=red)
  - Box-drawing result panels for Progressive MSA and Star Alignment
  - Side-by-side comparison mode
  - Alignment statistics summary

Works on any terminal that supports ANSI escape codes (Linux, macOS, Windows 10+).
Falls back to plain text automatically if colour is not supported.
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os


# ── Colour support detection ──────────────────────────────────────────────────

def _supports_colour():
    """Return True if the current terminal supports ANSI colour codes."""
    # Explicit disable via environment variable
    if os.environ.get("NO_COLOR") or os.environ.get("MSA_NO_COLOR"):
        return False
    # Not a real terminal (e.g. redirected to file)
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    # Windows: check for ANSI support (Windows 10 1511+ supports it)
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


COLOUR = _supports_colour()


# ── ANSI codes ────────────────────────────────────────────────────────────────

class _C:
    RESET   = "\033[0m"    if COLOUR else ""
    BOLD    = "\033[1m"    if COLOUR else ""
    DIM     = "\033[2m"    if COLOUR else ""

    # Foreground colours
    GREEN   = "\033[32m"   if COLOUR else ""
    YELLOW  = "\033[33m"   if COLOUR else ""
    RED     = "\033[31m"   if COLOUR else ""
    CYAN    = "\033[36m"   if COLOUR else ""
    WHITE   = "\033[97m"   if COLOUR else ""
    BLUE    = "\033[34m"   if COLOUR else ""
    MAGENTA = "\033[35m"   if COLOUR else ""
    GREY    = "\033[90m"   if COLOUR else ""

    # Background colours (for column highlighting)
    BG_GREEN  = "\033[42m" if COLOUR else ""
    BG_YELLOW = "\033[43m" if COLOUR else ""
    BG_RED    = "\033[41m" if COLOUR else ""
    BG_BLUE   = "\033[44m" if COLOUR else ""
    BG_RESET  = "\033[49m" if COLOUR else ""


# ── Box drawing ───────────────────────────────────────────────────────────────

_BOX = {
    "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝",
    "h":  "═", "v":  "║",
    "ml": "╠", "mr": "╣", "mt": "╦", "mb": "╩", "x": "╬",
    "sl": "├", "sr": "┤", "sh": "─",    # single-rule separator
}

WIDTH = 72  # total box inner width


def _box_top(title=""):
    inner = WIDTH - 2
    if title:
        pad = inner - len(title) - 2
        left  = pad // 2
        right = pad - left
        mid = f" {_C.BOLD}{_C.CYAN}{title}{_C.RESET} "
        return (f"{_C.BLUE}{_BOX['tl']}{_BOX['h'] * left}"
                f"{mid}"
                f"{_C.BLUE}{_BOX['h'] * right}{_BOX['tr']}{_C.RESET}")
    return f"{_C.BLUE}{_BOX['tl']}{_BOX['h'] * inner}{_BOX['tr']}{_C.RESET}"


def _box_bot():
    return f"{_C.BLUE}{_BOX['bl']}{_BOX['h'] * (WIDTH - 2)}{_BOX['br']}{_C.RESET}"


def _box_sep():
    return f"{_C.BLUE}{_BOX['ml']}{_BOX['sh'] * (WIDTH - 2)}{_BOX['mr']}{_C.RESET}"


def _box_row(text, pad_char=" "):
    """Pad `text` to fill one row inside the box, accounting for hidden ANSI codes."""
    visible_len = _visible_length(text)
    padding = WIDTH - 2 - visible_len
    padding = max(0, padding)
    return f"{_C.BLUE}{_BOX['v']}{_C.RESET} {text}{pad_char * padding}{_C.BLUE}{_BOX['v']}{_C.RESET}"


def _visible_length(text):
    """Length of text after stripping ANSI escape sequences."""
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    return len(ansi_escape.sub('', text))


# ── Alignment colouring ───────────────────────────────────────────────────────

ALPHABET = set("ACGT-")


def _column_type(column):
    """
    Classify a single alignment column.

    Returns:
      'conserved'  – all non-gap residues identical, no gaps
      'gap'        – at least one gap character present
      'mutation'   – multiple different non-gap residues present
    """
    residues = [c for c in column if c != '-']
    if len(column) != len(residues):     # at least one gap
        return 'gap'
    if len(set(residues)) == 1:
        return 'conserved'
    return 'mutation'


def _colour_char(char, col_type):
    """Apply ANSI colour to a single character based on its column type."""
    if char == '-':
        return f"{_C.RED}{_C.DIM}-{_C.RESET}"
    if col_type == 'conserved':
        return f"{_C.GREEN}{_C.BOLD}{char}{_C.RESET}"
    if col_type == 'mutation':
        return f"{_C.YELLOW}{char}{_C.RESET}"
    # gap column but this char is a residue (partial gap)
    return f"{_C.MAGENTA}{char}{_C.RESET}"


def _format_aligned_row(label, sequence, column_types, max_label_len):
    """Format one labelled alignment row with per-character colouring."""
    label_part = f"{_C.CYAN}{_C.BOLD}{label:<{max_label_len}}{_C.RESET}"
    seq_part   = " ".join(
        _colour_char(char, col_type)
        for char, col_type in zip(sequence, column_types)
    )
    return f"{label_part}  {seq_part}"


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_alignment_stats(aligned_sequences):
    """
    Compute column-level statistics for a set of aligned sequences.

    Returns a dict with:
      length        – total alignment length (number of columns)
      conserved     – number of fully conserved columns (no gaps, same residue)
      variable      – number of variable columns (no gaps, >1 distinct residue)
      gap_cols      – number of columns containing at least one gap
      pct_conserved – conserved / length * 100
      pct_variable  – variable / length * 100
      pct_gap       – gap_cols / length * 100
      column_types  – list of 'conserved'/'mutation'/'gap' per column
    """
    if not aligned_sequences:
        return {}

    length = len(aligned_sequences[0])
    conserved = variable = gap_cols = 0
    column_types = []

    for col_idx in range(length):
        column = [seq[col_idx] for seq in aligned_sequences]
        ct = _column_type(column)
        column_types.append(ct)
        if ct == 'conserved':
            conserved += 1
        elif ct == 'mutation':
            variable += 1
        else:
            gap_cols += 1

    return {
        "length":        length,
        "conserved":     conserved,
        "variable":      variable,
        "gap_cols":      gap_cols,
        "pct_conserved": 100 * conserved / length if length else 0,
        "pct_variable":  100 * variable  / length if length else 0,
        "pct_gap":       100 * gap_cols  / length if length else 0,
        "column_types":  column_types,
    }


# ── Public display functions ──────────────────────────────────────────────────

def print_alignment_result(
    aligned_sequences,
    labels,
    sp_score,
    runtime,
    algorithm_name="Progressive MSA",
    show_stats=True,
    max_cols_per_row=40,
):
    """
    Print a formatted, colour-coded alignment result inside a box.

    Parameters
    ----------
    aligned_sequences : list[str]
        The aligned sequences (all same length, gaps as '-').
    labels : list[str]
        Sequence labels matching aligned_sequences.
    sp_score : float
        Sum-of-Pairs score of the alignment.
    runtime : float
        Wall-clock runtime in seconds.
    algorithm_name : str
        Name shown in the box title.
    show_stats : bool
        Whether to print the column statistics block.
    max_cols_per_row : int
        Maximum alignment columns per printed row before wrapping.
    """
    stats        = compute_alignment_stats(aligned_sequences)
    column_types = stats["column_types"]
    max_label    = max(len(l) for l in labels) if labels else 8

    print()
    print(_box_top(algorithm_name))

    # ── Metadata row ──────────────────────────────────────────────────────
    meta = (
        f"{_C.BOLD}Sequences:{_C.RESET} {_C.WHITE}{len(aligned_sequences)}{_C.RESET}   "
        f"{_C.BOLD}Length:{_C.RESET} {_C.WHITE}{stats['length']} cols{_C.RESET}   "
        f"{_C.BOLD}SP Score:{_C.RESET} {_C.GREEN if sp_score >= 0 else _C.RED}"
        f"{sp_score:.1f}{_C.RESET}   "
        f"{_C.BOLD}Runtime:{_C.RESET} {_C.WHITE}{runtime:.3f}s{_C.RESET}"
    )
    print(_box_row(meta))
    print(_box_sep())

    # ── Alignment rows (wrapped) ──────────────────────────────────────────
    length = stats["length"]
    for start in range(0, length, max_cols_per_row):
        end     = min(start + max_cols_per_row, length)
        sub_ct  = column_types[start:end]

        # Column position ruler
        ruler_nums = ""
        for i in range(start, end):
            if (i + 1) % 10 == 0:
                ruler_nums += f"{_C.GREY}{(i+1):>2}{_C.RESET}"
            elif (i + 1) % 5 == 0:
                ruler_nums += f"{_C.GREY}·{_C.RESET} "
            else:
                ruler_nums += "  "
        ruler_label = " " * (max_label + 2)
        print(_box_row(f"{ruler_label}{ruler_nums}"))

        for label, seq in zip(labels, aligned_sequences):
            sub_seq = seq[start:end]
            row_str = _format_aligned_row(label, sub_seq, sub_ct, max_label)
            print(_box_row(row_str))

        if end < length:
            print(_box_row(""))   # blank spacer between wrapped chunks

    # ── Column statistics ─────────────────────────────────────────────────
    if show_stats:
        print(_box_sep())
        stat_title = f"{_C.BOLD}Column Statistics{_C.RESET}"
        print(_box_row(stat_title))
        print(_box_row(""))

        bar_width = 30

        def _bar(pct, colour):
            filled = int(round(pct / 100 * bar_width))
            empty  = bar_width - filled
            return f"{colour}{'█' * filled}{_C.GREY}{'░' * empty}{_C.RESET}"

        rows = [
            (f"{_C.GREEN}●{_C.RESET} Conserved",
             stats['conserved'], stats['pct_conserved'], _C.GREEN),
            (f"{_C.YELLOW}●{_C.RESET} Variable ",
             stats['variable'],  stats['pct_variable'],  _C.YELLOW),
            (f"{_C.RED}●{_C.RESET} Gap cols  ",
             stats['gap_cols'],  stats['pct_gap'],        _C.RED),
        ]
        for label_s, count, pct, colour in rows:
            bar   = _bar(pct, colour)
            right = f"{colour}{count:>4}{_C.RESET} cols  {_C.GREY}({pct:5.1f}%){_C.RESET}"
            line  = f"  {label_s}  {bar}  {right}"
            print(_box_row(line))

    print(_box_bot())
    print()


def print_comparison(
    prog_sequences, prog_labels, prog_score, prog_runtime,
    star_sequences, star_labels, star_score, star_runtime,
):
    """
    Print a compact side-by-side comparison panel for Progressive vs Star.
    Full alignment display is skipped; only metrics and statistics are shown.
    """
    prog_stats = compute_alignment_stats(prog_sequences)
    star_stats = compute_alignment_stats(star_sequences)

    print()
    print(_box_top("Progressive MSA  vs  Star Alignment"))

    # Header
    hdr = (
        f"{'Metric':<22}"
        f"{'Progressive MSA':>20}"
        f"  {'Star Alignment':>20}"
        f"  {'Winner':>10}"
    )
    print(_box_row(f"{_C.BOLD}{_C.CYAN}{hdr}{_C.RESET}"))
    print(_box_sep())

    def _winner(prog_val, star_val, higher_is_better=True):
        if higher_is_better:
            if prog_val > star_val:
                return f"{_C.GREEN}Progressive{_C.RESET}"
            elif star_val > prog_val:
                return f"{_C.YELLOW}Star      {_C.RESET}"
        else:  # lower is better (e.g. length)
            if prog_val < star_val:
                return f"{_C.GREEN}Progressive{_C.RESET}"
            elif star_val < prog_val:
                return f"{_C.YELLOW}Star      {_C.RESET}"
        return f"{_C.GREY}Equal     {_C.RESET}"

    def _fmt_row(label, prog_v, star_v, higher_better=True, fmt="{:.1f}"):
        p_str = fmt.format(prog_v)
        s_str = fmt.format(star_v)
        w_str = _winner(prog_v, star_v, higher_better)
        line = (
            f"{_C.BOLD}{label:<22}{_C.RESET}"
            f"{_C.WHITE}{p_str:>20}{_C.RESET}"
            f"  {_C.WHITE}{s_str:>20}{_C.RESET}"
            f"  {w_str:>10}"
        )
        print(_box_row(line))

    _fmt_row("SP Score",        prog_score,              star_score,              higher_better=True)
    _fmt_row("Alignment Length",prog_stats['length'],    star_stats['length'],    higher_better=False, fmt="{:d} cols")
    _fmt_row("Runtime (s)",     prog_runtime,            star_runtime,            higher_better=False, fmt="{:.3f}s")
    print(_box_sep())
    _fmt_row("Conserved cols",  prog_stats['conserved'], star_stats['conserved'], higher_better=True,  fmt="{:d}")
    _fmt_row("Variable cols",   prog_stats['variable'],  star_stats['variable'],  higher_better=False, fmt="{:d}")
    _fmt_row("Gap columns",     prog_stats['gap_cols'],  star_stats['gap_cols'],  higher_better=False, fmt="{:d}")

    # SP score difference callout
    print(_box_sep())
    diff = prog_score - star_score
    sign = "+" if diff >= 0 else ""
    colour = _C.GREEN if diff >= 0 else _C.RED
    pct  = abs(diff) / abs(star_score) * 100 if star_score != 0 else 0
    diff_line = (
        f"  SP Score difference: "
        f"{colour}{_C.BOLD}{sign}{diff:.1f} pts ({sign}{pct:.1f}%){_C.RESET}"
        f"  {'in favour of Progressive MSA' if diff > 0 else 'in favour of Star Alignment' if diff < 0 else '(equal)'}"
    )
    print(_box_row(diff_line))

    print(_box_bot())
    print()


def print_demo_header(m, l, mutation_rate, seed):
    """Print a compact header before the demo alignment output."""
    print()
    print(_box_top("MSA System  —  Demo Mode"))
    info = (
        f"  Sequences: {_C.WHITE}{m}{_C.RESET}   "
        f"Length: {_C.WHITE}{l} bp{_C.RESET}   "
        f"Mutation rate: {_C.WHITE}{mutation_rate}{_C.RESET}   "
        f"Seed: {_C.WHITE}{seed}{_C.RESET}"
    )
    print(_box_row(info))
    print(_box_bot())
    print()


def print_fasta_header(filepath, n_sequences):
    """Print a compact header for FASTA input mode."""
    print()
    print(_box_top("MSA System  —  FASTA Mode"))
    info = (
        f"  Input: {_C.WHITE}{filepath}{_C.RESET}   "
        f"Sequences loaded: {_C.WHITE}{n_sequences}{_C.RESET}"
    )
    print(_box_row(info))
    print(_box_bot())
    print()


def print_legend():
    """Print a one-line colour legend below any alignment display."""
    if not COLOUR:
        return
    print(
        f"  Legend:  "
        f"{_C.GREEN}█ Conserved{_C.RESET}   "
        f"{_C.YELLOW}█ Mutation{_C.RESET}   "
        f"{_C.RED}█ Gap{_C.RESET}   "
        f"{_C.MAGENTA}█ Partial gap{_C.RESET}"
    )
    print()


# ── Conserved region detection ────────────────────────────────────────────────

def find_conserved_regions(column_types, min_length=3):
    """
    Identify runs of consecutive conserved columns.

    A conserved region is a contiguous block of 'conserved' columns
    of at least `min_length` columns — these are the biologically
    most meaningful parts of the alignment (potential drug targets,
    functional domains, etc.).

    Parameters
    ----------
    column_types : list[str]
        Per-column type labels from compute_alignment_stats().
    min_length : int
        Minimum number of consecutive conserved columns to report.

    Returns
    -------
    list of dicts, each with keys:
        start  – 1-based start position
        end    – 1-based end position (inclusive)
        length – number of conserved columns in the run
    """
    regions = []
    in_run  = False
    run_start = 0

    for i, ct in enumerate(column_types):
        if ct == 'conserved':
            if not in_run:
                in_run    = True
                run_start = i
        else:
            if in_run:
                run_len = i - run_start
                if run_len >= min_length:
                    regions.append({
                        "start":  run_start + 1,   # convert to 1-based
                        "end":    i,               # inclusive 1-based
                        "length": run_len,
                    })
                in_run = False

    # Handle a conserved run that extends to the last column
    if in_run:
        run_len = len(column_types) - run_start
        if run_len >= min_length:
            regions.append({
                "start":  run_start + 1,
                "end":    len(column_types),
                "length": run_len,
            })

    return regions


# ── Standalone statistics summary panel ──────────────────────────────────────

def print_stats_summary(
    aligned_sequences,
    labels,
    algorithm_name="Alignment",
    min_conserved_run=3,
    show_regions=True,
):
    """
    Print a detailed standalone alignment statistics panel.

    This is separate from the compact bar chart inside print_alignment_result().
    It provides:
      - Per-residue frequency breakdown across the alignment
      - Conserved / variable / gap column counts with percentages
      - A minimap of column types across the full alignment width
      - A list of conserved regions (potential functional domains)

    Parameters
    ----------
    aligned_sequences : list[str]
        The aligned sequences (all same length).
    labels : list[str]
        Sequence labels.
    algorithm_name : str
        Used in the panel title.
    min_conserved_run : int
        Minimum consecutive conserved columns to report as a region.
    show_regions : bool
        Whether to print the conserved region list.
    """
    stats  = compute_alignment_stats(aligned_sequences)
    ctypes = stats["column_types"]
    length = stats["length"]

    if length == 0:
        print("  (no alignment to summarise)")
        return

    print()
    print(_box_top(f"Alignment Statistics  —  {algorithm_name}"))

    # ── Section 1: Overview counts ────────────────────────────────────────
    bar_width = 36

    def _bar(pct, colour):
        filled = int(round(pct / 100 * bar_width))
        empty  = bar_width - filled
        return f"{colour}{'█' * filled}{_C.GREY}{'░' * empty}{_C.RESET}"

    overview_title = f"  {_C.BOLD}{_C.CYAN}Column Classification  "  \
                     f"{_C.GREY}({length} total columns){_C.RESET}"
    print(_box_row(overview_title))
    print(_box_row(""))

    rows = [
        ("Conserved ", stats['conserved'], stats['pct_conserved'], _C.GREEN,
         "All sequences agree, no gaps  →  potential drug / vaccine target"),
        ("Variable  ", stats['variable'],  stats['pct_variable'],  _C.YELLOW,
         "Multiple residues present, no gaps  →  mutation site"),
        ("Gap cols  ", stats['gap_cols'],  stats['pct_gap'],        _C.RED,
         "At least one gap character  →  insertion / deletion event"),
    ]
    for lbl, count, pct, colour, meaning in rows:
        bar   = _bar(pct, colour)
        right = f"{colour}{_C.BOLD}{count:>4}{_C.RESET} {_C.GREY}({pct:5.1f}%){_C.RESET}"
        line  = f"  {colour}●{_C.RESET} {_C.BOLD}{lbl}{_C.RESET}  {bar}  {right}"
        print(_box_row(line))
        meaning_line = f"      {_C.GREY}{meaning}{_C.RESET}"
        print(_box_row(meaning_line))
        print(_box_row(""))

    # ── Section 2: Residue frequency across whole alignment ───────────────
    print(_box_sep())
    print(_box_row(f"  {_C.BOLD}{_C.CYAN}Residue Frequency Across All Columns{_C.RESET}"))
    print(_box_row(""))

    total_chars = length * len(aligned_sequences)
    counts = {"A": 0, "C": 0, "G": 0, "T": 0, "-": 0, "other": 0}
    for seq in aligned_sequences:
        for ch in seq.upper():
            if ch in counts:
                counts[ch] += 1
            else:
                counts["other"] += 1

    res_bar_w = 28
    for residue, colour in [("A", _C.GREEN), ("C", _C.BLUE),
                             ("G", _C.YELLOW), ("T", _C.MAGENTA),
                             ("-", _C.RED)]:
        cnt = counts[residue]
        pct = 100 * cnt / total_chars if total_chars else 0
        filled = int(round(pct / 100 * res_bar_w))
        empty  = res_bar_w - filled
        bar    = f"{colour}{'█' * filled}{_C.GREY}{'░' * empty}{_C.RESET}"
        line   = (f"    {colour}{_C.BOLD}{residue}{_C.RESET}   {bar}  "
                  f"{colour}{cnt:>6}{_C.RESET} {_C.GREY}({pct:5.1f}%){_C.RESET}")
        print(_box_row(line))

    if counts["other"] > 0:
        pct = 100 * counts["other"] / total_chars
        print(_box_row(f"    {_C.GREY}other  {counts['other']:>6} ({pct:.1f}%){_C.RESET}"))

    # ── Section 3: Alignment minimap ──────────────────────────────────────
    print(_box_sep())
    print(_box_row(f"  {_C.BOLD}{_C.CYAN}Column Type Minimap{_C.RESET}  "
                   f"{_C.GREY}(each character = 1 column){_C.RESET}"))
    print(_box_row(""))

    # Build minimap — compress to fit inside box if alignment is long
    map_width   = WIDTH - 6   # usable characters inside the box
    step        = max(1, length // map_width)
    minimap_str = ""

    for i in range(0, length, step):
        # For multi-column steps, use the majority type in that window
        window = ctypes[i: i + step]
        majority = max(set(window), key=window.count)
        if majority == 'conserved':
            minimap_str += f"{_C.GREEN}█{_C.RESET}"
        elif majority == 'mutation':
            minimap_str += f"{_C.YELLOW}▒{_C.RESET}"
        else:
            minimap_str += f"{_C.RED}░{_C.RESET}"

    # Print with position markers every 10 visible chars
    print(_box_row(f"  {minimap_str}"))
    print(_box_row(""))

    # Position ruler under minimap
    ruler = ""
    visible_pos = 0
    for i in range(0, length, step):
        real_pos = i + 1
        if visible_pos % 10 == 0 and visible_pos > 0:
            marker = str(real_pos)
            ruler += f"{_C.GREY}{marker}{_C.RESET}"
            visible_pos += len(marker)
        else:
            ruler += " "
            visible_pos += 1
    print(_box_row(f"  {ruler}"))

    # ── Section 4: Conserved regions ─────────────────────────────────────
    if show_regions:
        print(_box_sep())
        regions = find_conserved_regions(ctypes, min_length=min_conserved_run)

        region_title = (
            f"  {_C.BOLD}{_C.CYAN}Conserved Regions"
            f"{_C.RESET}  {_C.GREY}"
            f"(runs of ≥ {min_conserved_run} consecutive conserved columns){_C.RESET}"
        )
        print(_box_row(region_title))
        print(_box_row(""))

        if not regions:
            print(_box_row(
                f"  {_C.GREY}No conserved runs of ≥ {min_conserved_run} "
                f"columns found.  Sequences may be too divergent.{_C.RESET}"
            ))
        else:
            # Sort by length descending so the most conserved region is first
            regions_sorted = sorted(regions, key=lambda r: r["length"], reverse=True)

            hdr = (f"  {'#':<4} {'Start':>7} {'End':>7} {'Length':>8}  "
                   f"{'Significance':<30}")
            print(_box_row(f"{_C.BOLD}{_C.GREY}{hdr}{_C.RESET}"))

            for idx, reg in enumerate(regions_sorted, 1):
                length_r = reg["length"]
                # Assign biological significance label based on length
                if length_r >= 20:
                    sig   = "★★★ Major conserved domain"
                    scol  = _C.GREEN
                elif length_r >= 10:
                    sig   = "★★  Significant conserved block"
                    scol  = _C.GREEN
                elif length_r >= 5:
                    sig   = "★   Moderately conserved region"
                    scol  = _C.YELLOW
                else:
                    sig   = "    Short conserved run"
                    scol  = _C.GREY

                row = (
                    f"  {_C.GREY}{idx:<4}{_C.RESET}"
                    f" {_C.WHITE}{reg['start']:>7}{_C.RESET}"
                    f" {_C.WHITE}{reg['end']:>7}{_C.RESET}"
                    f" {_C.GREEN}{_C.BOLD}{length_r:>8}{_C.RESET}"
                    f"  {scol}{sig}{_C.RESET}"
                )
                print(_box_row(row))

            print(_box_row(""))
            total_conserved_in_regions = sum(r["length"] for r in regions)
            coverage = 100 * total_conserved_in_regions / length if length else 0
            summary_line = (
                f"  {len(regions)} region(s) found  ·  "
                f"{_C.GREEN}{total_conserved_in_regions}{_C.RESET} conserved columns in regions  ·  "
                f"{_C.CYAN}{coverage:.1f}%{_C.RESET} of alignment length"
            )
            print(_box_row(summary_line))

    print(_box_bot())
    print()