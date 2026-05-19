#==========================================================
#Ten plik czyta historię od najstarszej do najnowszej
#Gdzie najnowsza jest na samym końcu
#==========================================================

import re
import itertools
from collections import Counter, defaultdict
from datetime import datetime

import fitz
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Lotto Machine Memory Engine",
    page_icon="🎰",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

CUSTOM_CSS = """
<style>

.stApp {
    background:
    radial-gradient(circle at top, #1f3b73 0%, #0f1729 45%, #05070d 100%);
    color: #f5f7ff;
}

h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 900 !important;
}

.card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 15px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
}

.big-title {
    font-size: 34px;
    font-weight: 900;
    color: #ffe66d;
}

.green {
    color: #39ff88;
    font-weight: 800;
}

.yellow {
    color: #ffd166;
    font-weight: 800;
}

.red {
    color: #ff6b6b;
    font-weight: 800;
}

[data-testid="stMetricValue"] {
    color: #ffe66d;
}

.stButton > button {
    background: linear-gradient(90deg, #ffe66d, #ff9f1c);
    color: black;
    border: none;
    border-radius: 14px;
    font-weight: 900;
    font-size: 18px;
    padding: 12px 22px;
}

.stDownloadButton > button {
    background: linear-gradient(90deg, #39ff88, #00c853);
    color: black;
    border: none;
    border-radius: 14px;
    font-weight: 900;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# PDF PARSER
# =========================================================

def extract_draws_from_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    rows = []

    for page_index, page in enumerate(doc):
        words = page.get_text("words")

        items = []

        for w in words:
            x0, y0, x1, y1, text = w[:5]

            if re.fullmatch(r"\d{1,4}", text):
                items.append((x0, y0, text))

        items = sorted(items, key=lambda x: (x[1], x[0]))

        line_groups = []
        tolerance_y = 3.0

        for item in items:
            x, y, text = item

            if not line_groups:
                line_groups.append({
                    "y": y,
                    "items": [item]
                })
            else:
                last_group = line_groups[-1]

                if abs(last_group["y"] - y) <= tolerance_y:
                    last_group["items"].append(item)
                    last_group["y"] = sum(i[1] for i in last_group["items"]) / len(last_group["items"])
                else:
                    line_groups.append({
                        "y": y,
                        "items": [item]
                    })

        for group in line_groups:
            line_items = sorted(group["items"], key=lambda x: x[0])

            draw_ids = []
            nums = []

            for x, y, text in line_items:
                value = int(text)

                if 1000 <= value <= 9999 and x < 35:
                    draw_ids.append(value)

                elif 1 <= value <= 49 and x > 25:
                    nums.append(value)

            if len(draw_ids) == 1 and len(nums) == 6 and len(set(nums)) == 6:
                nums = sorted(nums)

                rows.append({
                    "page": page_index + 1,
                    "draw_id": draw_ids[0],
                    "numbers": nums,
                    "n1": nums[0],
                    "n2": nums[1],
                    "n3": nums[2],
                    "n4": nums[3],
                    "n5": nums[4],
                    "n6": nums[5],
                })

    if not rows:
        raise ValueError("Nie udało się odczytać losowań z PDF.")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["draw_id"])
    df = df.sort_values("draw_id", ascending=True)
    df = df.reset_index(drop=True)

    return df

# =========================================================
# HOT NUMBERS
# =========================================================

def number_frequency(df: pd.DataFrame) -> pd.DataFrame:

    counter = Counter()

    for nums in df["numbers"]:
        counter.update(nums)

    rows = []

    total = len(df)

    for n in range(1, 50):

        hits = counter[n]

        rows.append({
            "number": n,
            "hits": hits,
            "percent": round(hits / total * 100, 2)
        })

    return pd.DataFrame(rows).sort_values(
        ["hits", "number"],
        ascending=[False, True]
    )


# =========================================================
# COLD NUMBERS
# =========================================================

def gap_analysis(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for n in range(1, 50):

        indexes = []

        for i, nums in enumerate(df["numbers"]):

            if n in nums:
                indexes.append(i)

        if indexes:

            current_gap = len(df) - 1 - indexes[-1]

            gaps = np.diff(indexes).tolist()

            avg_gap = np.mean(gaps) if gaps else 0

            max_gap = max(gaps) if gaps else 0

        else:

            current_gap = len(df)

            avg_gap = 0

            max_gap = 0

        rows.append({
            "number": n,
            "current_gap": current_gap,
            "avg_gap": round(avg_gap, 2),
            "max_gap": max_gap
        })

    return pd.DataFrame(rows).sort_values(
        "current_gap",
        ascending=False
    )


# =========================================================
# PAIR STRENGTH
# =========================================================

def pair_strength(df: pd.DataFrame) -> pd.DataFrame:

    pair_counter = Counter()

    for nums in df["numbers"]:

        for pair in itertools.combinations(nums, 2):
            pair_counter[pair] += 1

    rows = []

    for pair, count in pair_counter.items():

        rows.append({
            "pair": f"{pair[0]:02d}-{pair[1]:02d}",
            "count": count,
            "strength_percent": round(
                count / len(df) * 100,
                2
            )
        })

    return pd.DataFrame(rows).sort_values(
        "count",
        ascending=False
    )


# =========================================================
# TRANSITIONS
# =========================================================

def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:

    transitions = defaultdict(Counter)

    for i in range(1, len(df)):

        prev_nums = df.loc[i - 1, "numbers"]

        curr_nums = df.loc[i, "numbers"]

        for a in prev_nums:
            for b in curr_nums:
                transitions[a][b] += 1

    rows = []

    for a in range(1, 50):

        total = sum(transitions[a].values())

        if total == 0:
            continue

        for b, count in transitions[a].items():

            rows.append({
                "from": a,
                "to": b,
                "count": count,
                "probability": round(
                    count / total * 100,
                    2
                )
            })

    return pd.DataFrame(rows)


# =========================================================
# MOVEMENT HISTORY
# =========================================================

def movement_history(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for i in range(1, len(df)):

        prev_set = set(df.loc[i - 1, "numbers"])

        curr_set = set(df.loc[i, "numbers"])

        repeated = sorted(prev_set & curr_set)

        new_nums = sorted(curr_set - prev_set)

        lost_nums = sorted(prev_set - curr_set)

        prev_sum = sum(prev_set)

        curr_sum = sum(curr_set)

        dynamic = (
            len(new_nums)
            + abs(curr_sum - prev_sum) / 10
        )

        rows.append({
            "draw": int(df.loc[i, "draw_id"]),
            "previous_draw": int(df.loc[i - 1, "draw_id"]),
            "repeated": " ".join(f"{x:02d}" for x in repeated),
            "new_numbers": " ".join(f"{x:02d}" for x in new_nums),
            "lost_numbers": " ".join(f"{x:02d}" for x in lost_nums),
            "sum_change": curr_sum - prev_sum,
            "dynamic": round(dynamic, 2)
        })

    return pd.DataFrame(rows).sort_values(
        "draw",
        ascending=False
    )


# =========================================================
# MACHINE MEMORY
# =========================================================

def machine_memory_score(df: pd.DataFrame) -> pd.DataFrame:

    freq = number_frequency(df)

    gaps = gap_analysis(df)

    recent20 = df.tail(20)

    recent50 = df.tail(50)

    c20 = Counter()

    c50 = Counter()

    for nums in recent20["numbers"]:
        c20.update(nums)

    for nums in recent50["numbers"]:
        c50.update(nums)

    rows = []

    for n in range(1, 50):

        total_hits = int(
            freq.loc[
                freq["number"] == n,
                "hits"
            ].iloc[0]
        )

        gap = int(
            gaps.loc[
                gaps["number"] == n,
                "current_gap"
            ].iloc[0]
        )

        hot_score = total_hits / len(df)

        momentum20 = c20[n] / max(1, len(recent20))

        momentum50 = c50[n] / max(1, len(recent50))

        cold_pressure = min(gap / 50, 1.5)

        score = (
            hot_score * 40
            + momentum20 * 30
            + momentum50 * 20
            + cold_pressure * 10
        )

        rows.append({
            "number": n,
            "hits_total": total_hits,
            "hits_20": c20[n],
            "hits_50": c50[n],
            "gap": gap,
            "memory_score": round(score, 4)
        })

    return pd.DataFrame(rows).sort_values(
        "memory_score",
        ascending=False
    )


# =========================================================
# GENERATE COUPON
# =========================================================

def generate_coupon(memory_df, mode):

    rng = np.random.default_rng()

    if mode == "MEMORY":

        pool = memory_df.head(24).copy()

        weights = pool["memory_score"].to_numpy()

        weights = weights / weights.sum()

        chosen = rng.choice(
            pool["number"].to_numpy(),
            size=6,
            replace=False,
            p=weights
        )

    elif mode == "HOT":

        pool = memory_df.sort_values(
            "hits_total",
            ascending=False
        ).head(20)

        chosen = rng.choice(
            pool["number"].to_numpy(),
            size=6,
            replace=False
        )

    elif mode == "COLD":

        pool = memory_df.sort_values(
            "gap",
            ascending=False
        ).head(24)

        chosen = rng.choice(
            pool["number"].to_numpy(),
            size=6,
            replace=False
        )

    else:

        hot = memory_df.sort_values(
            "hits_total",
            ascending=False
        ).head(18)["number"].tolist()

        cold = memory_df.sort_values(
            "gap",
            ascending=False
        ).head(18)["number"].tolist()

        memory = memory_df.head(18)["number"].tolist()

        pool = list(dict.fromkeys(hot + cold + memory))

        chosen = rng.choice(
            pool,
            size=6,
            replace=False
        )

    return sorted(int(x) for x in chosen)


# =========================================================
# SCORE
# =========================================================

def coupon_score(
    coupon,
    pair_df,
    memory_df,
    transition_df,
    move_df
):

    pair_map = {}

    for _, row in pair_df.iterrows():

        a, b = row["pair"].split("-")

        pair_map[(int(a), int(b))] = row["count"]

    memory_map = dict(
        zip(
            memory_df["number"],
            memory_df["memory_score"]
        )
    )

    transition_score = 0

    top_transitions = transition_df.sort_values(
        "count",
        ascending=False
    ).head(300)

    for _, row in top_transitions.iterrows():

        if (
            int(row["from"]) in coupon
            and int(row["to"]) in coupon
        ):
            transition_score += float(row["count"])

    pair_points = 0

    for a, b in itertools.combinations(coupon, 2):

        pair_points += pair_map.get((a, b), 0)

    memory_points = sum(
        memory_map.get(n, 0)
        for n in coupon
    )

    recent_dynamic = move_df.head(30)["dynamic"].mean()

    dynamic_bonus = min(recent_dynamic, 20)

    spread = max(coupon) - min(coupon)

    even_count = sum(
        1 for n in coupon
        if n % 2 == 0
    )

    low_count = sum(
        1 for n in coupon
        if n <= 24
    )

    balance_bonus = 0

    if even_count in [2, 3, 4]:
        balance_bonus += 12

    if low_count in [2, 3, 4]:
        balance_bonus += 12

    if 25 <= spread <= 45:
        balance_bonus += 12

    final_score = (
        memory_points * 9.0
        + pair_points * 0.85
        + transition_score * 0.35
        + dynamic_bonus
        + balance_bonus
    )

    return round(final_score, 2)


# =========================================================
# TXT REPORT
# =========================================================

def create_report(
    df,
    freq_df,
    gap_df,
    pair_df,
    move_df,
    generated
):

    lines = []

    lines.append("LOTTO MACHINE MEMORY ENGINE")
    lines.append("=" * 50)

    lines.append(
        f"Generated: {datetime.now()}"
    )

    lines.append(
        f"Draws loaded: {len(df)}"
    )

    lines.append("")

    latest = df.iloc[-1]

    latest_result = " ".join(
        f"{x:02d}"
        for x in latest["numbers"]
    )

    lines.append(
        f"LATEST DRAW: {latest['draw_id']} -> {latest_result}"
    )

    lines.append("")

    lines.append("TOP HOT NUMBERS")

    for _, r in freq_df.head(15).iterrows():

        lines.append(
            f"{int(r['number']):02d} | hits={r['hits']} | {r['percent']}%"
        )

    lines.append("")

    lines.append("TOP COLD NUMBERS")

    for _, r in gap_df.head(15).iterrows():

        lines.append(
            f"{int(r['number']):02d} | gap={r['current_gap']}"
        )

    lines.append("")

    lines.append("TOP PAIRS")

    for _, r in pair_df.head(20).iterrows():

        lines.append(
            f"{r['pair']} | count={r['count']}"
        )

    lines.append("")

    lines.append("GENERATED RESULTS")

    for i, item in enumerate(generated, 1):

        nums = " ".join(
            f"{x:02d}"
            for x in item["numbers"]
        )

        lines.append(
            f"{i}. {nums} | POWER={item['score']}"
        )

    lines.append("")

    lines.append("RECENT MOVEMENT DYNAMICS")

    for _, r in move_df.head(20).iterrows():

        lines.append(
            f"{r['previous_draw']} -> {r['draw']} | "
            f"Repeated: {r['repeated']} | "
            f"New: {r['new_numbers']} | "
            f"Dynamic={r['dynamic']}"
        )

    return "\n".join(lines)


# =========================================================
# UI
# =========================================================

st.markdown(
    '<div class="big-title">🎰 LOTTO MACHINE MEMORY ENGINE</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">
    Machine analyzes:
    <br><br>

    ✔ HOT numbers<br>
    ✔ COLD numbers<br>
    ✔ Draw transitions<br>
    ✔ Pair strength<br>
    ✔ Machine memory<br>
    ✔ Movement history<br>
    ✔ Dynamics of change<br>
    ✔ Full historical movement of balls<br>
    ✔ Most probable 6-number result
    </div>
    """,
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload Lotto PDF",
    type=["pdf"]
)

if uploaded is None:

    st.info(
        "Upload PDF file."
    )

    st.stop()

# =========================================================
# LOAD
# =========================================================

try:

    pdf_bytes = uploaded.read()

    df = extract_draws_from_pdf(pdf_bytes)

except Exception as e:

    st.error(f"ERROR: {e}")

    st.stop()

freq_df = number_frequency(df)

gap_df = gap_analysis(df)

pair_df = pair_strength(df)

transition_df = transition_matrix(df)

move_df = movement_history(df)

memory_df = machine_memory_score(df)

# =========================================================
# METRICS
# =========================================================

latest = df.iloc[-1]

latest_result = " ".join(
    f"{x:02d}"
    for x in latest["numbers"]
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Draws",
    len(df)
)

m2.metric(
    "First draw",
    int(df["draw_id"].min())
)

m3.metric(
    "Latest draw",
    int(df["draw_id"].max())
)

m4.metric(
    "Latest result",
    latest_result
)

# =========================================================
# TABLES
# =========================================================

st.markdown("## 🧠 MACHINE MEMORY")

st.dataframe(
    memory_df.head(20),
    use_container_width=True,
    hide_index=True
)

st.markdown("## 🔥 HOT NUMBERS")

st.dataframe(
    freq_df.head(20),
    use_container_width=True,
    hide_index=True
)

st.markdown("## ❄️ COLD NUMBERS")

st.dataframe(
    gap_df.head(20),
    use_container_width=True,
    hide_index=True
)

st.markdown("## 🔗 STRONGEST PAIRS")

st.dataframe(
    pair_df.head(40),
    use_container_width=True,
    hide_index=True
)

st.markdown("## 🔄 MOVEMENT HISTORY")

st.dataframe(
    move_df.head(60),
    use_container_width=True,
    hide_index=True
)

st.markdown("## ➡️ TRANSITIONS")

st.dataframe(
    transition_df.sort_values(
        "count",
        ascending=False
    ).head(100),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# GENERATOR
# =========================================================

st.markdown("## 🎯 MOST PROBABLE RESULT GENERATOR")

c1, c2, c3 = st.columns(3)

with c1:

    mode = st.selectbox(
        "Generation mode",
        [
            "MEMORY",
            "HOT",
            "COLD",
            "HYBRID"
        ]
    )

with c2:

    attempts = st.slider(
        "Attempts",
        100,
        5000,
        1000
    )

with c3:

    results_count = st.slider(
        "Results count",
        1,
        20,
        5
    )

generated_results = []

if st.button(
    "🎰 GENERATE MOST PROBABLE RESULT"
):

    for _ in range(results_count):

        best_coupon = None

        best_score = -1

        for _ in range(attempts):

            coupon = generate_coupon(
                memory_df,
                mode
            )

            score = coupon_score(
                coupon,
                pair_df,
                memory_df,
                transition_df,
                move_df
            )

            if score > best_score:

                best_score = score

                best_coupon = coupon

        generated_results.append({
            "numbers": best_coupon,
            "score": best_score
        })

    st.session_state["generated_results"] = generated_results

# =========================================================
# SHOW RESULTS
# =========================================================

if "generated_results" in st.session_state:

    generated_results = st.session_state[
        "generated_results"
    ]

    rows = []

    for i, item in enumerate(
        generated_results,
        1
    ):

        rows.append({
            "result": i,
            "numbers": " ".join(
                f"{x:02d}"
                for x in item["numbers"]
            ),
            "power": item["score"]
        })

    result_df = pd.DataFrame(rows)

    st.markdown(
        """
        <div class="card">
        <span class="green">
        MACHINE FOUND THE STRONGEST HISTORICAL CONFIGURATION
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    report = create_report(
        df,
        freq_df,
        gap_df,
        pair_df,
        move_df,
        generated_results
    )

    st.download_button(
        label="💾 DOWNLOAD TXT REPORT",
        data=report.encode("utf-8"),
        file_name="lotto_machine_report.txt",
        mime="text/plain"
    )

# =========================================================
# FULL HISTORY
# =========================================================

st.markdown("## 📜 FULL DRAW HISTORY")

history_df = df.copy()

history_df["result"] = history_df["numbers"].apply(
    lambda x: " ".join(
        f"{n:02d}"
        for n in x
    )
)

st.dataframe(
    history_df[
        [
            "draw_id",
            "result",
            "page"
        ]
    ],
    use_container_width=True,
    hide_index=True
)
