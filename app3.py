#======================================================================================================================================
# Najnowsza aktualizacja pliku - nowe rozumowanie i metoda analizy
# Wersja Pro dostosowana pod PC wedle danych udostępnionych (RAM, KARTA, ITD)
#======================================================================================================================================

import re
import itertools
from collections import Counter, defaultdict
from datetime import datetime

import fitz
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Lotto Machine Memory Engine PRO",
    page_icon="🎰",
    layout="wide"
)


CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at top, #1f3b73 0%, #0f1729 45%, #05070d 100%);
    color: #f5f7ff;
}
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 900 !important;
}
.card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,0.075);
    border: 1px solid rgba(255,255,255,0.14);
    margin-bottom: 15px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
}
.big-title {
    font-size: 36px;
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
                line_groups.append({"y": y, "items": [item]})
            else:
                last_group = line_groups[-1]

                if abs(last_group["y"] - y) <= tolerance_y:
                    last_group["items"].append(item)
                    last_group["y"] = sum(i[1] for i in last_group["items"]) / len(last_group["items"])
                else:
                    line_groups.append({"y": y, "items": [item]})

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
    df = df.sort_values("draw_id", ascending=False)
    df = df.reset_index(drop=True)

    return df


def number_frequency(df: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()

    for nums in df["numbers"]:
        counter.update(nums)

    rows = []
    total = len(df)

    for n in range(1, 50):
        hits = counter[n]
        rows.append({
            "liczba": n,
            "trafienia": hits,
            "procent": round(hits / total * 100, 2)
        })

    return pd.DataFrame(rows).sort_values(["trafienia", "liczba"], ascending=[False, True])


def gap_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for n in range(1, 50):
        hit_indexes = []

        for i, nums in enumerate(df["numbers"]):
            if n in nums:
                hit_indexes.append(i)

        if hit_indexes:
            current_gap = hit_indexes[0]
            gaps = np.diff(hit_indexes).tolist()
            avg_gap = np.mean(gaps) if gaps else 0
            max_gap = max(gaps) if gaps else 0
        else:
            current_gap = len(df)
            avg_gap = 0
            max_gap = 0

        rows.append({
            "liczba": n,
            "przerwa_od_najnowszego": current_gap,
            "średnia_przerwa": round(avg_gap, 2),
            "maksymalna_przerwa": max_gap
        })

    return pd.DataFrame(rows).sort_values("przerwa_od_najnowszego", ascending=False)


def pair_strength(df: pd.DataFrame) -> pd.DataFrame:
    pair_counter = Counter()

    for nums in df["numbers"]:
        for pair in itertools.combinations(nums, 2):
            pair_counter[pair] += 1

    rows = []

    for pair, count in pair_counter.items():
        rows.append({
            "para": f"{pair[0]:02d}-{pair[1]:02d}",
            "wystąpienia": count,
            "siła_procent": round(count / len(df) * 100, 2)
        })

    return pd.DataFrame(rows).sort_values("wystąpienia", ascending=False)


def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    transitions = defaultdict(Counter)

    for i in range(1, len(df)):
        newer_nums = df.loc[i - 1, "numbers"]
        older_nums = df.loc[i, "numbers"]

        for a in newer_nums:
            for b in older_nums:
                transitions[a][b] += 1

    rows = []

    for a in range(1, 50):
        total = sum(transitions[a].values())

        if total == 0:
            continue

        for b, count in transitions[a].items():
            rows.append({
                "z_nowszej": a,
                "do_starszej": b,
                "liczba_przejść": count,
                "prawdopodobieństwo": round(count / total * 100, 2)
            })

    return pd.DataFrame(rows)


def movement_history(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for i in range(1, len(df)):
        newer_set = set(df.loc[i - 1, "numbers"])
        older_set = set(df.loc[i, "numbers"])

        repeated = sorted(newer_set & older_set)
        entered_older = sorted(older_set - newer_set)
        disappeared = sorted(newer_set - older_set)

        newer_sum = sum(newer_set)
        older_sum = sum(older_set)

        dynamic = len(entered_older) + abs(older_sum - newer_sum) / 10

        rows.append({
            "nowsze_losowanie": int(df.loc[i - 1, "draw_id"]),
            "starsze_losowanie": int(df.loc[i, "draw_id"]),
            "powtórzone": " ".join(f"{x:02d}" for x in repeated),
            "weszły_w_starszym": " ".join(f"{x:02d}" for x in entered_older),
            "zniknęły_z_nowszego": " ".join(f"{x:02d}" for x in disappeared),
            "zmiana_sumy": older_sum - newer_sum,
            "dynamika": round(dynamic, 2)
        })

    return pd.DataFrame(rows)


def machine_memory_score(df: pd.DataFrame) -> pd.DataFrame:
    freq = number_frequency(df)
    gaps = gap_analysis(df)

    recent20 = df.head(20)
    recent50 = df.head(50)
    recent100 = df.head(100)

    c20 = Counter()
    c50 = Counter()
    c100 = Counter()

    for nums in recent20["numbers"]:
        c20.update(nums)

    for nums in recent50["numbers"]:
        c50.update(nums)

    for nums in recent100["numbers"]:
        c100.update(nums)

    rows = []

    for n in range(1, 50):
        total_hits = int(freq.loc[freq["liczba"] == n, "trafienia"].iloc[0])
        gap = int(gaps.loc[gaps["liczba"] == n, "przerwa_od_najnowszego"].iloc[0])

        hot_score = total_hits / len(df)
        momentum20 = c20[n] / max(1, len(recent20))
        momentum50 = c50[n] / max(1, len(recent50))
        momentum100 = c100[n] / max(1, len(recent100))
        cold_pressure = min(gap / 55, 1.8)

        score = (
            hot_score * 35
            + momentum20 * 30
            + momentum50 * 22
            + momentum100 * 12
            + cold_pressure * 12
        )

        rows.append({
            "liczba": n,
            "trafienia_całość": total_hits,
            "trafienia_20": c20[n],
            "trafienia_50": c50[n],
            "trafienia_100": c100[n],
            "przerwa_od_najnowszego": gap,
            "wynik_pamięci": round(score, 4)
        })

    return pd.DataFrame(rows).sort_values("wynik_pamięci", ascending=False)


def historical_shape_profile(df: pd.DataFrame) -> dict:
    sums = []
    spreads = []
    even_counts = []
    low_counts = []

    for nums in df["numbers"]:
        sums.append(sum(nums))
        spreads.append(max(nums) - min(nums))
        even_counts.append(sum(1 for n in nums if n % 2 == 0))
        low_counts.append(sum(1 for n in nums if n <= 24))

    return {
        "suma_min_10pct": int(np.percentile(sums, 10)),
        "suma_max_90pct": int(np.percentile(sums, 90)),
        "rozrzut_min_10pct": int(np.percentile(spreads, 10)),
        "rozrzut_max_90pct": int(np.percentile(spreads, 90)),
        "średnia_suma": round(float(np.mean(sums)), 2),
        "średni_rozrzut": round(float(np.mean(spreads)), 2),
        "typowe_parzyste": Counter(even_counts).most_common(3),
        "typowe_niskie": Counter(low_counts).most_common(3)
    }


def recent_similarity_penalty(coupon: list[int], df: pd.DataFrame, recent_window: int = 40) -> float:
    coupon_set = set(coupon)
    penalty = 0.0

    for nums in df.head(recent_window)["numbers"]:
        common = len(coupon_set & set(nums))

        if common >= 5:
            penalty += 90
        elif common == 4:
            penalty += 40
        elif common == 3:
            penalty += 11

    return penalty


def duplicate_pattern_penalty(coupon: list[int], generated_memory: list[list[int]]) -> float:
    penalty = 0.0
    coupon_set = set(coupon)

    for previous in generated_memory:
        common = len(coupon_set & set(previous))

        if common >= 5:
            penalty += 90
        elif common == 4:
            penalty += 40
        elif common == 3:
            penalty += 12

    return penalty


def explain_coupon(coupon, source_mode, memory_df, pair_df, gap_df, freq_df, df) -> str:
    memory_map = dict(zip(memory_df["liczba"], memory_df["wynik_pamięci"]))
    gap_map = dict(zip(gap_df["liczba"], gap_df["przerwa_od_najnowszego"]))
    hit_map = dict(zip(freq_df["liczba"], freq_df["trafienia"]))

    strongest_memory = sorted(coupon, key=lambda x: memory_map.get(x, 0), reverse=True)[:2]
    coldest = sorted(coupon, key=lambda x: gap_map.get(x, 0), reverse=True)[:2]
    hottest = sorted(coupon, key=lambda x: hit_map.get(x, 0), reverse=True)[:2]

    pair_set = set()

    for _, row in pair_df.head(150).iterrows():
        a, b = row["para"].split("-")
        pair_set.add((int(a), int(b)))

    strong_pairs = []

    for a, b in itertools.combinations(coupon, 2):
        if (a, b) in pair_set:
            strong_pairs.append(f"{a:02d}-{b:02d}")

    latest = set(df.iloc[0]["numbers"])
    repeated_latest = sorted(set(coupon) & latest)

    explanation = []
    explanation.append(f"Taktyka: {source_mode}")
    explanation.append(f"Najsilniejsza pamięć maszyny: {' '.join(f'{x:02d}' for x in strongest_memory)}")
    explanation.append(f"Presja powrotu zimnych kul: {' '.join(f'{x:02d}' for x in coldest)}")
    explanation.append(f"Wsparcie liczb gorących: {' '.join(f'{x:02d}' for x in hottest)}")

    if strong_pairs:
        explanation.append(f"Mocne pary w układzie: {', '.join(strong_pairs[:5])}")
    else:
        explanation.append("Mocne pary: średnie lub celowo osłabione dla większej różnorodności.")

    if repeated_latest:
        explanation.append(f"Powtórzenia z ostatniego losowania: {' '.join(f'{x:02d}' for x in repeated_latest)}")
    else:
        explanation.append("Brak powtórzeń z ostatniego losowania — profil anty-kopiujący.")

    explanation.append(f"Suma zestawu: {sum(coupon)}")
    explanation.append(f"Rozrzut: {max(coupon) - min(coupon)}")
    explanation.append("Zestaw oceniony przez pamięć, pary, przejścia, odbicie zimnych kul, balans i filtr anty-powtórzeń.")

    return " | ".join(explanation)


def generate_coupon(memory_df, mode):
    rng = np.random.default_rng()

    if mode == "PAMIĘĆ MASZYNY":
        pool = memory_df.head(26).copy()
        weights = pool["wynik_pamięci"].to_numpy()
        weights = weights / weights.sum()
        chosen = rng.choice(pool["liczba"].to_numpy(), size=6, replace=False, p=weights)

    elif mode == "GORĄCE LICZBY":
        pool = memory_df.sort_values("trafienia_całość", ascending=False).head(24)
        chosen = rng.choice(pool["liczba"].to_numpy(), size=6, replace=False)

    elif mode == "ZIMNE LICZBY":
        pool = memory_df.sort_values("przerwa_od_najnowszego", ascending=False).head(26)
        chosen = rng.choice(pool["liczba"].to_numpy(), size=6, replace=False)

    elif mode == "ODBICIE":
        pool = memory_df[
            (memory_df["przerwa_od_najnowszego"] >= memory_df["przerwa_od_najnowszego"].quantile(0.60))
            & (memory_df["wynik_pamięci"] >= memory_df["wynik_pamięci"].quantile(0.35))
        ]

        if len(pool) < 10:
            pool = memory_df.sort_values("przerwa_od_najnowszego", ascending=False).head(26)

        chosen = rng.choice(pool["liczba"].to_numpy(), size=6, replace=False)

    elif mode == "KONTROLOWANY CHAOS":
        core = memory_df.head(20)["liczba"].tolist()
        outsider = memory_df.tail(20)["liczba"].tolist()

        chosen_core = rng.choice(core, size=4, replace=False).tolist()
        chosen_outside = rng.choice(outsider, size=2, replace=False).tolist()
        chosen = chosen_core + chosen_outside

    else:
        hot = memory_df.sort_values("trafienia_całość", ascending=False).head(20)["liczba"].tolist()
        cold = memory_df.sort_values("przerwa_od_najnowszego", ascending=False).head(20)["liczba"].tolist()
        memory = memory_df.head(20)["liczba"].tolist()

        pool = list(dict.fromkeys(hot + cold + memory))
        chosen = rng.choice(pool, size=6, replace=False)

    return sorted(int(x) for x in chosen)


def coupon_score(coupon, pair_df, memory_df, transition_df, move_df, df, shape_profile, generated_memory=None):
    if generated_memory is None:
        generated_memory = []

    pair_map = {}

    for _, row in pair_df.iterrows():
        a, b = row["para"].split("-")
        pair_map[(int(a), int(b))] = row["wystąpienia"]

    memory_map = dict(zip(memory_df["liczba"], memory_df["wynik_pamięci"]))

    transition_score = 0
    top_transitions = transition_df.sort_values("liczba_przejść", ascending=False).head(500)

    for _, row in top_transitions.iterrows():
        if int(row["z_nowszej"]) in coupon and int(row["do_starszej"]) in coupon:
            transition_score += float(row["liczba_przejść"])

    pair_points = 0

    for a, b in itertools.combinations(coupon, 2):
        pair_points += pair_map.get((a, b), 0)

    memory_points = sum(memory_map.get(n, 0) for n in coupon)
    recent_dynamic = move_df.head(40)["dynamika"].mean()
    dynamic_bonus = min(recent_dynamic, 25)

    total_sum = sum(coupon)
    spread = max(coupon) - min(coupon)
    even_count = sum(1 for n in coupon if n % 2 == 0)
    low_count = sum(1 for n in coupon if n <= 24)

    balance_bonus = 0

    if even_count in [2, 3, 4]:
        balance_bonus += 16

    if low_count in [2, 3, 4]:
        balance_bonus += 16

    if shape_profile["suma_min_10pct"] <= total_sum <= shape_profile["suma_max_90pct"]:
        balance_bonus += 18

    if shape_profile["rozrzut_min_10pct"] <= spread <= shape_profile["rozrzut_max_90pct"]:
        balance_bonus += 18

    similarity_penalty = recent_similarity_penalty(coupon, df, 40)
    duplicate_penalty = duplicate_pattern_penalty(coupon, generated_memory)

    final_score = (
        memory_points * 9.5
        + pair_points * 0.86
        + transition_score * 0.40
        + dynamic_bonus
        + balance_bonus
        - similarity_penalty
        - duplicate_penalty
    )

    return round(final_score, 2)


def generate_reasoning_master_engine(
    memory_df,
    pair_df,
    transition_df,
    move_df,
    gap_df,
    freq_df,
    df,
    attempts,
    results_count
):
    tactics = [
        "PAMIĘĆ MASZYNY",
        "GORĄCE LICZBY",
        "ZIMNE LICZBY",
        "HYBRYDA",
        "ODBICIE",
        "KONTROLOWANY CHAOS"
    ]

    shape_profile = historical_shape_profile(df)
    all_candidates = {}
    generated_memory = []
    attempts_per_tactic = max(1, attempts // len(tactics))

    progress = st.progress(0)
    total_steps = attempts_per_tactic * len(tactics)
    current_step = 0

    for tactic in tactics:
        for _ in range(attempts_per_tactic):
            coupon = generate_coupon(memory_df, tactic)
            key = tuple(coupon)

            score = coupon_score(
                coupon=coupon,
                pair_df=pair_df,
                memory_df=memory_df,
                transition_df=transition_df,
                move_df=move_df,
                df=df,
                shape_profile=shape_profile,
                generated_memory=generated_memory
            )

            if key not in all_candidates or score > all_candidates[key]["score"]:
                all_candidates[key] = {
                    "numbers": coupon,
                    "score": score,
                    "source_mode": tactic,
                    "reasoning": explain_coupon(
                        coupon=coupon,
                        source_mode=tactic,
                        memory_df=memory_df,
                        pair_df=pair_df,
                        gap_df=gap_df,
                        freq_df=freq_df,
                        df=df
                    )
                }

            current_step += 1

            if current_step % 250 == 0:
                progress.progress(min(current_step / total_steps, 1.0))

    progress.progress(1.0)

    ranked = sorted(all_candidates.values(), key=lambda x: x["score"], reverse=True)
    final_results = []

    for candidate in ranked:
        if len(final_results) >= results_count:
            break

        candidate_set = set(candidate["numbers"])
        too_similar = False

        for accepted in final_results:
            if len(candidate_set & set(accepted["numbers"])) >= 4:
                too_similar = True
                break

        if not too_similar:
            final_results.append(candidate)
            generated_memory.append(candidate["numbers"])

    if len(final_results) < results_count:
        for candidate in ranked:
            if candidate not in final_results:
                final_results.append(candidate)

            if len(final_results) >= results_count:
                break

    return final_results


def generate_ultimate_next_draw(memory_df, pair_df, transition_df, move_df, df, attempts, results_count):
    shape_profile = historical_shape_profile(df)
    modes = ["PAMIĘĆ MASZYNY", "GORĄCE LICZBY", "ZIMNE LICZBY", "HYBRYDA"]
    candidates = {}
    attempts_per_mode = max(1, attempts // len(modes))

    for mode in modes:
        for _ in range(attempts_per_mode):
            coupon = generate_coupon(memory_df, mode)
            key = tuple(coupon)

            score = coupon_score(
                coupon=coupon,
                pair_df=pair_df,
                memory_df=memory_df,
                transition_df=transition_df,
                move_df=move_df,
                df=df,
                shape_profile=shape_profile,
                generated_memory=[]
            )

            if key not in candidates or score > candidates[key]["score"]:
                candidates[key] = {
                    "numbers": coupon,
                    "score": score,
                    "source_mode": mode,
                    "reasoning": "Tryb Ultimate łączy pamięć maszyny, gorące liczby, zimne liczby i hybrydę."
                }

    ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:results_count]


def create_report(df, freq_df, gap_df, pair_df, move_df, generated):
    lines = []

    lines.append("LOTTO MACHINE MEMORY ENGINE PRO")
    lines.append("=" * 60)
    lines.append(f"Data raportu: {datetime.now()}")
    lines.append(f"Liczba odczytanych losowań: {len(df)}")
    lines.append("Kolejność historii: od najnowszego do najstarszego")
    lines.append("")

    latest = df.iloc[0]
    oldest = df.iloc[-1]

    latest_result = " ".join(f"{x:02d}" for x in latest["numbers"])
    oldest_result = " ".join(f"{x:02d}" for x in oldest["numbers"])

    lines.append(f"Najnowsze losowanie: {latest['draw_id']} -> {latest_result}")
    lines.append(f"Najstarsze losowanie: {oldest['draw_id']} -> {oldest_result}")
    lines.append("")

    lines.append("TOP GORĄCE LICZBY")
    for _, r in freq_df.head(15).iterrows():
        lines.append(f"{int(r['liczba']):02d} | trafienia={r['trafienia']} | {r['procent']}%")

    lines.append("")
    lines.append("TOP ZIMNE LICZBY")
    for _, r in gap_df.head(15).iterrows():
        lines.append(f"{int(r['liczba']):02d} | przerwa={r['przerwa_od_najnowszego']}")

    lines.append("")
    lines.append("TOP PARY")
    for _, r in pair_df.head(20).iterrows():
        lines.append(f"{r['para']} | wystąpienia={r['wystąpienia']}")

    lines.append("")
    lines.append("WYGENEROWANE WYNIKI")
    for i, item in enumerate(generated, 1):
        nums = " ".join(f"{x:02d}" for x in item["numbers"])
        source_mode = item.get("source_mode", "NORMAL")
        reasoning = item.get("reasoning", "")
        lines.append(f"{i}. {nums} | MOC={item['score']} | TRYB={source_mode}")
        lines.append(f"   ROZUMOWANIE: {reasoning}")

    lines.append("")
    lines.append("OSTATNIA DYNAMIKA RUCHU")
    for _, r in move_df.head(20).iterrows():
        lines.append(
            f"{r['nowsze_losowanie']} -> {r['starsze_losowanie']} | "
            f"Powtórzone: {r['powtórzone']} | "
            f"Weszły: {r['weszły_w_starszym']} | "
            f"Dynamika={r['dynamika']}"
        )

    return "\n".join(lines)


st.markdown('<div class="big-title">🎰 Lotto Machine Memory Engine PRO</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="card">
    <b>Najmocniejsza wersja aplikacji pod komputer użytkownika:</b><br><br>
    ✔ AMD Ryzen 5 7535HS — dobry procesor do większej liczby prób<br>
    ✔ 16 GB RAM — bezpiecznie można używać 10 000–30 000 prób<br>
    ✔ RTX 2050 — aplikacja działa głównie na CPU, ale komputer ma zapas mocy<br>
    ✔ Historia traktowana od najnowszego do najstarszego<br>
    ✔ Tryb główny: <b>Silnik Rozumowania Maszyny PRO</b><br>
    ✔ Aplikacja nie tylko losuje, ale ocenia taktyki, podobieństwo, pary, ruch, dynamikę i pamięć maszyny
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("📘 Opis funkcji aplikacji po polsku"):
    st.markdown(
        """
        **Czytnik PDF** — odczytuje tabelę Lotto z pliku PDF po współrzędnych, dzięki czemu wynik jest poprawnie przypisany do numeru losowania.

        **Historia od najnowszego do najstarszego** — najnowsze losowanie jest zawsze na górze i od niego liczona jest pamięć, przerwy oraz dynamika.

        **Gorące liczby** — liczby najczęściej występujące w całej historii.

        **Zimne liczby** — liczby z największą przerwą od najnowszego losowania.

        **Siła par** — analiza par liczb, które historycznie często pojawiały się razem.

        **Przejścia** — badanie relacji między sąsiednimi losowaniami.

        **Historia ruchu** — pokazuje, które liczby się powtórzyły, które weszły i które zniknęły między układami.

        **Pamięć maszyny** — łączy pełną historię, ostatnie 20, 50 i 100 losowań oraz presję zimnych liczb.

        **Profil kształtu historycznego** — sprawdza typową sumę, rozrzut, liczbę parzystych i proporcję niskich/wysokich liczb.

        **Silnik Rozumowania Maszyny PRO** — najmocniejszy tryb. Generuje wiele kandydatów z różnych taktyk, karze układy zbyt podobne do ostatnich losowań, blokuje powtarzanie ciągle tych samych zestawów i wybiera najmocniejsze konfiguracje.

        **Ultimate Next Draw** — mocny tryb łączący pamięć, gorące, zimne i hybrydowe podejście.

        **Liczba prób** — im większa, tym więcej układów aplikacja analizuje przed wyborem wyniku.

        **Liczba wyników** — ile najlepszych zestawów aplikacja ma pokazać.
        """
    )

uploaded = st.file_uploader("Wgraj plik PDF Lotto 6/49", type=["pdf"])

if uploaded is None:
    st.info("Wgraj plik PDF, aby rozpocząć analizę.")
    st.stop()

try:
    pdf_bytes = uploaded.read()
    df = extract_draws_from_pdf(pdf_bytes)
except Exception as e:
    st.error(f"BŁĄD: {e}")
    st.stop()

freq_df = number_frequency(df)
gap_df = gap_analysis(df)
pair_df = pair_strength(df)
transition_df = transition_matrix(df)
move_df = movement_history(df)
memory_df = machine_memory_score(df)
shape_profile = historical_shape_profile(df)

latest = df.iloc[0]
oldest = df.iloc[-1]

latest_result = " ".join(f"{x:02d}" for x in latest["numbers"])
oldest_result = " ".join(f"{x:02d}" for x in oldest["numbers"])

m1, m2, m3, m4 = st.columns(4)

m1.metric("Losowań", len(df))
m2.metric("Najnowsze", int(latest["draw_id"]))
m3.metric("Najstarsze", int(oldest["draw_id"]))
m4.metric("Najnowszy wynik", latest_result)

st.markdown("## 🧠 Pamięć maszyny")
st.dataframe(memory_df.head(25), use_container_width=True, hide_index=True)

st.markdown("## 📐 Profil kształtu historycznego")
st.dataframe(pd.DataFrame([shape_profile]), use_container_width=True, hide_index=True)

st.markdown("## 🔥 Gorące liczby")
st.dataframe(freq_df.head(25), use_container_width=True, hide_index=True)

st.markdown("## ❄️ Zimne liczby")
st.dataframe(gap_df.head(25), use_container_width=True, hide_index=True)

st.markdown("## 🔗 Najsilniejsze pary")
st.dataframe(pair_df.head(50), use_container_width=True, hide_index=True)

st.markdown("## 🔄 Historia ruchu — od najnowszego do najstarszego")
st.dataframe(move_df.head(80), use_container_width=True, hide_index=True)

st.markdown("## ➡️ Przejścia między losowaniami")
st.dataframe(
    transition_df.sort_values("liczba_przejść", ascending=False).head(150),
    use_container_width=True,
    hide_index=True
)

st.markdown("## 🎯 Generator najbardziej prawdopodobnego wyniku")

c1, c2, c3 = st.columns(3)

with c1:
    mode = st.selectbox(
        "Tryb generowania",
        [
            "SILNIK ROZUMOWANIA MASZYNY PRO",
            "ULTIMATE NEXT DRAW",
            "PAMIĘĆ MASZYNY",
            "GORĄCE LICZBY",
            "ZIMNE LICZBY",
            "HYBRYDA",
            "ODBICIE",
            "KONTROLOWANY CHAOS"
        ]
    )

with c2:
    attempts = st.slider(
        "Liczba prób",
        500,
        60000,
        15000,
        step=500
    )

with c3:
    results_count = st.slider(
        "Liczba wyników",
        1,
        20,
        3
    )

st.markdown(
    """
    <div class="card">
    <b>Rekomendacja dla Twojego laptopa:</b><br>
    Najlepszy balans: <b>15 000 prób</b> i <b>3 wyniki</b>.<br>
    Tryb maksymalny: <b>30 000–60 000 prób</b>, ale może liczyć dłużej.
    </div>
    """,
    unsafe_allow_html=True
)

generated_results = []

if st.button("🎰 GENERUJ NAJMOCNIEJSZY WYNIK"):
    if mode == "SILNIK ROZUMOWANIA MASZYNY PRO":
        generated_results = generate_reasoning_master_engine(
            memory_df=memory_df,
            pair_df=pair_df,
            transition_df=transition_df,
            move_df=move_df,
            gap_df=gap_df,
            freq_df=freq_df,
            df=df,
            attempts=attempts,
            results_count=results_count
        )

    elif mode == "ULTIMATE NEXT DRAW":
        generated_results = generate_ultimate_next_draw(
            memory_df=memory_df,
            pair_df=pair_df,
            transition_df=transition_df,
            move_df=move_df,
            df=df,
            attempts=attempts,
            results_count=results_count
        )

    else:
        generated_memory = []

        for _ in range(results_count):
            best_coupon = None
            best_score = -999999

            for _ in range(attempts):
                coupon = generate_coupon(memory_df, mode)
                score = coupon_score(
                    coupon=coupon,
                    pair_df=pair_df,
                    memory_df=memory_df,
                    transition_df=transition_df,
                    move_df=move_df,
                    df=df,
                    shape_profile=shape_profile,
                    generated_memory=generated_memory
                )

                if score > best_score:
                    best_score = score
                    best_coupon = coupon

            generated_memory.append(best_coupon)

            generated_results.append({
                "numbers": best_coupon,
                "score": best_score,
                "source_mode": mode,
                "reasoning": explain_coupon(
                    coupon=best_coupon,
                    source_mode=mode,
                    memory_df=memory_df,
                    pair_df=pair_df,
                    gap_df=gap_df,
                    freq_df=freq_df,
                    df=df
                )
            })

    st.session_state["generated_results"] = generated_results

if "generated_results" in st.session_state:
    generated_results = st.session_state["generated_results"]

    rows = []

    for i, item in enumerate(generated_results, 1):
        rows.append({
            "wynik": i,
            "liczby": " ".join(f"{x:02d}" for x in item["numbers"]),
            "moc": item["score"],
            "tryb": item.get("source_mode", "NORMAL"),
            "rozumowanie": item.get("reasoning", "")
        })

    result_df = pd.DataFrame(rows)

    st.markdown(
        """
        <div class="card">
        <span class="green">
        MASZYNA ZNALAZŁA NAJMOCNIEJSZĄ KONFIGURACJĘ WEDŁUG ROZUMOWANIA PRO
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.markdown("## 🧠 Wyjaśnienie rozumowania")

    for i, item in enumerate(generated_results, 1):
        nums = " ".join(f"{x:02d}" for x in item["numbers"])

        st.markdown(
            f"""
            <div class="card">
            <b>Wynik {i}: {nums}</b><br>
            <b>Moc:</b> {item["score"]}<br>
            <b>Taktyka źródłowa:</b> {item.get("source_mode", "NORMAL")}<br><br>
            {item.get("reasoning", "")}
            </div>
            """,
            unsafe_allow_html=True
        )

    report = create_report(
        df=df,
        freq_df=freq_df,
        gap_df=gap_df,
        pair_df=pair_df,
        move_df=move_df,
        generated=generated_results
    )

    st.download_button(
        label="💾 Pobierz raport TXT",
        data=report.encode("utf-8"),
        file_name="lotto_machine_reasoning_pro_report.txt",
        mime="text/plain"
    )

st.markdown("## 📜 Pełna historia losowań — od najnowszego do najstarszego")

history_df = df.copy()
history_df["wynik"] = history_df["numbers"].apply(lambda x: " ".join(f"{n:02d}" for n in x))

st.dataframe(
    history_df[["draw_id", "wynik", "page"]],
    use_container_width=True,
    hide_index=True
)
