import os
import sys
import json
import glob
import re
import time
from datetime import datetime
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") #headless plot rendering
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data");
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts");
CURRENT_YEAR = datetime.now().year;
SIMILARITY_THRESHOLD = 0.3;

"""
Years (Optional):
- One year (e.g., 2022): Analyze from that year to Present.
- Two years (e.g., 2022 2023): Analyze inclusive range.
- No years: Analyze from 2000 to Present.

Product Filter (Optional):
- Text string (e.g., CARROTS).
- If provided, filter the dataset to include only records where a Suspect Product matches this name (case-insensitive substring match).
"""
def parseArgs(argv):
    years = [];
    tokens = [];

    for arg in argv:
        if(re.match(r"^\d{4}$", arg) and 2000 <= int(arg) <= 2027):
            years.append(int(arg));
        else:
            tokens.append(arg);
    
    if(len(years) == 0):
        startYear, endYear = 2000, CURRENT_YEAR;
    elif(len(years) == 1):
        startYear, endYear = years[0], CURRENT_YEAR;
    else:
        startYear, endYear = min(years), max(years);
    
    # mash all other strings into one for flexibility
    productFilter = " ".join(tokens).strip() if tokens else None;

    return startYear, endYear, productFilter


"""
load all records from the data folder
"""
def loadAllRecords():
    records = [];
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")));

    if(not files):
        print("JSON files not found, run part1.js or check the data folder");
        sys.exit(1);
    
    print("Loading [" + str(len(files)) + "] json files...");

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f);
            if isinstance(data, list):
                records.extend(data);
            elif isinstance(data, dict) and "results" in data:
                records.extend(data["results"]);
            else:
                print("Bad structure in file: " + os.path.basename(filepath) + " skipping...");
        except(json.JSONDecodeError):
            print("file: " + os.path.basename(filepath) + " could not be read");
    
    print("Loaded [" + str(len(records)) + "] records from files");
    return records;

def getYear(string):
    if(not string or not isinstance(string, str)):
        return None;
    match = re.match(r"(\d{4})", string);
    if(match):
        return int(match.group(1));
    return None;

def filterRecords(records, start, end, productFilter):
    filtered = [];
    low = productFilter.lower() if productFilter else None;

    for rec in records:
        year = getYear(rec.get("date_started"));
        if(year is None):
            continue; # skip if the year is bad
        if(year < start or year > end):
            continue; # skip if the year is outside the filter
        
        if(low):
            products = rec.get("products", []);
            if(not products):
                continue;

            matched = False;
            for prod in products:
                role = prod.get("role", "").strip();
                name = prod.get("name_brand", "").strip();
                if(role.lower() == "suspect" and low in name.lower()):
                    matched = True;
                    break;
            if(not matched):
                continue;
        filtered.append(rec);
    return filtered;

def ageToYears(age_value, age_unit):
    if(age_value is None or age_unit is None):
        return None

    try:
        age_num = float(age_value)
    except (ValueError, TypeError):
        return None

    if(age_num <= 0):
        return None

    unit = (age_unit or "").strip().lower()

    if(unit.startswith("year")):
        age_years = age_num
    elif(unit.startswith("decade")):
        age_years = age_num * 10
    elif(unit.startswith("month")):
        age_years = age_num / 12.0
    elif(unit.startswith("week")):
        age_years = age_num / 52.0
    elif(unit.startswith("day")):
        age_years = age_num / 365.25
    else:
        return None

    if(age_years > 120):
        return None

    return round(age_years, 2)

def normalize_text(text):
    if not text:
        return ""
    t = text.upper().strip()
    # Remove common trademark / registration marks
    t = re.sub(r"[®™©]", "", t)
    # Replace non-alphanumeric with space (removes hyphens, parens, slashes)
    t = re.sub(r"[^A-Z0-9 ]", " ", t)
    # Collapse multiple whitespace to single space
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _word_set(text):
    """
    Returns the set of word tokens from a normalized string,
    excluding very short tokens (length 1) that are too ambiguous.

    Args:
        text (str): Normalized (uppercased) text.

    Returns:
        frozenset[str]: Unique word tokens.
    """
    return frozenset(w for w in text.split() if len(w) >= 2)


def _word_jaccard(set_a, set_b):
    """
    Computes Jaccard similarity between two word-token sets.
    Returns 0.0 if both sets are empty.

    Jaccard = |A ∩ B| / |A ∪ B|

    This is extremely fast — just set operations on a handful of strings.

    Args:
        set_a (frozenset[str]): First word set.
        set_b (frozenset[str]): Second word set.

    Returns:
        float: Jaccard similarity in [0, 1].
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def deduplicate_names(counter, threshold=SIMILARITY_THRESHOLD):
    """
    Merges similar names in a Counter using word-level Jaccard similarity.

    Algorithm:
      1. Sort names by frequency (most common first) — the most common
         variant becomes the canonical representative.
      2. For each name, compute its word-token set.
      3. Use a **word → canonical-index** bucket map so we only compare
         against canonical names sharing at least one word.  For each
         candidate, compute word-Jaccard similarity (a single set
         intersection + union — microseconds).
      4. If Jaccard >= threshold, merge into that canonical entry.
      5. Otherwise, register as a new canonical entry.

    Jaccard on small word sets is O(k) where k ~ 3-6 tokens, so each
    comparison costs nanoseconds.  The bucket map keeps the number of
    comparisons low.  Bucket sizes are capped so common words like
    "VITAMIN" don't cause blowup.

    Default threshold of 0.67 means two names must share ≥ 2/3 of their
    combined unique words.  Examples:
      - {"VITAMIN", "D3"} vs {"VITAMIN", "D"}  → 1/3 = 0.33  (not merged)
      - {"VITAMIN", "D3"} vs {"VITAMIN", "D3"} → 2/2 = 1.00  (merged)
      - {"GREEN", "TEA", "EXTRACT"} vs {"GREEN", "TEA", "EXTRACT", "CAPSULES"}
            → 3/4 = 0.75  (merged)

    Args:
        counter (Counter): {name: count}
        threshold (float): Word-Jaccard similarity above which names merge.

    Returns:
        Counter: Deduplicated counter with merged counts.
    """
    sorted_items = counter.most_common()

    canonical = []              # list of [canonical_name, count]
    canonical_word_sets = []    # parallel list of frozenset word tokens

    # word → set of canonical indices containing that word.
    # Cap bucket size so ultra-common words don't cause O(N) fan-out.
    MAX_BUCKET = 500
    word_buckets = defaultdict(set)

    for name, count in sorted_items:
        norm = normalize_text(name)
        if not norm:
            continue
        words = _word_set(norm)
        if not words:
            # Single-character name or empty after filtering — keep as-is
            canonical.append([name, count])
            canonical_word_sets.append(words)
            continue

        # Collect candidate canonical indices that share at least one word
        candidate_indices = set()
        for w in words:
            bucket = word_buckets.get(w)
            if bucket:
                candidate_indices.update(bucket)

        # Find best match among candidates
        best_idx = -1
        best_score = 0.0
        for i in candidate_indices:
            score = _word_jaccard(words, canonical_word_sets[i])
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold:
            # Merge into existing canonical entry
            canonical[best_idx][1] += count
        else:
            # New canonical entry
            new_idx = len(canonical)
            canonical.append([name, count])
            canonical_word_sets.append(words)
            for w in words:
                bucket = word_buckets[w]
                if len(bucket) < MAX_BUCKET:
                    bucket.add(new_idx)

    return Counter({name: cnt for name, cnt in canonical})

def analyze(records):
    outcomesRaw = Counter();
    reactionsRaw = Counter();
    productsRaw = Counter();
    agesAll = [];
    agesFemale = [];
    agesMale = [];
    casesByYear = Counter();

    for rec in records:
        year = getYear(rec.get("date_started"));
        if year is not None:
            casesByYear[year] += 1;
        
        for outcome in (rec.get("outcomes") or []):
            if outcome and isinstance(outcome, str) and outcome.strip():
                outcomesRaw[outcome.strip()] += 1;

        for reaction in (rec.get("reactions") or []):
            if reaction and isinstance(reaction, str) and reaction.strip():
                reactionsRaw[reaction.strip()] += 1;

        for prod in (rec.get("products") or []):
            role = (prod.get("role") or "").strip().lower();
            name = (prod.get("name_brand") or "").strip();
            if role == "suspect" and name:
                productsRaw[name] += 1;
        
        consumer = rec.get("consumer") or {}
        ageYears = ageToYears(consumer.get("age"), consumer.get("age_unit"))
        if ageYears is not None:
            agesAll.append(ageYears)
            gender = (consumer.get("gender") or "").strip().lower()
            if gender == "female":
                agesFemale.append(ageYears)
            elif gender == "male":
                agesMale.append(ageYears)

    print("Deduplicating outcomes")
    outcomesDeduped = deduplicate_names(outcomesRaw)
    #outcomesDeduped = outcomesRaw;
    print("Deduplicating reactions")
    reactionsDeduped = deduplicate_names(reactionsRaw)
    #reactionsDeduped = reactionsRaw;
    print("Deduplicating product names")
    productsDeduped = deduplicate_names(productsRaw)
    #productsDeduped = productsRaw;

    return {
        "total_records": len(records),
        "outcomes_counter": outcomesDeduped,
        "reactions_counter": reactionsDeduped,
        "products_counter": productsDeduped,
        "ages_all": agesAll,
        "ages_female": agesFemale,
        "ages_male": agesMale,
        "cases_by_year": dict(casesByYear),
    }

def print_top_n(title, counter, n=25):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for rank, (name, count) in enumerate(counter.most_common(n), start=1):
        print(f"  {rank:>3}. {name:<45s} {count:>7,}")
    print()


def print_age_stats(ages_all, ages_female, ages_male):
    print(f"\n{'='*60}")
    print(f"  Average Consumer Age")
    print(f"{'='*60}")

    def _fmt(label, ages_list):
        if ages_list:
            avg = np.mean(ages_list)
            med = np.median(ages_list)
            print(f"  {label:<15s}  Avg: {avg:6.1f} yrs  |  Median: {med:5.1f} yrs  |  N = {len(ages_list):,}")
        else:
            print(f"  {label:<15s}  No data available.")

    _fmt("Total", ages_all)
    _fmt("Female", ages_female)
    _fmt("Male", ages_male)
    print()


def print_report(stats):
    print(f"\n{'#'*60}")
    print(f"  FDA Adverse Food Event – Analysis Report")
    print(f"{'#'*60}")
    print(f"  Total Records: {stats['total_records']:,}")

    print_top_n("Top 25 Outcomes", stats["outcomes_counter"], 25)
    print_top_n("Top 25 Reactions", stats["reactions_counter"], 25)
    print_top_n("Top 25 Suspect Products", stats["products_counter"], 25)
    print_age_stats(stats["ages_all"], stats["ages_female"], stats["ages_male"])

def generateChart(stats):
    # Ensure charts directory exists
    os.makedirs(CHARTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = os.path.join(CHARTS_DIR, f"{timestamp}.png")

    # --- Prepare data ---

    # Cases by year (sorted)
    years_sorted = sorted(stats["cases_by_year"].keys())
    counts_sorted = [stats["cases_by_year"][y] for y in years_sorted]

    # Ages
    ages = stats["ages_all"]

    # --- Create figure with 2 subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("FDA Adverse Food Event Analysis", fontsize=16, fontweight="bold")

    # ----- Subplot 1: Bar chart of cases by year -----
    if years_sorted:
        bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(years_sorted)))
        ax1.bar([str(y) for y in years_sorted], counts_sorted, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax1.set_title("Total Cases by Year", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Year", fontsize=11)
        ax1.set_ylabel("Number of Cases", fontsize=11)
        ax1.tick_params(axis="x", rotation=45)
        # Add value labels on top of each bar
        for i, (yr, cnt) in enumerate(zip(years_sorted, counts_sorted)):
            ax1.text(i, cnt + max(counts_sorted) * 0.01, f"{cnt:,}",
                     ha="center", va="bottom", fontsize=7, rotation=45)
    else:
        ax1.text(0.5, 0.5, "No year data available", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title("Total Cases by Year", fontsize=13, fontweight="bold")

    # ----- Subplot 2: Age distribution histogram -----
    if ages:
        # Bin by every integer year from 0 to max age
        maxAge = int(np.ceil(max(ages))) + 1
        bins = np.arange(0, maxAge + 1, 1)  # 1-year bins
        ax2.hist(ages, bins=bins, color="steelblue", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax2.set_title("Distribution of Consumer Ages", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Age (years)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        # Add a vertical line for the mean
        meanAge = np.mean(ages)
        ax2.axvline(meanAge, color="red", linestyle="--", linewidth=1.5,
                     label=f"Mean: {meanAge:.1f} yrs")
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No age data available", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Distribution of Consumer Ages", fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)

    return chart_path

def main():
    args = sys.argv[1:]
    startYear, endYear, productFilter = parseArgs(args);

    print("start:", startYear, "end:", endYear, "product: \""+productFilter if productFilter else "none"+"\"");

    overall_start = time.time()

    allRecords = loadAllRecords();

    filtered = filterRecords(allRecords, startYear, endYear, productFilter);

    print("Filtered down to ["+str(len(filtered))+"] from [" + str(len(allRecords)) + "]");

    if(len(filtered) == 0):
        print("No records left after filtering, please change criteria.");

    print("Analyzing stats...");
    stats = analyze(filtered);
    
    print_report(stats);

    print("Generating chart stats...");
    generateChart(stats);

    elapsed = time.time() - overall_start
    print(f"\nDone in {elapsed:.1f}s.")
    

if __name__ == "__main__":
    main();