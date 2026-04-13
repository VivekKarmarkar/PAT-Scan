"""
Salary Estimate for the PAT-Scan Codebase Developer

Based on the skill profile required (PyTorch ML + FEM + differentiable physics + inverse problems)
and the cost_estimate.md analysis showing 2,411 development hours at $100-$175/hr contractor rates.

This script produces a salary estimate and visualizations.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. SKILLS PROFILE ANALYSIS
# ============================================================

skills = {
    "PyTorch / Deep Learning": "Advanced",
    "GPU/CUDA Programming": "Intermediate-Advanced",
    "Finite Element Methods (FEM)": "Advanced",
    "Differentiable Programming": "Advanced",
    "Computational Mechanics": "Advanced",
    "Inverse Problems / PINNs": "Specialized",
    "Scientific Python (NumPy, SciPy)": "Advanced",
    "LaTeX / Technical Writing": "Intermediate",
}

# ============================================================
# 2. ROLE MAPPING & SALARY DATA
# ============================================================

# The developer profile maps to these roles:
# - Machine Learning Engineer (with scientific computing focus)
# - Computational Scientist / Research Engineer
# - Applied Scientist (ML + Physics)

# Salary data sources: Levels.fyi, Glassdoor, BLS, industry reports (2025-2026)
# All figures in USD, annual, total compensation (base + bonus + equity where applicable)

salary_data = {
    "Role Title": "ML Research Engineer / Computational Scientist",
    "Required Experience": "3-7 years (given codebase complexity and domain depth)",

    "By Experience Level": {
        "Mid-Level (3-5 yr)":   {"low": 120_000, "median": 155_000, "high": 195_000},
        "Senior (5-8 yr)":      {"low": 155_000, "median": 195_000, "high": 260_000},
        "Staff/Principal (8+)": {"low": 200_000, "median": 260_000, "high": 350_000},
    },

    "By Company Type": {
        "Academia / National Lab":       {"low":  85_000, "median": 115_000, "high": 145_000},
        "Startup (seed-Series B)":       {"low": 120_000, "median": 160_000, "high": 210_000},
        "Mid-size Tech / Defense":       {"low": 145_000, "median": 185_000, "high": 240_000},
        "Big Tech (FAANG/equiv)":        {"low": 190_000, "median": 260_000, "high": 380_000},
        "Quant Finance / HFT":           {"low": 250_000, "median": 350_000, "high": 500_000},
    },

    "By Location": {
        "Remote / LCOL":          {"low": 110_000, "median": 145_000, "high": 190_000},
        "Mid-tier city (Austin, Denver)": {"low": 135_000, "median": 170_000, "high": 220_000},
        "SF Bay Area / NYC":      {"low": 170_000, "median": 220_000, "high": 310_000},
    },

    "Contractor-to-Salary Conversion": {
        "note": "cost_estimate.md uses $100-175/hr contractor rates",
        "contractor_low_hr": 100,
        "contractor_avg_hr": 135,
        "contractor_high_hr": 175,
        "billable_hours_per_year": 1_880,
        "salary_equivalent_factor": 0.60,  # salary ~ 60% of contractor gross (benefits, overhead, bench time)
        "implied_salary_low":  int(100 * 1880 * 0.60),   # $112,800
        "implied_salary_avg":  int(135 * 1880 * 0.60),    # $152,280
        "implied_salary_high": int(175 * 1880 * 0.60),    # $201,600
    },
}

# Best single-number estimate
best_estimate = {
    "most_likely_range": "$150,000 - $210,000",
    "point_estimate": "$180,000",
    "context": (
        "Senior ML Research Engineer or Computational Scientist role, "
        "5-7 years experience, mid-to-large tech or defense company, "
        "US-based (mixed remote/hybrid). This reflects the rare intersection of "
        "ML engineering + computational mechanics + differentiable physics."
    ),
}

# ============================================================
# 3. VISUALIZATIONS
# ============================================================

def fig_salary_by_company_type():
    """Bar chart: salary ranges by company type."""
    data = salary_data["By Company Type"]
    categories = list(data.keys())
    lows = [data[c]["low"] for c in categories]
    medians = [data[c]["median"] for c in categories]
    highs = [data[c]["high"] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.5

    # Draw range bars
    for i in range(len(categories)):
        ax.barh(i, highs[i] - lows[i], left=lows[i], height=0.4,
                color='#4A90D9', alpha=0.3, edgecolor='#4A90D9')
        ax.plot(medians[i], i, 'D', color='#C0392B', markersize=10, zorder=5)

    ax.set_yticks(x)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel("Annual Total Compensation (USD)", fontsize=12)
    ax.set_title("Estimated Salary for PAT-Scan Developer\nby Company Type", fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))

    # Add best estimate band
    ax.axvspan(150_000, 210_000, alpha=0.12, color='green', label='Most likely range ($150K-$210K)')
    ax.legend(loc='lower right', fontsize=10)

    ax.set_xlim(50_000, 520_000)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "salary_by_company_type.png")
    fig.savefig(path, dpi=150)
    plt.close()
    return path

def fig_salary_by_experience():
    """Horizontal bar chart: salary by experience level."""
    data = salary_data["By Experience Level"]
    categories = list(data.keys())
    lows = [data[c]["low"] for c in categories]
    medians = [data[c]["median"] for c in categories]
    highs = [data[c]["high"] for c in categories]

    fig, ax = plt.subplots(figsize=(9, 4))

    for i in range(len(categories)):
        ax.barh(i, highs[i] - lows[i], left=lows[i], height=0.4,
                color='#2ECC71', alpha=0.35, edgecolor='#27AE60')
        ax.plot(medians[i], i, 'D', color='#8E44AD', markersize=10, zorder=5)
        ax.text(medians[i], i + 0.25, f"${medians[i]/1000:.0f}K",
                ha='center', fontsize=10, color='#8E44AD', fontweight='bold')

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel("Annual Total Compensation (USD)", fontsize=12)
    ax.set_title("Salary by Experience Level\n(ML Research Engineer / Computational Scientist)", fontsize=13, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
    ax.set_xlim(80_000, 380_000)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "salary_by_experience.png")
    fig.savefig(path, dpi=150)
    plt.close()
    return path

def fig_skills_radar():
    """Radar chart of the required skills profile."""
    labels = list(skills.keys())
    level_map = {"Intermediate": 2, "Intermediate-Advanced": 3, "Advanced": 4, "Specialized": 5}
    values = [level_map[skills[s]] for s in labels]
    values += values[:1]  # close the polygon

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#3498DB', alpha=0.25)
    ax.plot(angles, values, 'o-', color='#2980B9', linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["Basic", "Intermediate", "Int-Adv", "Advanced", "Specialized"], fontsize=8)
    ax.set_title("Required Skills Profile\nfor PAT-Scan Developer", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "skills_radar.png")
    fig.savefig(path, dpi=150)
    plt.close()
    return path

def fig_contractor_vs_salary():
    """Comparison: contractor rates from cost_estimate.md vs equivalent salary."""
    conv = salary_data["Contractor-to-Salary Conversion"]
    labels = ["Low ($100/hr)", "Average ($135/hr)", "High ($175/hr)"]
    contractor_annual = [
        conv["contractor_low_hr"] * conv["billable_hours_per_year"],
        conv["contractor_avg_hr"] * conv["billable_hours_per_year"],
        conv["contractor_high_hr"] * conv["billable_hours_per_year"],
    ]
    salary_equiv = [
        conv["implied_salary_low"],
        conv["implied_salary_avg"],
        conv["implied_salary_high"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, [c/1000 for c in contractor_annual], width,
                   label='Contractor Gross (annual)', color='#E74C3C', alpha=0.7)
    bars2 = ax.bar(x + width/2, [s/1000 for s in salary_equiv], width,
                   label='Equivalent Salary (x0.60)', color='#3498DB', alpha=0.7)

    ax.set_ylabel("Annual Compensation ($K)", fontsize=12)
    ax.set_title("Contractor Rate to Salary Conversion\n(from cost_estimate.md rates)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"${bar.get_height():.0f}K", ha='center', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"${bar.get_height():.0f}K", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "contractor_vs_salary.png")
    fig.savefig(path, dpi=150)
    plt.close()
    return path

# ============================================================
# 4. GENERATE ALL OUTPUTS
# ============================================================

if __name__ == "__main__":
    print("Generating salary estimate visualizations...")

    p1 = fig_salary_by_company_type()
    print(f"  -> {p1}")

    p2 = fig_salary_by_experience()
    print(f"  -> {p2}")

    p3 = fig_skills_radar()
    print(f"  -> {p3}")

    p4 = fig_contractor_vs_salary()
    print(f"  -> {p4}")

    # Write summary JSON
    summary = {
        "question": "How much would someone who built this codebase get paid yearly?",
        "best_estimate": best_estimate,
        "salary_data": salary_data,
        "skills_required": skills,
        "methodology": (
            "1. Analyzed codebase complexity: 29K lines Python, 44 files, spanning ML/AI, FEM, "
            "differentiable physics, and inverse problems. "
            "2. Mapped required skill profile to industry roles (ML Research Engineer, "
            "Computational Scientist, Applied Scientist). "
            "3. Cross-referenced contractor rates from cost_estimate.md ($100-175/hr) with "
            "standard 0.60x salary conversion factor. "
            "4. Compared against 2025-2026 market data for similar roles by company type, "
            "experience level, and location."
        ),
    }

    json_path = os.path.join(OUTPUT_DIR, "salary_estimate.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {json_path}")

    print("\n" + "="*60)
    print("SALARY ESTIMATE SUMMARY")
    print("="*60)
    print(f"\nRole: {salary_data['Role Title']}")
    print(f"Experience: {salary_data['Required Experience']}")
    print(f"\nMost Likely Annual Salary: {best_estimate['most_likely_range']}")
    print(f"Point Estimate: {best_estimate['point_estimate']}")
    print(f"\nContext: {best_estimate['context']}")
    print(f"\nContractor-to-salary crosscheck:")
    conv = salary_data["Contractor-to-Salary Conversion"]
    print(f"  $100/hr contractor -> ~${conv['implied_salary_low']:,}/yr salary")
    print(f"  $135/hr contractor -> ~${conv['implied_salary_avg']:,}/yr salary")
    print(f"  $175/hr contractor -> ~${conv['implied_salary_high']:,}/yr salary")
    print(f"\nBy company type (median):")
    for ct, vals in salary_data["By Company Type"].items():
        print(f"  {ct}: ${vals['median']:,}")
    print("="*60)
