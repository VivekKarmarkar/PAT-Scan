"""
PAT-Scan Project — Estimated Salary Analysis
Derives annual salary equivalents from the project's development cost estimate.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── From cost_estimate.md ───
total_hours = 2411
unique_python_lines = 21237
total_python_lines = 29092

# Hourly contractor rates (2026 US market)
rates = {
    "Low-end": 100,
    "Average": 135,
    "High-end": 175,
}

# ─── Salary Conversion Assumptions ───
# A full-time developer works ~2,080 hours/year (40 hrs/week x 52 weeks)
# Billable utilization for contractors is typically 60-75%; for salaried employees
# the full 2,080 hours are "paid." We convert contractor rates to salary equivalents
# using a benefits/overhead multiplier of 1.25-1.40x (health insurance, 401k, PTO,
# payroll taxes, equipment).

FTE_HOURS_PER_YEAR = 2080
BENEFITS_MULTIPLIER_LOW = 1.25
BENEFITS_MULTIPLIER_HIGH = 1.40

# Duration estimate: how long would a single FTE take?
project_duration_years = total_hours / FTE_HOURS_PER_YEAR  # ~1.16 years

# ─── Salary Calculations ───
salary_data = {}
for tier, rate in rates.items():
    base_salary = rate * FTE_HOURS_PER_YEAR
    total_comp_low = base_salary * BENEFITS_MULTIPLIER_LOW
    total_comp_high = base_salary * BENEFITS_MULTIPLIER_HIGH
    project_cost = rate * total_hours
    salary_data[tier] = {
        "hourly_rate": rate,
        "base_salary": base_salary,
        "total_comp_low": total_comp_low,
        "total_comp_high": total_comp_high,
        "project_cost": project_cost,
    }

# Market salary benchmarks for comparison (2026 estimates, US)
market_benchmarks = {
    "ML Engineer (mid)": 145_000,
    "ML Engineer (senior)": 195_000,
    "Computational Scientist": 160_000,
    "Research Engineer (FEM+ML)": 185_000,
    "Staff ML Engineer (FAANG)": 300_000,
}

# ─── Build summary text ───
lines = []
lines.append("=" * 70)
lines.append("PAT-SCAN PROJECT -- ESTIMATED SALARY ANALYSIS")
lines.append("=" * 70)
lines.append("")
lines.append(f"Total development effort:  {total_hours:,} hours")
lines.append(f"FTE-equivalent duration:   {project_duration_years:.2f} years ({project_duration_years*12:.1f} months)")
lines.append(f"Unique Python lines:       {unique_python_lines:,}")
lines.append("")
lines.append("-" * 70)
lines.append("ANNUAL SALARY EQUIVALENTS (Base Salary = Hourly Rate x 2,080 hrs)")
lines.append("-" * 70)
lines.append(f"{'Tier':<15} {'Hourly':>10} {'Base Salary':>15} {'Total Comp':>22}")
lines.append(f"{'':15} {'Rate':>10} {'(annual)':>15} {'(w/ benefits)':>22}")
lines.append("-" * 70)
for tier, d in salary_data.items():
    lines.append(
        f"{tier:<15} ${d['hourly_rate']:>8,}/hr  ${d['base_salary']:>12,}  "
        f"${d['total_comp_low']:>10,.0f} - ${d['total_comp_high']:>10,.0f}"
    )
lines.append("")
lines.append("-" * 70)
lines.append("PROJECT COST (Engineering Hours x Rate)")
lines.append("-" * 70)
for tier, d in salary_data.items():
    lines.append(f"{tier:<15} {total_hours:,} hrs x ${d['hourly_rate']}/hr = ${d['project_cost']:>10,}")
lines.append("")
lines.append("-" * 70)
lines.append("MARKET SALARY BENCHMARKS (2026 US, for comparison)")
lines.append("-" * 70)
for role, sal in market_benchmarks.items():
    lines.append(f"  {role:<35} ${sal:>10,}")
lines.append("")
lines.append("-" * 70)
lines.append("INTERPRETATION")
lines.append("-" * 70)
lines.append("""
This project requires a rare combination of:
  - PyTorch / deep learning engineering (U-Net, GPU training)
  - Finite Element Methods (plane stress, triangular elements)
  - Differentiable programming (end-to-end gradients through FEM solver)
  - Inverse problem theory (PINN-based reconstruction)

The recommended salary range for a developer capable of building this
project from scratch is:

  BASE SALARY:    $208,000 - $364,000 / year
  TOTAL COMP:     $260,000 - $510,000 / year (including benefits/equity)

  RECOMMENDED MIDPOINT:  $281,000 base / $365,000 total comp

This places the role between a Senior ML Engineer and a Staff-level
Research Engineer at a technology company, reflecting the specialized
domain knowledge required.

Project cost if contracted out: $241,000 - $422,000
(equivalent to 6-12 months of a senior contractor's time)
""")

summary_text = "\n".join(lines)
print(summary_text)

with open(os.path.join(OUTPUT_DIR, "salary_estimate.txt"), "w") as f:
    f.write(summary_text)

# ─── Visualization 1: Salary Comparison Bar Chart ───
fig, ax = plt.subplots(figsize=(10, 6))

categories = list(salary_data.keys())
base_salaries = [salary_data[t]["base_salary"] for t in categories]
comp_low = [salary_data[t]["total_comp_low"] for t in categories]
comp_high = [salary_data[t]["total_comp_high"] for t in categories]

x = np.arange(len(categories))
width = 0.3

bars1 = ax.bar(x - width/2, [s/1000 for s in base_salaries], width,
               label="Base Salary", color="#2196F3", edgecolor="white")
bars2 = ax.bar(x + width/2, [s/1000 for s in comp_high], width,
               label="Total Comp (high)", color="#FF9800", edgecolor="white", alpha=0.85)

# Add market benchmark lines
benchmark_colors = ["#4CAF50", "#9C27B0", "#F44336", "#00BCD4", "#795548"]
for (role, sal), color in zip(market_benchmarks.items(), benchmark_colors):
    ax.axhline(y=sal/1000, color=color, linestyle="--", alpha=0.6, linewidth=1)
    ax.text(len(categories)-0.5, sal/1000 + 3, role, fontsize=7.5,
            color=color, ha="right", va="bottom")

ax.set_ylabel("Annual Compensation ($K)", fontsize=12)
ax.set_title("PAT-Scan Developer Salary Estimates vs. Market Benchmarks", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}K"))
ax.set_ylim(0, 520)
ax.grid(axis="y", alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"${bar.get_height():.0f}K", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"${bar.get_height():.0f}K", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "salary_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─── Visualization 2: Cost Breakdown Pie Chart ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Hours by domain
domain_hours = {
    "ML/AI (U-Net, training)": 645,
    "FEM / Comp. Mechanics": 275,
    "Data Generation": 139,
    "Utilities / Other": 24,
    "Prompts & Docs": 122,
}
overhead_hours = {
    "Architecture & Design": 181,
    "Debugging": 362,
    "Code Review": 60,
    "Documentation": 121,
    "Integration & Testing": 181,
    "Learning Curve": 301,
}

colors1 = ["#2196F3", "#FF5722", "#4CAF50", "#9E9E9E", "#FF9800"]
wedges1, texts1, autotexts1 = ax1.pie(
    domain_hours.values(), labels=domain_hours.keys(), autopct="%1.0f%%",
    colors=colors1, startangle=90, textprops={"fontsize": 9}
)
ax1.set_title(f"Base Development Hours\n({sum(domain_hours.values()):,} hrs)", fontsize=12, fontweight="bold")

colors2 = ["#3F51B5", "#E91E63", "#009688", "#FFC107", "#673AB7", "#F44336"]
wedges2, texts2, autotexts2 = ax2.pie(
    overhead_hours.values(), labels=overhead_hours.keys(), autopct="%1.0f%%",
    colors=colors2, startangle=90, textprops={"fontsize": 9}
)
ax2.set_title(f"Overhead Hours\n({sum(overhead_hours.values()):,} hrs)", fontsize=12, fontweight="bold")

plt.suptitle(f"Total: {total_hours:,} Development Hours", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hours_breakdown.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─── Visualization 3: Project Cost Waterfall ───
fig, ax = plt.subplots(figsize=(10, 5))

# Show how costs build up for the "Average" scenario
steps = ["ML/AI Code", "FEM Code", "Data Gen", "Utils/Docs", "Overhead (debug,\ndesign, testing)", "Total"]
costs = [645*135, 275*135, 139*135, 146*135, 1206*135, 0]
cumulative = np.cumsum(costs[:-1])
total_cost = cumulative[-1]
costs[-1] = total_cost

colors_wf = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#333333"]
bottoms = [0] + list(cumulative[:-1])

for i in range(len(steps)-1):
    ax.bar(steps[i], costs[i]/1000, bottom=bottoms[i]/1000, color=colors_wf[i],
           edgecolor="white", width=0.6)
    ax.text(i, (bottoms[i] + costs[i]/2)/1000, f"${costs[i]/1000:.0f}K",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

ax.bar(steps[-1], total_cost/1000, color=colors_wf[-1], edgecolor="white", width=0.6)
ax.text(len(steps)-1, total_cost/2000, f"${total_cost/1000:.0f}K",
        ha="center", va="center", fontsize=11, color="white", fontweight="bold")

ax.set_ylabel("Cost ($K)", fontsize=12)
ax.set_title("Project Cost Buildup (Average Rate: $135/hr)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}K"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cost_waterfall.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─── Save structured data as JSON ───
output_json = {
    "project": "PAT-Scan",
    "analysis_date": "2026-03-05",
    "total_development_hours": total_hours,
    "fte_equivalent_years": round(project_duration_years, 2),
    "salary_estimates": {
        tier: {
            "hourly_rate_usd": d["hourly_rate"],
            "annual_base_salary_usd": d["base_salary"],
            "annual_total_comp_range_usd": [round(d["total_comp_low"]), round(d["total_comp_high"])],
            "project_cost_usd": d["project_cost"],
        }
        for tier, d in salary_data.items()
    },
    "recommended": {
        "base_salary_range_usd": [208_000, 364_000],
        "total_comp_range_usd": [260_000, 510_000],
        "midpoint_base_usd": 281_000,
        "midpoint_total_comp_usd": 365_000,
    },
    "market_benchmarks_2026": market_benchmarks,
    "methodology": (
        "Hourly contractor rates converted to annual salary (x 2,080 hrs/yr). "
        "Total compensation includes 1.25-1.40x benefits multiplier for health insurance, "
        "retirement, PTO, payroll taxes, and equipment."
    ),
}

with open(os.path.join(OUTPUT_DIR, "salary_estimate.json"), "w") as f:
    json.dump(output_json, f, indent=2)

print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("  - salary_estimate.txt")
print("  - salary_estimate.json")
print("  - salary_comparison.png")
print("  - hours_breakdown.png")
print("  - cost_waterfall.png")
