"""
Realistic Profit Calculator with Country-Specific Economic Factors
Considers: staff costs, import duties, tax rates, and operational expenses
"""

import pandas as pd
import numpy as np
from config import CATEGORY_MIDPOINTS, GROSS_MARGIN, MONTHS_PER_YEAR

# ============================================================================
# COUNTRY-SPECIFIC ECONOMIC DATA (2019 figures)
# ============================================================================

COUNTRY_DATA = {
    # Airport: Country, Avg Monthly Wage (EUR), Import Duty Rate, Corporate Tax Rate, Has FTA with EU
    "PEK": {  # Beijing, China
        "country": "China",
        "avg_monthly_wage_eur": 1200,
        "import_duty_rate": 0.10,  # 10% average duty on clothing
        "corporate_tax_rate": 0.25,
        "has_eu_fta": False,
        "currency": "CNY",
    },
    "EZE": {  # Buenos Aires, Argentina
        "country": "Argentina",
        "avg_monthly_wage_eur": 800,
        "import_duty_rate": 0.35,  # High import duties
        "corporate_tax_rate": 0.30,
        "has_eu_fta": True,  # EU-Mercosur agreement
        "currency": "ARS",
    },
    "DFW": {  # Dallas, USA
        "country": "USA",
        "avg_monthly_wage_eur": 3500,
        "import_duty_rate": 0.165,  # Average clothing duty
        "corporate_tax_rate": 0.21,  # Federal corporate tax
        "has_eu_fta": False,
        "currency": "USD",
    },
    "DXB": {  # Dubai, UAE
        "country": "UAE",
        "avg_monthly_wage_eur": 2800,
        "import_duty_rate": 0.05,  # Low duties in UAE
        "corporate_tax_rate": 0.09,  # Introduced in 2023, but use low estimate for 2019
        "has_eu_fta": False,
        "currency": "AED",
    },
    "HKG": {  # Hong Kong
        "country": "Hong Kong",
        "avg_monthly_wage_eur": 2200,
        "import_duty_rate": 0.00,  # Free port - no import duties
        "corporate_tax_rate": 0.165,
        "has_eu_fta": False,
        "currency": "HKD",
    },
    "KUL": {  # Kuala Lumpur, Malaysia
        "country": "Malaysia",
        "avg_monthly_wage_eur": 900,
        "import_duty_rate": 0.20,
        "corporate_tax_rate": 0.24,
        "has_eu_fta": True,  # EU-Malaysia FTA benefits
        "currency": "MYR",
    },
    "MEL": {  # Melbourne, Australia
        "country": "Australia",
        "avg_monthly_wage_eur": 3800,
        "import_duty_rate": 0.05,  # Low due to FTA
        "corporate_tax_rate": 0.30,
        "has_eu_fta": True,  # EU-Australia FTA
        "currency": "AUD",
    },
    "JFK": {  # New York, USA
        "country": "USA",
        "avg_monthly_wage_eur": 4200,  # Higher than Dallas
        "import_duty_rate": 0.165,
        "corporate_tax_rate": 0.21,
        "has_eu_fta": False,
        "currency": "USD",
    },
    "SFO": {  # San Francisco, USA
        "country": "USA",
        "avg_monthly_wage_eur": 5000,  # Highest in USA
        "import_duty_rate": 0.165,
        "corporate_tax_rate": 0.21,
        "has_eu_fta": False,
        "currency": "USD",
    },
    "PVG": {  # Shanghai, China
        "country": "China",
        "avg_monthly_wage_eur": 1400,  # Higher than Beijing
        "import_duty_rate": 0.10,
        "corporate_tax_rate": 0.25,
        "has_eu_fta": False,
        "currency": "CNY",
    },
    "SIN": {  # Singapore
        "country": "Singapore",
        "avg_monthly_wage_eur": 3200,
        "import_duty_rate": 0.00,  # Free trade port
        "corporate_tax_rate": 0.17,
        "has_eu_fta": True,  # EU-Singapore FTA
        "currency": "SGD",
    },
    "HND": {  # Tokyo, Japan
        "country": "Japan",
        "avg_monthly_wage_eur": 2800,
        "import_duty_rate": 0.125,
        "corporate_tax_rate": 0.308,  # Combined national + local
        "has_eu_fta": True,  # EU-Japan EPA
        "currency": "JPY",
    },
}


# ============================================================================
# STAFFING ASSUMPTIONS
# ============================================================================


def calculate_staff_costs(store_sqm, avg_monthly_wage_eur):
    """
    Calculate monthly staff costs based on store size and local wages

    Assumptions:
    - 1 FTE per 40 sqm of store space
    - Add 30% for benefits, training, uniforms
    """
    num_staff = max(2, store_sqm / 40)  # Minimum 2 staff
    base_salary = num_staff * avg_monthly_wage_eur
    total_staff_cost = base_salary * 1.30  # Add 30% for benefits

    return total_staff_cost


# ============================================================================
# COST OF GOODS SOLD (COGS) WITH IMPORT DUTIES
# ============================================================================


def calculate_cogs_with_duties(revenue, gross_margin, import_duty_rate, has_eu_fta):
    """
    Calculate COGS including import duties

    Logic:
    - Base COGS = Revenue * (1 - Gross Margin)
    - If no FTA: Add import duties on the landed cost
    - If FTA: Reduced or zero duties
    """
    base_cogs = revenue * (1 - gross_margin)

    if has_eu_fta:
        # FTA reduces duties significantly
        effective_duty_rate = import_duty_rate * 0.2  # 80% reduction
    else:
        effective_duty_rate = import_duty_rate

    # Import duties are calculated on landed cost (base COGS)
    import_duties = base_cogs * effective_duty_rate

    total_cogs = base_cogs + import_duties

    return {
        "base_cogs": base_cogs,
        "import_duties": import_duties,
        "total_cogs": total_cogs,
    }


# ============================================================================
# PROFIT CALCULATION WITH REALISTIC FACTORS
# ============================================================================


def calculate_realistic_profit(
    airport_code, annual_revenue, store_sqm, monthly_lease_cost
):
    """
    Calculate realistic annual profit considering all factors

    Returns detailed breakdown of costs and profit
    """

    # Get country-specific data
    if airport_code not in COUNTRY_DATA:
        print(f"Warning: No country data for {airport_code}, using defaults")
        country_info = {
            "avg_monthly_wage_eur": 2000,
            "import_duty_rate": 0.15,
            "corporate_tax_rate": 0.25,
            "has_eu_fta": False,
            "country": "Unknown",
        }
    else:
        country_info = COUNTRY_DATA[airport_code]

    # 1. REVENUE
    monthly_revenue = annual_revenue / MONTHS_PER_YEAR

    # 2. COST OF GOODS SOLD (including import duties)
    cogs_breakdown = calculate_cogs_with_duties(
        annual_revenue,
        GROSS_MARGIN,
        country_info["import_duty_rate"],
        country_info["has_eu_fta"],
    )

    # 3. GROSS PROFIT (after COGS and duties)
    annual_gross_profit = annual_revenue - cogs_breakdown["total_cogs"]

    # 4. OPERATING EXPENSES

    # 4a. Staff costs (monthly -> annual)
    monthly_staff_cost = calculate_staff_costs(
        store_sqm, country_info["avg_monthly_wage_eur"]
    )
    annual_staff_cost = monthly_staff_cost * MONTHS_PER_YEAR

    # 4b. Lease/Rent (monthly -> annual)
    annual_lease_cost = monthly_lease_cost * MONTHS_PER_YEAR

    # 4c. Other operating expenses (5% of revenue)
    # Utilities, marketing, supplies, shrinkage
    annual_other_opex = annual_revenue * 0.05

    # Total operating expenses
    total_opex = annual_staff_cost + annual_lease_cost + annual_other_opex

    # 5. EBIT (Earnings Before Interest and Tax)
    ebit = annual_gross_profit - total_opex

    # 6. CORPORATE TAX
    corporate_tax = max(0, ebit * country_info["corporate_tax_rate"])

    # 7. NET PROFIT (After-Tax)
    net_profit = ebit - corporate_tax

    # 8. PROFIT MARGIN
    profit_margin = (net_profit / annual_revenue * 100) if annual_revenue > 0 else 0

    return {
        # Revenue
        "annual_revenue": annual_revenue,
        "monthly_revenue": monthly_revenue,
        # COGS
        "base_cogs": cogs_breakdown["base_cogs"],
        "import_duties": cogs_breakdown["import_duties"],
        "total_cogs": cogs_breakdown["total_cogs"],
        "cogs_percent": (cogs_breakdown["total_cogs"] / annual_revenue * 100),
        # Gross Profit
        "gross_profit": annual_gross_profit,
        "gross_margin_percent": (annual_gross_profit / annual_revenue * 100),
        # Operating Expenses
        "staff_cost": annual_staff_cost,
        "lease_cost": annual_lease_cost,
        "other_opex": annual_other_opex,
        "total_opex": total_opex,
        # EBIT
        "ebit": ebit,
        "ebit_margin_percent": (
            (ebit / annual_revenue * 100) if annual_revenue > 0 else 0
        ),
        # Tax
        "corporate_tax": corporate_tax,
        "effective_tax_rate": country_info["corporate_tax_rate"] * 100,
        # Net Profit
        "net_profit": net_profit,
        "profit_margin_percent": profit_margin,
        # Country info
        "country": country_info["country"],
        "has_eu_fta": country_info["has_eu_fta"],
        "import_duty_rate_percent": country_info["import_duty_rate"] * 100,
        # Store info
        "store_sqm": store_sqm,
        "num_staff": max(2, store_sqm / 40),
    }


# ============================================================================
# BATCH CALCULATION FOR ALL AIRPORTS
# ============================================================================


def calculate_profits_for_all_airports(revenue_by_airport, lease_data):
    """
    Calculate realistic profits for all airports

    Args:
        revenue_by_airport: DataFrame with annual_revenue by airport
        lease_data: DataFrame with sqm and monthly_cost by airport

    Returns:
        DataFrame with detailed profit breakdown
    """
    results = []

    for airport in revenue_by_airport.index:
        if airport not in lease_data.index:
            print(f"Warning: No lease data for {airport}")
            continue

        profit_breakdown = calculate_realistic_profit(
            airport_code=airport,
            annual_revenue=revenue_by_airport.loc[airport, "annual_revenue"],
            store_sqm=lease_data.loc[airport, "sqm"],
            monthly_lease_cost=lease_data.loc[airport, "monthly_cost"],
        )

        profit_breakdown["airport"] = airport

        # ADD THESE LINES - Preserve passenger_count and avg_confidence from revenue_by_airport
        profit_breakdown["passenger_count"] = revenue_by_airport.loc[
            airport, "passenger_count"
        ]
        profit_breakdown["avg_confidence"] = revenue_by_airport.loc[
            airport, "avg_confidence"
        ]

        results.append(profit_breakdown)

    df_results = pd.DataFrame(results)
    df_results.set_index("airport", inplace=True)

    # Sort by net profit
    df_results.sort_values("net_profit", ascending=False, inplace=True)

    return df_results


# ============================================================================
# PRINT DETAILED BREAKDOWN
# ============================================================================


def print_profit_breakdown(airport_code, breakdown):
    """Print detailed profit breakdown for one airport"""

    print(f"\n{'=' * 80}")
    print(f"PROFIT BREAKDOWN: {airport_code} ({breakdown['country']})")
    print(f"{'=' * 80}")

    print(f"\n📊 REVENUE:")
    print(f"  Annual Revenue:        €{breakdown['annual_revenue']:>12,.0f}")
    print(f"  Monthly Revenue:       €{breakdown['monthly_revenue']:>12,.0f}")

    print(f"\n📦 COST OF GOODS SOLD:")
    print(f"  Base COGS (60%):       €{breakdown['base_cogs']:>12,.0f}")
    print(
        f"  Import Duties ({breakdown['import_duty_rate_percent']:.1f}%): €{breakdown['import_duties']:>12,.0f}"
    )
    if breakdown["has_eu_fta"]:
        print(f"    ✓ EU FTA: 80% duty reduction applied")
    print(
        f"  Total COGS:            €{breakdown['total_cogs']:>12,.0f} ({breakdown['cogs_percent']:.1f}%)"
    )

    print(f"\n💰 GROSS PROFIT:")
    print(f"  Gross Profit:          €{breakdown['gross_profit']:>12,.0f}")
    print(f"  Gross Margin:          {breakdown['gross_margin_percent']:>12.1f}%")

    print(f"\n💼 OPERATING EXPENSES:")
    print(f"  Staff Costs:           €{breakdown['staff_cost']:>12,.0f}")
    print(
        f"    ({breakdown['num_staff']:.1f} FTE for {breakdown['store_sqm']:.0f} sqm)"
    )
    print(f"  Lease/Rent:            €{breakdown['lease_cost']:>12,.0f}")
    print(f"  Other OpEx (5%):       €{breakdown['other_opex']:>12,.0f}")
    print(f"  Total OpEx:            €{breakdown['total_opex']:>12,.0f}")

    print(f"\n📈 EBIT:")
    print(f"  EBIT:                  €{breakdown['ebit']:>12,.0f}")
    print(f"  EBIT Margin:           {breakdown['ebit_margin_percent']:>12.1f}%")

    print(f"\n🏛️ TAXES:")
    print(
        f"  Corporate Tax ({breakdown['effective_tax_rate']:.1f}%): €{breakdown['corporate_tax']:>12,.0f}"
    )

    print(f"\n✅ NET PROFIT:")
    print(f"  Net Profit:            €{breakdown['net_profit']:>12,.0f}")
    print(f"  Profit Margin:         {breakdown['profit_margin_percent']:>12.1f}%")
    print(f"{'=' * 80}\n")


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("REALISTIC PROFIT CALCULATOR - TEST")
    print("=" * 80)

    # Test with Melbourne example
    test_breakdown = calculate_realistic_profit(
        airport_code="MEL", annual_revenue=872000, store_sqm=80, monthly_lease_cost=5600
    )

    print_profit_breakdown("MEL", test_breakdown)

    print("\n✓ Realistic profit calculator module ready!")
    print("Import this module in 3_prediction_analysis.py")
