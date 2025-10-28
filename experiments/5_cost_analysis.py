"""
Cost Analysis: ELTL vs ELTL++
Corrected cost model with realistic storage and compute pricing
"""

import pandas as pd
from deltalake import DeltaTable
import os
from pathlib import Path

def get_directory_size_mb(path):
    """Calculate directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def calculate_storage_costs():
    """
    Calculate storage costs for ELTL vs ELTL++
    Using realistic Azure Blob Storage pricing (West Europe, Dec 2024)
    """
    print("\n" + "=" * 60)
    print("Storage Cost Analysis")
    print("=" * 60)
    
    # Get actual data sizes
    l1_size_mb = get_directory_size_mb("data/raw_l1")
    l2_size_mb = get_directory_size_mb("data/semantic_l2")
    
    print(f"\nActual Data Sizes:")
    print(f"  L1 (Raw): {l1_size_mb:.2f} MB")
    print(f"  L2 (Semantic): {l2_size_mb:.2f} MB")
    
    # Scale up to realistic enterprise scenario
    # Assume this is a daily snapshot, scale to 1 year
    daily_growth_mb = l1_size_mb  # New data added daily
    days_per_year = 365
    
    # Temporal distribution (based on our data generation)
    hot_percent = 0.15   # 0-90 days
    cool_percent = 0.20  # 91-365 days
    archive_percent = 0.65  # 365+ days
    
    # Calculate yearly accumulation
    total_year_mb = daily_growth_mb * days_per_year
    
    print(f"\nScaled Annual Scenario:")
    print(f"  Daily ingestion: {daily_growth_mb:.2f} MB")
    print(f"  Annual accumulation: {total_year_mb:.2f} MB ({total_year_mb/1024:.2f} GB)")
    
    # Azure Blob Storage Pricing (per GB-month, West Europe)
    # Source: https://azure.microsoft.com/en-us/pricing/details/storage/blobs/
    PRICING = {
        'hot_storage_gb_month': 0.0184,      # Hot tier
        'cool_storage_gb_month': 0.010,      # Cool tier
        'archive_storage_gb_month': 0.002,   # Archive tier
        'hot_read_per_10k': 0.0044,          # Read operations
        'cool_read_per_10k': 0.01,
        'archive_read_per_10k': 5.50,        # Expensive!
        'archive_rehydration_gb': 0.022,     # Rehydration cost
    }
    
    total_gb = total_year_mb / 1024
    
    # ELTL Baseline: All data in Hot tier
    print("\n" + "-" * 60)
    print("ELTL (Baseline) - All Hot Storage")
    print("-" * 60)
    
    eltl_hot_gb = total_gb
    eltl_storage_cost = eltl_hot_gb * PRICING['hot_storage_gb_month'] * 12  # Annual
    
    # Access patterns: 80% of queries hit last 90 days
    # Assume 1000 queries/month, each scans average 10 GB
    monthly_queries = 1000
    avg_scan_gb_per_query = 10
    eltl_read_ops = monthly_queries * (avg_scan_gb_per_query * 1024 / 64) / 10000  # 64KB blocks
    eltl_read_cost = eltl_read_ops * PRICING['hot_read_per_10k'] * 12
    
    # Compute cost: Higher because querying raw data requires more processing
    # Assume Azure Synapse Serverless: $5/TB scanned
    eltl_compute_cost = (monthly_queries * avg_scan_gb_per_query / 1024) * 5 * 12
    
    eltl_total = eltl_storage_cost + eltl_read_cost + eltl_compute_cost
    
    print(f"  Hot storage ({eltl_hot_gb:.2f} GB): ${eltl_storage_cost:.2f}/year")
    print(f"  Read operations: ${eltl_read_cost:.2f}/year")
    print(f"  Compute (query processing): ${eltl_compute_cost:.2f}/year")
    print(f"  TOTAL: ${eltl_total:.2f}/year")
    
    # ELTL++: Tiered storage + Semantic layer
    print("\n" + "-" * 60)
    print("ELTL++ - Tiered Storage + Semantic Layer")
    print("-" * 60)
    
    eltl_pp_hot_gb = total_gb * hot_percent
    eltl_pp_cool_gb = total_gb * cool_percent
    eltl_pp_archive_gb = total_gb * archive_percent
    
    # Storage costs
    eltl_pp_hot_storage = eltl_pp_hot_gb * PRICING['hot_storage_gb_month'] * 12
    eltl_pp_cool_storage = eltl_pp_cool_gb * PRICING['cool_storage_gb_month'] * 12
    eltl_pp_archive_storage = eltl_pp_archive_gb * PRICING['archive_storage_gb_month'] * 12
    eltl_pp_storage_cost = eltl_pp_hot_storage + eltl_pp_cool_storage + eltl_pp_archive_storage
    
    # L2 storage (smaller, denormalized)
    l2_overhead_factor = 1.5  # Denormalization adds ~50% overhead
    eltl_pp_l2_storage = (eltl_pp_hot_gb * l2_overhead_factor) * PRICING['hot_storage_gb_month'] * 12
    
    # Read operations: Reduced because 80% hit hot tier, occasional cool/archive access
    hot_read_percent = 0.80
    cool_read_percent = 0.15
    archive_read_percent = 0.05
    
    eltl_pp_hot_read_ops = (monthly_queries * hot_read_percent * avg_scan_gb_per_query * 1024 / 64) / 10000
    eltl_pp_cool_read_ops = (monthly_queries * cool_read_percent * avg_scan_gb_per_query * 1024 / 64) / 10000
    eltl_pp_archive_read_ops = (monthly_queries * archive_read_percent * avg_scan_gb_per_query * 1024 / 64) / 10000
    
    eltl_pp_read_cost = (
        eltl_pp_hot_read_ops * PRICING['hot_read_per_10k'] +
        eltl_pp_cool_read_ops * PRICING['cool_read_per_10k'] +
        eltl_pp_archive_read_ops * PRICING['archive_read_per_10k']
    ) * 12
    
    # Archive rehydration (occasional)
    archive_rehydration_gb_month = eltl_pp_archive_gb * 0.05  # 5% accessed monthly
    eltl_pp_rehydration_cost = archive_rehydration_gb_month * PRICING['archive_rehydration_gb'] * 12
    
    # Compute cost: Lower because semantic layer reduces data scanned
    # Semantic layer reduces scan volume by ~70% (from performance tests)
    scan_reduction_factor = 0.3
    eltl_pp_compute_cost = (monthly_queries * avg_scan_gb_per_query * scan_reduction_factor / 1024) * 5 * 12
    
    eltl_pp_total = (
        eltl_pp_storage_cost + 
        eltl_pp_l2_storage + 
        eltl_pp_read_cost + 
        eltl_pp_rehydration_cost + 
        eltl_pp_compute_cost
    )
    
    print(f"  L1 Storage:")
    print(f"    Hot ({eltl_pp_hot_gb:.2f} GB): ${eltl_pp_hot_storage:.2f}/year")
    print(f"    Cool ({eltl_pp_cool_gb:.2f} GB): ${eltl_pp_cool_storage:.2f}/year")
    print(f"    Archive ({eltl_pp_archive_gb:.2f} GB): ${eltl_pp_archive_storage:.2f}/year")
    print(f"  L2 Storage (semantic): ${eltl_pp_l2_storage:.2f}/year")
    print(f"  Read operations: ${eltl_pp_read_cost:.2f}/year")
    print(f"  Archive rehydration: ${eltl_pp_rehydration_cost:.2f}/year")
    print(f"  Compute (optimized): ${eltl_pp_compute_cost:.2f}/year")
    print(f"  TOTAL: ${eltl_pp_total:.2f}/year")
    
    # Summary
    print("\n" + "=" * 60)
    print("Cost Comparison Summary")
    print("=" * 60)
    
    savings = eltl_total - eltl_pp_total
    savings_percent = (savings / eltl_total) * 100
    
    comparison_data = {
        'Architecture': ['ELTL', 'ELTL++'],
        'Storage': [f"${eltl_storage_cost:.2f}", f"${eltl_pp_storage_cost + eltl_pp_l2_storage:.2f}"],
        'Read Ops': [f"${eltl_read_cost:.2f}", f"${eltl_pp_read_cost + eltl_pp_rehydration_cost:.2f}"],
        'Compute': [f"${eltl_compute_cost:.2f}", f"${eltl_pp_compute_cost:.2f}"],
        'Total/Year': [f"${eltl_total:.2f}", f"${eltl_pp_total:.2f}"],
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    print(f"\nCost Reduction: ${savings:.2f}/year ({savings_percent:.1f}%)")
    print(f"ROI Period: Immediate (operational savings)")
    print("=" * 60)
    
    return {
        'eltl_total': eltl_total,
        'eltl_pp_total': eltl_pp_total,
        'savings': savings,
        'savings_percent': savings_percent
    }

def main():
    print("=" * 60)
    print("ELTL++ Cost Analysis")
    print("=" * 60)
    
    cost_results = calculate_storage_costs()
    
    # Save results
    import json
    Path("results").mkdir(exist_ok=True)
    with open("results/cost_analysis_results.json", 'w') as f:
        json.dump(cost_results, f, indent=2)
    
    print("\nResults saved to: results/cost_analysis_results.json")

if __name__ == "__main__":
    main()
