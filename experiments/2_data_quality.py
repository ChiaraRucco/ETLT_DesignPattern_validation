"""
Data Quality & Contract Validation
Demonstrates ELTL++ data governance capabilities using simple validation rules
"""

import pandas as pd
from deltalake import DeltaTable, write_deltalake
import json
from pathlib import Path

def load_from_delta(path):
    """Load DataFrame from Delta Lake"""
    dt = DeltaTable(path)
    return dt.to_pandas()

def create_data_contract():
    """
    Define data contract using simple validation rules
    This enforces schema and quality rules at L1 → L2 boundary
    """
    print("\n" + "=" * 60)
    print("Creating Data Contract")
    print("=" * 60)
    
    contract = {
        'orders': [
            {
                'rule': 'column_exists',
                'column': 'order_id',
                'description': 'Order ID must exist'
            },
            {
                'rule': 'not_null',
                'column': 'order_id',
                'description': 'Order ID cannot be null'
            },
            {
                'rule': 'unique',
                'column': 'order_id',
                'description': 'Order ID must be unique'
            },
            {
                'rule': 'not_null',
                'column': 'customer_id',
                'description': 'Customer ID cannot be null'
            },
            {
                'rule': 'not_null',
                'column': 'total_amount',
                'description': 'Total amount cannot be null'
            },
            {
                'rule': 'between',
                'column': 'total_amount',
                'min_value': 0,
                'max_value': 100000,
                'description': 'Total amount must be between 0 and 100,000'
            },
            {
                'rule': 'in_set',
                'column': 'order_status',
                'values': ['PENDING', 'SHIPPED', 'DELIVERED', 'CANCELLED'],
                'description': 'Order status must be valid'
            }
        ],
        'lineitems': [
            {
                'rule': 'between',
                'column': 'quantity',
                'min_value': 1,
                'max_value': 100,
                'description': 'Quantity must be between 1 and 100'
            },
            {
                'rule': 'between',
                'column': 'discount',
                'min_value': 0,
                'max_value': 1,
                'description': 'Discount must be between 0 and 1'
            }
        ]
    }
    
    return contract

def validate_rule(df, rule):
    """
    Validate a single rule against a DataFrame
    Returns: (passed, success_count, total_count, details)
    """
    column = rule.get('column')
    rule_type = rule['rule']
    
    try:
        if rule_type == 'column_exists':
            passed = column in df.columns
            success_count = len(df) if passed else 0
            total_count = len(df)
            
        elif rule_type == 'not_null':
            null_count = df[column].isnull().sum()
            passed = null_count == 0
            success_count = len(df) - null_count
            total_count = len(df)
            
        elif rule_type == 'unique':
            dup_count = df[column].duplicated().sum()
            passed = dup_count == 0
            success_count = len(df) - dup_count
            total_count = len(df)
            
        elif rule_type == 'between':
            min_val = rule['min_value']
            max_val = rule['max_value']
            valid_mask = df[column].between(min_val, max_val)
            success_count = valid_mask.sum()
            passed = success_count == len(df)
            total_count = len(df)
            
        elif rule_type == 'in_set':
            valid_values = rule['values']
            valid_mask = df[column].isin(valid_values)
            success_count = valid_mask.sum()
            passed = success_count == len(df)
            total_count = len(df)
        
        else:
            passed = True
            success_count = len(df)
            total_count = len(df)
        
        return passed, success_count, total_count, None
        
    except Exception as e:
        return False, 0, len(df), str(e)

def validate_data_quality(df, table_name, contract):
    """
    Validate data quality against contract
    Returns: (is_valid, validation_results)
    """
    print(f"\nValidating {table_name}...")
    
    rules = contract.get(table_name, [])
    results = []
    
    for rule in rules:
        passed, success_count, total_count, error = validate_rule(df, rule)
        
        result = {
            'rule': rule['rule'],
            'column': rule.get('column'),
            'description': rule['description'],
            'passed': passed,
            'success_count': int(success_count),
            'total_count': int(total_count),
            'success_rate': float(success_count / total_count * 100) if total_count > 0 else 0
        }
        
        if error:
            result['error'] = error
        
        results.append(result)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {rule['description']} ({success_count}/{total_count} rows)")
    
    overall_passed = all(r['passed'] for r in results)
    return overall_passed, results

def clean_and_promote_to_l2(df_dirty, validation_results):
    """
    Clean data based on validation results and promote to L2
    This is the ELTL++ transformation step: L1 (raw) → L2 (semantic)
    """
    print("\nCleaning data for L2 promotion...")
    df_clean = df_dirty.copy()
    
    # Remove nulls in critical columns
    df_clean = df_clean[df_clean['total_amount'].notna()]
    df_clean = df_clean[df_clean['customer_id'].notna()]
    
    # Remove invalid values
    df_clean = df_clean[df_clean['total_amount'] >= 0]
    df_clean = df_clean[df_clean['total_amount'] <= 100000]
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['order_id'])
    
    # Ensure valid statuses
    valid_statuses = ['PENDING', 'SHIPPED', 'DELIVERED', 'CANCELLED']
    df_clean = df_clean[df_clean['order_status'].isin(valid_statuses)]
    
    removed_count = len(df_dirty) - len(df_clean)
    print(f"  Removed {removed_count:,} invalid rows ({removed_count/len(df_dirty)*100:.2f}%)")
    print(f"  Promoted {len(df_clean):,} clean rows to L2")
    
    return df_clean

def main():
    print("=" * 60)
    print("ELTL++ Data Quality & Contract Validation")
    print("=" * 60)
    
    # Load raw data from L1
    orders_dirty = load_from_delta("data/raw_l1/orders")
    lineitems = load_from_delta("data/raw_l1/lineitems")
    
    print(f"\nLoaded from L1:")
    print(f"  Orders (raw): {len(orders_dirty):,} rows")
    print(f"  Line Items: {len(lineitems):,} rows")
    
    # Create data contract
    contract = create_data_contract()
    print(f"\nData contract created with {len(contract['orders'])} rules for orders")
    
    # Validate against contract
    orders_valid, orders_results = validate_data_quality(orders_dirty, 'orders', contract)
    lineitems_valid, lineitems_results = validate_data_quality(lineitems, 'lineitems', contract)
    
    # Clean and promote to L2 if validation fails
    if not orders_valid:
        orders_clean = clean_and_promote_to_l2(orders_dirty, orders_results)
        
        # Re-validate cleaned data
        print("\nRe-validating cleaned data...")
        orders_valid_clean, orders_results_clean = validate_data_quality(orders_clean, 'orders', contract)
        
        if orders_valid_clean:
            print("\n✓ Cleaned data passes all quality checks!")
            
            # Save to L2 (Semantic Layer)
            Path("data/semantic_l2").mkdir(parents=True, exist_ok=True)
            write_deltalake("data/semantic_l2/orders_clean", orders_clean, mode='overwrite')
            print("  Saved to L2: data/semantic_l2/orders_clean")
        else:
            print("\n⚠ Some issues remain after cleaning:")
            for result in orders_results_clean:
                if not result['passed']:
                    print(f"    - {result['description']}: {result['success_count']}/{result['total_count']}")
    else:
        print("\n✓ Orders data passes all quality checks!")
        orders_clean = orders_dirty
        write_deltalake("data/semantic_l2/orders_clean", orders_clean, mode='overwrite')
    
    # Summary
    print("\n" + "=" * 60)
    print("Data Quality Summary")
    print("=" * 60)
    print(f"L1 (Raw) Orders: {len(orders_dirty):,} rows")
    if not orders_valid:
        print(f"L2 (Clean) Orders: {len(orders_clean):,} rows")
        print(f"Data Quality Rate: {len(orders_clean)/len(orders_dirty)*100:.2f}%")
    else:
        print(f"Data Quality Rate: 100%")
    
    print(f"\nContract Validation:")
    print(f"  Orders: {'✓ PASS' if orders_valid else '✗ FAIL (cleaned and promoted)'}")
    print(f"  Line Items: {'✓ PASS' if lineitems_valid else '✗ FAIL'}")
    
    # Detailed results
    failed_rules = [r for r in orders_results if not r['passed']]
    if failed_rules:
        print(f"\nFailed Rules ({len(failed_rules)}):")
        for rule in failed_rules:
            print(f"  - {rule['description']}")
            print(f"    Success rate: {rule['success_rate']:.1f}%")
    
    print("=" * 60)
    
    # Save validation results
    Path("results").mkdir(exist_ok=True)
    with open("results/data_quality_results.json", 'w') as f:
        json.dump({
            'orders': orders_results,
            'lineitems': lineitems_results,
            'summary': {
                'l1_rows': int(len(orders_dirty)),
                'l2_rows': int(len(orders_clean)),
                'quality_rate': float(len(orders_clean)/len(orders_dirty)*100),
                'orders_passed': orders_valid,
                'lineitems_passed': lineitems_valid
            }
        }, f, indent=2, default=str)
    print("\nResults saved to: results/data_quality_results.json")

if __name__ == "__main__":
    main()
