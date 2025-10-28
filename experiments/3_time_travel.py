"""
Time Travel & Versioning using Delta Lake
Demonstrates ELTL++ capability to track data lineage and reprocess
"""

import pandas as pd
from deltalake import DeltaTable, write_deltalake
from datetime import datetime
import time

def load_from_delta(path):
    """Load DataFrame from Delta Lake"""
    dt = DeltaTable(path)
    return dt.to_pandas()

def create_versioned_updates():
    """
    Simulate multiple updates to demonstrate versioning:
    Version 0: Initial load
    Version 1: Update order statuses
    Version 2: Correct pricing errors
    Version 3: Add new orders
    """
    print("\n" + "=" * 60)
    print("Creating Versioned Updates")
    print("=" * 60)
    
    # Load initial data
    orders = load_from_delta("data/raw_l1/orders")
    
    print(f"\nVersion 0 (Initial): {len(orders):,} orders")
    
    # Version 1: Update some order statuses
    print("\nCreating Version 1: Updating order statuses...")
    time.sleep(1)  # Ensure different timestamp
    orders_v1 = orders.copy()
    update_mask = orders_v1['order_status'] == 'PENDING'
    orders_v1.loc[update_mask, 'order_status'] = 'SHIPPED'
    write_deltalake("data/raw_l1/orders", orders_v1, mode='overwrite')
    print(f"  Updated {update_mask.sum():,} orders from PENDING → SHIPPED")
    
    # Version 2: Correct pricing errors
    print("\nCreating Version 2: Correcting pricing errors...")
    time.sleep(1)
    orders_v2 = orders_v1.copy()
    error_mask = orders_v2['total_amount'] < 0
    orders_v2.loc[error_mask, 'total_amount'] = abs(orders_v2.loc[error_mask, 'total_amount'])
    write_deltalake("data/raw_l1/orders", orders_v2, mode='overwrite')
    print(f"  Corrected {error_mask.sum():,} negative amounts")
    
    # Version 3: Add new orders with CONSISTENT datetime types
    print("\nCreating Version 3: Adding new orders...")
    time.sleep(1)
    
    # Get the maximum order_id
    max_order_id = int(orders_v2['order_id'].max())
    
    # Create new orders matching the exact schema of existing data
    now = pd.Timestamp.now()
    new_orders = pd.DataFrame({
        'order_id': range(max_order_id + 1, max_order_id + 1001),
        'customer_id': [1] * 1000,
        'order_date': [now] * 1000,  # Use pd.Timestamp
        'order_status': ['PENDING'] * 1000,
        'total_amount': [100.0] * 1000,
        'created_at': [now] * 1000,
        'age_days': [0] * 1000
    })
    
    # Ensure datetime columns match the schema
    new_orders['order_date'] = pd.to_datetime(new_orders['order_date'])
    new_orders['created_at'] = pd.to_datetime(new_orders['created_at'])
    
    # Concatenate with existing data
    orders_v3 = pd.concat([orders_v2, new_orders], ignore_index=True)
    
    # Ensure all datetime columns are properly typed
    orders_v3['order_date'] = pd.to_datetime(orders_v3['order_date'])
    orders_v3['created_at'] = pd.to_datetime(orders_v3['created_at'])
    
    write_deltalake("data/raw_l1/orders", orders_v3, mode='overwrite')
    print(f"  Added {len(new_orders):,} new orders")
    
    return orders, orders_v1, orders_v2, orders_v3

def demonstrate_time_travel():
    """
    Demonstrate Delta Lake time travel capabilities
    """
    print("\n" + "=" * 60)
    print("Time Travel & Versioning Demo")
    print("=" * 60)
    
    # Load Delta table
    dt = DeltaTable("data/raw_l1/orders")
    
    # Get version history
    history = dt.history()
    print(f"\nVersion History:")
    print(f"  Total versions: {len(history)}")
    
    for i, version_info in enumerate(history):
        timestamp = version_info.get('timestamp', 'Unknown')
        operation = version_info.get('operation', 'Unknown')
        print(f"  Version {i}: {operation} at {timestamp}")
    
    # Time travel to different versions
    print("\n" + "-" * 60)
    print("Time Travel Examples:")
    print("-" * 60)
    
    for version in range(min(4, len(history))):
        try:
            dt_version = DeltaTable("data/raw_l1/orders", version=version)
            df_version = dt_version.to_pandas()
            
            pending_count = (df_version['order_status'] == 'PENDING').sum()
            shipped_count = (df_version['order_status'] == 'SHIPPED').sum()
            negative_count = (df_version['total_amount'] < 0).sum()
            
            print(f"\nVersion {version}:")
            print(f"  Total orders: {len(df_version):,}")
            print(f"  PENDING orders: {pending_count:,}")
            print(f"  SHIPPED orders: {shipped_count:,}")
            print(f"  Negative amounts: {negative_count:,}")
            
        except Exception as e:
            print(f"\nVersion {version}: Not available ({str(e)})")
    
    return history

def demonstrate_rollback():
    """
    Demonstrate rollback capability (restore to previous version)
    """
    print("\n" + "=" * 60)
    print("Rollback Demonstration")
    print("=" * 60)
    
    dt = DeltaTable("data/raw_l1/orders")
    current_version = len(dt.history()) - 1
    
    print(f"\nCurrent version: {current_version}")
    print("  Rolling back to version 1...")
    
    # Load version 1
    try:
        dt_v1 = DeltaTable("data/raw_l1/orders", version=1)
        df_v1 = dt_v1.to_pandas()
        
        # Write as new version (this is the "rollback")
        write_deltalake("data/raw_l1/orders", df_v1, mode='overwrite')
        
        print(f"  ✓ Rolled back successfully")
        print(f"  New version: {len(DeltaTable('data/raw_l1/orders').history())}")
        
    except Exception as e:
        print(f"  ✗ Rollback failed: {str(e)}")

def main():
    print("=" * 60)
    print("ELTL++ Time Travel & Versioning")
    print("=" * 60)
    
    # Create versioned updates
    orders_v0, orders_v1, orders_v2, orders_v3 = create_versioned_updates()
    
    # Demonstrate time travel
    history = demonstrate_time_travel()
    
    # Demonstrate rollback
    # demonstrate_rollback()  # Commented out to preserve versions
    
    # Summary
    print("\n" + "=" * 60)
    print("Time Travel Summary")
    print("=" * 60)
    print(f"Total versions created: {len(history)}")
    print(f"Version 0 (Initial): {len(orders_v0):,} orders")
    print(f"Version 3 (Current): {len(orders_v3):,} orders")
    print(f"Growth: +{len(orders_v3) - len(orders_v0):,} orders")
    print("\n✓ Time travel and versioning capabilities validated")
    print("=" * 60)

if __name__ == "__main__":
    main()
