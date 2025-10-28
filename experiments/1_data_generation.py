"""
Generate TPC-H-like dataset with temporal distribution
Creates realistic e-commerce data for ELTL++ validation
"""

import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

# Set seeds for reproducibility
Faker.seed(42)
random.seed(42)

fake = Faker()

def generate_customers(n=10000):
    """Generate customer dimension"""
    print(f"Generating {n} customers...")
    return pd.DataFrame({
        'customer_id': range(1, n+1),
        'customer_name': [fake.company() for _ in range(n)],
        'customer_segment': [random.choice(['ENTERPRISE', 'SMB', 'CONSUMER']) for _ in range(n)],
        'region': [random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST', 'CENTRAL']) for _ in range(n)],
        'created_date': [fake.date_between(start_date='-5y', end_date='today') for _ in range(n)]
    })

def generate_products(n=1000):
    """Generate product dimension"""
    print(f"Generating {n} products...")
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home', 'Sports', 'Toys']
    return pd.DataFrame({
        'product_id': range(1, n+1),
        'product_name': [fake.catch_phrase() for _ in range(n)],
        'category': [random.choice(categories) for _ in range(n)],
        'unit_price': [round(random.uniform(5, 500), 2) for _ in range(n)],
        'supplier_id': [random.randint(1, 100) for _ in range(n)]
    })

def generate_orders(customers, n=100000):
    """
    Generate orders with realistic temporal distribution:
    - 15% in last 90 days (hot tier)
    - 20% in 91-365 days (cool tier)
    - 65% older than 365 days (archive tier)
    """
    print(f"Generating {n} orders...")
    now = datetime.now()
    
    orders = []
    for i in range(1, n+1):
        # Temporal distribution
        rand = random.random()
        if rand < 0.15:  # 15% hot
            days_ago = random.randint(0, 90)
        elif rand < 0.35:  # 20% cool
            days_ago = random.randint(91, 365)
        else:  # 65% archive
            days_ago = random.randint(366, 2555)  # ~7 years
        
        order_date = now - timedelta(days=days_ago)
        
        orders.append({
            'order_id': i,
            'customer_id': random.choice(customers['customer_id'].values),
            'order_date': order_date,
            'order_status': random.choice(['PENDING', 'SHIPPED', 'DELIVERED', 'CANCELLED']),
            'total_amount': round(random.uniform(10, 5000), 2),
            'created_at': order_date,
            'age_days': days_ago  # Already calculated!
        })
    
    df = pd.DataFrame(orders)
    return df

def generate_lineitems(orders, products, avg_items=3):
    """Generate order line items"""
    print(f"Generating line items (avg {avg_items} per order)...")
    lineitems = []
    lineitem_id = 1
    
    for _, order in orders.iterrows():
        n_items = random.randint(1, avg_items * 2)
        for _ in range(n_items):
            product = products.sample(1).iloc[0]
            quantity = random.randint(1, 10)
            discount = round(random.uniform(0, 0.3), 2)
            
            lineitems.append({
                'lineitem_id': lineitem_id,
                'order_id': order['order_id'],
                'product_id': product['product_id'],
                'quantity': quantity,
                'price': product['unit_price'],
                'discount': discount,
                'line_total': round(quantity * product['unit_price'] * (1 - discount), 2)
            })
            lineitem_id += 1
    
    return pd.DataFrame(lineitems)

def introduce_quality_issues(df, issue_rate=0.05):
    """
    Introduce realistic data quality issues:
    - Missing values
    - Invalid values
    - Duplicates
    """
    print(f"Introducing {issue_rate*100}% quality issues...")
    df_dirty = df.copy()
    n_issues = int(len(df) * issue_rate)
    
    # Missing values in total_amount
    null_indices = random.sample(range(len(df)), min(n_issues // 3, len(df)))
    df_dirty.loc[null_indices, 'total_amount'] = None
    
    # Negative values (invalid)
    neg_indices = random.sample(range(len(df)), min(n_issues // 3, len(df)))
    for idx in neg_indices:
        if pd.notna(df_dirty.loc[idx, 'total_amount']):
            df_dirty.loc[idx, 'total_amount'] = -abs(df_dirty.loc[idx, 'total_amount'])
    
    # Duplicates
    dup_indices = random.sample(range(len(df)), min(n_issues // 3, len(df)))
    df_dirty = pd.concat([df_dirty, df_dirty.loc[dup_indices]], ignore_index=True)
    
    return df_dirty

def save_to_delta(df, path, mode='overwrite'):
    """Save DataFrame to Delta Lake format"""
    from deltalake import write_deltalake
    write_deltalake(path, df, mode=mode)
    print(f"Saved to Delta Lake: {path}")

def main():
    print("=" * 60)
    print("ELTL++ Data Generation")
    print("=" * 60)
    
    # Create directories
    Path("data/raw_l1").mkdir(parents=True, exist_ok=True)
    Path("data/semantic_l2").mkdir(parents=True, exist_ok=True)
    
    # Generate dimensions
    customers = generate_customers(10000)
    products = generate_products(1000)
    
    # Generate facts
    orders = generate_orders(customers, 100000)
    lineitems = generate_lineitems(orders, products)
    
    # Add quality issues to simulate real-world data
    orders_dirty = introduce_quality_issues(orders, issue_rate=0.05)
    
    # Save to Delta Lake (L1 - Raw Layer)
    print("\nSaving to L1 (Raw Layer)...")
    save_to_delta(customers, "data/raw_l1/customers")
    save_to_delta(products, "data/raw_l1/products")
    save_to_delta(orders_dirty, "data/raw_l1/orders")  # Dirty version
    save_to_delta(lineitems, "data/raw_l1/lineitems")
    
    # Statistics
    print("\n" + "=" * 60)
    print("Data Generation Summary")
    print("=" * 60)
    print(f"Customers: {len(customers):,}")
    print(f"Products: {len(products):,}")
    print(f"Orders: {len(orders_dirty):,} (includes {len(orders_dirty) - len(orders):,} quality issues)")
    print(f"Line Items: {len(lineitems):,}")
    print(f"\nTemporal Distribution:")
    hot_count = len(orders[orders['age_days'] <= 90])
    cool_count = len(orders[(orders['age_days'] > 90) & (orders['age_days'] <= 365)])
    archive_count = len(orders[orders['age_days'] > 365])
    print(f"  Hot (0-90 days): {hot_count:,} ({hot_count/len(orders)*100:.1f}%)")
    print(f"  Cool (91-365 days): {cool_count:,} ({cool_count/len(orders)*100:.1f}%)")
    print(f"  Archive (365+ days): {archive_count:,} ({archive_count/len(orders)*100:.1f}%)")
    
    total_size = (customers.memory_usage(deep=True).sum() + 
                  orders.memory_usage(deep=True).sum() + 
                  lineitems.memory_usage(deep=True).sum()) / 1024**2
    print(f"\nTotal Size: ~{total_size:.1f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
