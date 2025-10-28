"""
Query Performance: Raw L1 vs Semantic L2
Compares query performance between normalized raw tables and denormalized semantic layer
"""

import pandas as pd
import time
from deltalake import DeltaTable
import duckdb
from tabulate import tabulate

def load_from_delta(path):
    """Load DataFrame from Delta Lake"""
    dt = DeltaTable(path)
    return dt.to_pandas()

def create_semantic_layer():
    """
    Create denormalized semantic layer (L2) from raw tables (L1)
    This is the core ELTL++ transformation
    """
    print("\n" + "=" * 60)
    print("Creating Semantic Layer (L2)")
    print("=" * 60)
    
    # Load raw data from L1
    customers = load_from_delta("data/raw_l1/customers")
    products = load_from_delta("data/raw_l1/products")
    orders = load_from_delta("data/raw_l1/orders")
    lineitems = load_from_delta("data/raw_l1/lineitems")
    
    print(f"\nL1 (Raw) tables loaded:")
    print(f"  Customers: {len(customers):,} rows")
    print(f"  Products: {len(products):,} rows")
    print(f"  Orders: {len(orders):,} rows")
    print(f"  Line Items: {len(lineitems):,} rows")
    
    # Create denormalized fact table
    print("\nBuilding denormalized fact_sales table...")
    fact_sales = (
        lineitems
        .merge(orders, on='order_id', how='left')
        .merge(customers, on='customer_id', how='left')
        .merge(products, on='product_id', how='left')
    )
    
    # Add computed columns
    fact_sales['extended_price'] = fact_sales['quantity'] * fact_sales['price']
    fact_sales['discount_amount'] = fact_sales['extended_price'] * fact_sales['discount']
    fact_sales['net_amount'] = fact_sales['extended_price'] - fact_sales['discount_amount']
    
    print(f"  fact_sales: {len(fact_sales):,} rows")
    
    # Create aggregated customer dimension
    print("\nBuilding dim_customer_summary...")
    dim_customer = fact_sales.groupby('customer_id').agg({
        'order_id': 'nunique',
        'net_amount': 'sum',
        'customer_name': 'first',
        'customer_segment': 'first',
        'region': 'first'
    }).reset_index()
    
    dim_customer.columns = [
        'customer_id', 'total_orders', 'lifetime_value',
        'customer_name', 'customer_segment', 'region'
    ]
    
    print(f"  dim_customer: {len(dim_customer):,} rows")
    
    # Save to L2
    from deltalake import write_deltalake
    write_deltalake("data/semantic_l2/fact_sales", fact_sales, mode='overwrite')
    write_deltalake("data/semantic_l2/dim_customer", dim_customer, mode='overwrite')
    
    print("\n✓ Semantic layer created and saved to L2")
    
    return fact_sales, dim_customer

def benchmark_query(name, query_func, iterations=30):
    """
    Run query multiple times and collect statistics
    """
    times = []
    result = None
    
    for i in range(iterations):
        start = time.time()
        result = query_func()
        elapsed = time.time() - start
        times.append(elapsed)
    
    times_sorted = sorted(times)
    
    return {
        'query': name,
        'iterations': iterations,
        'mean': sum(times) / len(times),
        'median': times_sorted[len(times) // 2],
        'p95': times_sorted[int(len(times) * 0.95)],
        'p99': times_sorted[int(len(times) * 0.99)],
        'min': min(times),
        'max': max(times),
        'result_rows': len(result) if hasattr(result, '__len__') else 1
    }

def run_performance_tests():
    """
    Run performance benchmarks comparing L1 (raw) vs L2 (semantic)
    """
    print("\n" + "=" * 60)
    print("Query Performance Benchmarks")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    customers = load_from_delta("data/raw_l1/customers")
    products = load_from_delta("data/raw_l1/products")
    orders = load_from_delta("data/raw_l1/orders")
    lineitems = load_from_delta("data/raw_l1/lineitems")
    fact_sales = load_from_delta("data/semantic_l2/fact_sales")
    dim_customer = load_from_delta("data/semantic_l2/dim_customer")
    
    # Use DuckDB for faster querying
    con = duckdb.connect(':memory:')
    con.register('customers', customers)
    con.register('products', products)
    con.register('orders', orders)
    con.register('lineitems', lineitems)
    con.register('fact_sales', fact_sales)
    con.register('dim_customer', dim_customer)
    
    print("✓ Data loaded into DuckDB")
    
    # Define queries
    queries = []
    
    # Query 1: Total revenue by customer segment
    def q1_raw():
        return con.execute("""
            SELECT 
                c.customer_segment,
                SUM(l.quantity * l.price * (1 - l.discount)) as total_revenue
            FROM lineitems l
            JOIN orders o ON l.order_id = o.order_id
            JOIN customers c ON o.customer_id = c.customer_id
            GROUP BY c.customer_segment
            ORDER BY total_revenue DESC
        """).df()
    
    def q1_semantic():
        return con.execute("""
            SELECT 
                customer_segment,
                SUM(net_amount) as total_revenue
            FROM fact_sales
            GROUP BY customer_segment
            ORDER BY total_revenue DESC
        """).df()
    
    queries.append(('Q1: Revenue by Segment', q1_raw, q1_semantic))
    
    # Query 2: Top 10 customers by revenue
    def q2_raw():
        return con.execute("""
            SELECT 
                c.customer_id,
                c.customer_name,
                SUM(l.quantity * l.price * (1 - l.discount)) as total_revenue
            FROM lineitems l
            JOIN orders o ON l.order_id = o.order_id
            JOIN customers c ON o.customer_id = c.customer_id
            GROUP BY c.customer_id, c.customer_name
            ORDER BY total_revenue DESC
            LIMIT 10
        """).df()
    
    def q2_semantic():
        return con.execute("""
            SELECT 
                customer_id,
                customer_name,
                lifetime_value as total_revenue
            FROM dim_customer
            ORDER BY lifetime_value DESC
            LIMIT 10
        """).df()
    
    queries.append(('Q2: Top 10 Customers', q2_raw, q2_semantic))
    
    # Query 3: Monthly revenue trend
    def q3_raw():
        return con.execute("""
            SELECT 
                DATE_TRUNC('month', o.order_date) as month,
                COUNT(DISTINCT o.order_id) as num_orders,
                SUM(l.quantity * l.price * (1 - l.discount)) as total_revenue
            FROM lineitems l
            JOIN orders o ON l.order_id = o.order_id
            WHERE o.order_date >= '2024-01-01'
            GROUP BY DATE_TRUNC('month', o.order_date)
            ORDER BY month
        """).df()
    
    def q3_semantic():
        return con.execute("""
            SELECT 
                DATE_TRUNC('month', order_date) as month,
                COUNT(DISTINCT order_id) as num_orders,
                SUM(net_amount) as total_revenue
            FROM fact_sales
            WHERE order_date >= '2024-01-01'
            GROUP BY DATE_TRUNC('month', order_date)
            ORDER BY month
        """).df()
    
    queries.append(('Q3: Monthly Revenue Trend', q3_raw, q3_semantic))
    
    # Query 4: Product category performance
    def q4_raw():
        return con.execute("""
            SELECT 
                p.category,
                COUNT(DISTINCT l.lineitem_id) as items_sold,
                SUM(l.quantity) as total_quantity,
                SUM(l.quantity * l.price * (1 - l.discount)) as total_revenue
            FROM lineitems l
            JOIN products p ON l.product_id = p.product_id
            GROUP BY p.category
            ORDER BY total_revenue DESC
        """).df()
    
    def q4_semantic():
        return con.execute("""
            SELECT 
                category,
                COUNT(DISTINCT lineitem_id) as items_sold,
                SUM(quantity) as total_quantity,
                SUM(net_amount) as total_revenue
            FROM fact_sales
            GROUP BY category
            ORDER BY total_revenue DESC
        """).df()
    
    queries.append(('Q4: Category Performance', q4_raw, q4_semantic))
    
    # Run benchmarks
    results = []
    
    for query_name, query_raw, query_semantic in queries:
        print(f"\nBenchmarking: {query_name}")
        print("  Testing raw L1...")
        result_raw = benchmark_query(f"{query_name} (Raw L1)", query_raw, iterations=30)
        print(f"    Median: {result_raw['median']*1000:.2f}ms")
        
        print("  Testing semantic L2...")
        result_semantic = benchmark_query(f"{query_name} (Semantic L2)", query_semantic, iterations=30)
        print(f"    Median: {result_semantic['median']*1000:.2f}ms")
        
        speedup = result_raw['median'] / result_semantic['median']
        print(f"    Speedup: {speedup:.2f}x")
        
        results.append(result_raw)
        results.append(result_semantic)
    
    con.close()
    
    return results

def display_results(results):
    """Display benchmark results in a table"""
    print("\n" + "=" * 60)
    print("Performance Results Summary")
    print("=" * 60)
    
    # Prepare data for table
    table_data = []
    for i in range(0, len(results), 2):
        raw = results[i]
        semantic = results[i+1]
        speedup = raw['median'] / semantic['median']
        
        table_data.append([
            raw['query'].replace(' (Raw L1)', ''),
            f"{raw['median']*1000:.2f}",
            f"{semantic['median']*1000:.2f}",
            f"{speedup:.2f}x",
            f"{(1 - semantic['median']/raw['median'])*100:.1f}%"
        ])
    
    headers = ['Query', 'Raw L1 (ms)', 'Semantic L2 (ms)', 'Speedup', 'Improvement']
    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Calculate averages
    raw_avg = sum(results[i]['median'] for i in range(0, len(results), 2)) / (len(results) // 2)
    semantic_avg = sum(results[i]['median'] for i in range(1, len(results), 2)) / (len(results) // 2)
    avg_speedup = raw_avg / semantic_avg
    
    print(f"\nOverall Average:")
    print(f"  Raw L1: {raw_avg*1000:.2f}ms")
    print(f"  Semantic L2: {semantic_avg*1000:.2f}ms")
    print(f"  Average Speedup: {avg_speedup:.2f}x")
    print(f"  Average Improvement: {(1 - semantic_avg/raw_avg)*100:.1f}%")
    print("=" * 60)

def main():
    print("=" * 60)
    print("ELTL++ Query Performance Validation")
    print("=" * 60)
    
    # Create semantic layer
    fact_sales, dim_customer = create_semantic_layer()
    
    # Run performance tests
    results = run_performance_tests()
    
    # Display results
    display_results(results)
    
    # Save results
    import json
    from pathlib import Path
    
    Path("results").mkdir(exist_ok=True)
    with open("results/performance_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to: results/performance_results.json")

if __name__ == "__main__":
    main()
