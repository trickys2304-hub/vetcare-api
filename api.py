from fastapi import FastAPI, HTTPException
import pandas as pd
import mysql.connector
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

app = FastAPI(title="VetCare AI Recommendation API")

def load_and_train():
    try:
        conn = mysql.connector.connect(
            host="mysql-326aceab-trickys2304-dce9.j.aivencloud.com", user="avnadmin", password=os.getenv('DB_PASSWORD'), database="defaultdb",ssl_disabled=False,ssl_ca=None,use_pure=True
        )
        query = """
            SELECT o.order_id, p.name 
            FROM orders o 
            JOIN order_items oi ON o.order_id = oi.order_id 
            JOIN products p ON oi.product_id = p.product_id
        """
        df = pd.read_sql(query, conn)
        conn.close()
        transactions = df.groupby('order_id')['name'].apply(list).tolist()
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        df_matrix = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = fpgrowth(df_matrix, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        print("--- Training hoàn tất từ dữ liệu Cloud! ---")
        return rules
    except Exception as e:
        print(f"Lỗi kết nối Database hoặc Training: {e}")
        return None
