from fastapi import FastAPI, HTTPException
import pandas as pd
import mysql.connector
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

app = FastAPI(title="VetCare AI Recommendation API")

# 1. HÀM KẾT NỐI VÀ TRAIN MODEL (Chạy 1 lần khi bật máy)
def load_and_train():
    try:
        conn = mysql.connector.connect(
            host="localhost", user="root", password="", database="local_kms"
        )
        # Hút dữ liệu giỏ hàng
        query = """
            SELECT o.order_id, p.name 
            FROM orders o 
            JOIN order_items oi ON o.order_id = oi.order_id 
            JOIN products p ON oi.product_id = p.product_id
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # Tiền xử lý dữ liệu cho FP-Growth
        transactions = df.groupby('order_id')['name'].apply(list).tolist()
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        df_matrix = pd.DataFrame(te_ary, columns=te.columns_)

        # Chạy thuật toán FP-Growth
        frequent_itemsets = fpgrowth(df_matrix, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        return rules
    except Exception as e:
        print(f"Lỗi kết nối Database: {e}")
        return None

# Load sẵn luật vào RAM để Web gọi là có ngay
rules = load_and_train()

# 2. ĐỊNH NGHĨA CỔNG CHỜ (ENDPOINT)
@app.get("/recommend")
def get_recommendation(product_name: str):
    if rules is None:
        raise HTTPException(status_code=500, detail="Model chưa được khởi tạo")

    # Tìm luật mà sản phẩm khách đang xem nằm ở vế trái (antecedents)
    results = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    
    if results.empty:
        return {"message": "Chưa có gợi ý cho sản phẩm này", "data": []}

    # Lấy top 5 sản phẩm gợi ý có độ tin cậy cao nhất
    recommendations = []
    top_results = results.sort_values(by='confidence', ascending=False).head(5)
    
    for _, row in top_results.iterrows():
        for item in row['consequents']:
            if item not in [r['name'] for r in recommendations]:
                recommendations.append({
                    "name": item,
                    "confidence": round(row['confidence'] * 100, 2),
                    "lift": round(row['lift'], 2)
                })

    return {
        "viewing": product_name,
        "recommendations": recommendations
    }

# Lệnh chạy server: uvicorn api_vetcare:app --reload