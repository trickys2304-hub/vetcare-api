from fastapi import FastAPI, HTTPException
import pandas as pd
import mysql.connector
import os  
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

app = FastAPI(title="VetCare AI Recommendation API")

rules = None
def load_and_train():
    try:
        db_password = os.getenv('DB_PASSWORD')
        
        conn = mysql.connector.connect(
            host="mysql-326aceab-trickys2304-dce9.j.aivencloud.com",
            user="avnadmin",
            password=db_password,
            database="defaultdb",
            ssl_disabled=False,
            use_pure=True
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
        trained_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        print("--- Training hoàn tất từ dữ liệu Cloud! ---")
        return trained_rules
    except Exception as e:
        print(f"Lỗi kết nối Database hoặc Training: {e}")
        return None
@app.on_event("startup")
def startup_event():
    global rules
    rules = load_and_train()
@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với VetCare AI API!", "status": "Online"}

@app.get("/recommend")
def get_recommendation(product_name: str):
    global rules
    if rules is None:
        raise HTTPException(status_code=503, detail="Model đang nạp dữ liệu, vui lòng thử lại sau vài giây")
    results = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    
    if results.empty:
        return {"viewing": product_name, "recommendations": [], "message": "Không tìm thấy gợi ý phù hợp"}
    
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
