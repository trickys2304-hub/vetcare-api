import os
from fastapi import FastAPI, HTTPException
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

app = FastAPI(title="VetCare AI Recommendation API")

# Biến toàn cục lưu trữ luật kết hợp
rules = None

def load_and_train():
    try:
        # 1. Đọc dữ liệu từ file data.csv (Phải nằm cùng thư mục với api.py)
        file_path = "data.csv"
        if not os.path.exists(file_path):
            print(f"--- Lỗi: Không tìm thấy file {file_path} ---")
            return None
            
        print(f"--- Đang đọc dữ liệu từ {file_path} ---")
        df = pd.read_csv(file_path)

        # 2. Tiền xử lý dữ liệu: Gom nhóm sản phẩm theo từng đơn hàng (order_id)
        # data.csv của ông có 3 cột: order_id, product_id, name
        transactions = df.groupby('order_id')['name'].apply(list).tolist()

        # 3. Chuyển đổi dữ liệu sang dạng ma trận True/False cho FP-Growth
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        df_matrix = pd.DataFrame(te_ary, columns=te.columns_)

        # 4. Chạy thuật toán FP-Growth (min_support = 1% đơn hàng)
        frequent_itemsets = fpgrowth(df_matrix, min_support=0.01, use_colnames=True)
        
        # 5. Sinh luật kết hợp
        trained_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        print(f"--- Training hoàn tất! Đã học được {len(trained_rules)} quy luật ---")
        return trained_rules
        
    except Exception as e:
        print(f"--- Lỗi khi xử lý FP-Growth: {e} ---")
        return None

@app.on_event("startup")
def startup_event():
    global rules
    rules = load_and_train()

@app.get("/")
def home():
    return {
        "message": "API VetCare đang chạy từ file CSV thành công!",
        "status": "Online",
        "rules_count": len(rules) if rules is not None else 0
    }

@app.get("/recommend")
def get_recommendation(product_name: str):
    global rules
    if rules is None or rules.empty:
        return {"viewing": product_name, "recommendations": [], "message": "Hệ thống chưa nạp được dữ liệu luật."}
    
    # Tìm các luật mà product_name nằm trong 'antecedents'
    results = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    
    if results.empty:
        return {"viewing": product_name, "recommendations": []}
    
    recommendations = []
    # Lấy Top 5 gợi ý có độ tin cậy cao nhất
    top_results = results.sort_values(by='confidence', ascending=False).head(5)
    
    for _, row in top_results.iterrows():
        for item in row['consequents']:
            if item not in [r['name'] for r in recommendations]:
                recommendations.append({
                    "name": item,
                    "confidence": round(row['confidence'] * 100, 2)
                })
                
    return {"viewing": product_name, "recommendations": recommendations}
