import os
from fastapi import FastAPI, HTTPException
import pandas as pd
# Không cần mysql-connector nữa nếu dùng CSV
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

app = FastAPI(title="VetCare AI Recommendation API")

rules = None

def load_and_train():
    try:
        # CHỖ THAY ĐỔI QUAN TRỌNG: Đọc trực tiếp từ file data.csv ông vừa up lên
        # Đảm bảo file data.csv nằm cùng thư mục với file api.py
        if not os.path.exists('data.csv'):
            print("Lỗi: Không tìm thấy file data.csv trong thư mục!")
            return None
            
        df = pd.read_csv('data.csv')
        
        # Vì file data.csv của ông có 3 cột: order_id, product_id, name
        # Thuật toán cần gom nhóm theo order_id để lấy danh sách tên sản phẩm
        transactions = df.groupby('order_id')['name'].apply(list).tolist()
        
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        df_matrix = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Tính toán tập phổ biến (FP-Growth)
        # min_support 0.01 là 1%, nếu ít gợi ý ông có thể hạ xuống 0.005
        frequent_itemsets = fpgrowth(df_matrix, min_support=0.01, use_colnames=True)
        
        # Tạo luật kết hợp
        trained_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        print(f"--- Training hoàn tất! Đã học được {len(trained_rules)} luật kết hợp ---")
        return trained_rules
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu CSV: {e}")
        return None

@app.on_event("startup")
def startup_event():
    global rules
    rules = load_and_train()

@app.get("/")
def home():
    return {"message": "API VetCare đang chạy từ file CSV!", "status": "Online"}

@app.get("/recommend")
def get_recommendation(product_name: str):
    global rules
    if rules is None or rules.empty:
        # Nếu chưa có luật nào, trả về danh sách rỗng thay vì báo lỗi 503 để Web không bị treo
        return {"viewing": product_name, "recommendations": [], "message": "Chưa tìm thấy quy luật nào."}
    
    # Tìm các luật mà 'product_name' nằm trong phần tiền đề (antecedents)
    results = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    
    if results.empty:
        return {"viewing": product_name, "recommendations": []}
    
    recommendations = []
    # Lấy top 5 kết quả có độ tin cậy (confidence) cao nhất
    top_results = results.sort_values(by='confidence', ascending=False).head(5)
    
    for _, row in top_results.iterrows():
        for item in row['consequents']:
            if item not in [r['name'] for r in recommendations]:
                recommendations.append({
                    "name": item,
                    "confidence": round(row['confidence'] * 100, 2)
                })
                
    return {"viewing": product_name, "recommendations": recommendations}
