import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sys

# --- HÀM KNN TỰ ĐỊNH NGHĨA ---
def KNN(X_train, X_test, y_train, k):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    y_pred = np.zeros((num_test, num_train))

    # Tính khoảng cách Euclidean
    for i in range(num_test):
        for j in range(num_train):
            y_pred[i, j] = np.sqrt(np.sum(np.power(X_test[i, :] - X_train[j, :], 2)))

    results = []
    # Sắp xếp và đếm
    for i in range(len(y_pred)):
        zipped = zip(y_pred[i, :], y_train)
        res = sorted(zipped, key=lambda x: x[0])
        results_topk = res[:k]

        classes = {}
        for _, j in results_topk:
            j = str(j) 
            if j not in classes:
                classes[j] = 1
            else:
                classes[j] = classes[j] + 1                
        # Trả về class có số lượng lớn nhất
        results.append(max(classes, key=classes.get))
    return np.array(results)

# --- HÀM HIỂN THỊ SƠ ĐỒ  ---
def visualize_knn(X_train, y_train, new_point, k, prediction):    
    plt.figure(figsize=(10, 7))
    
    # Gán màu cho từng lớp
    unique_classes = np.unique(y_train)
    
    try:
        cmap = plt.colormaps['viridis'] 
    except:
        cmap = plt.cm.get_cmap('viridis', len(unique_classes)) 
    
    num_classes = len(unique_classes)
    
    # Vẽ dữ liệu huấn luyện
    for i, class_label in enumerate(unique_classes):
        data = X_train[y_train == class_label]
        
        # Chỉ định màu
        if num_classes > 1 and i < 256: # Đảm bảo chỉ mục màu hợp lệ
             color_index = cmap(i / (num_classes - 1))
        elif num_classes == 1:
             color_index = cmap(0.5)
        else:
             color_index = cmap(i % 256)
             
        plt.scatter(data[:, 0], data[:, 1], color=color_index, marker='o', 
                    s=100, label=f'Mẫu đã biết: {class_label}')
    
    # Vẽ điểm mới
    plt.scatter(new_point[0, 0], new_point[0, 1], color='red', marker='D', 
                s=200, label=f'Mẫu mới được phân loại (K={k})', edgecolors='black')
    
    # Hiển thị kết quả phân loại
    plt.title(f'K-NN Phân loại (K={k}): Kết quả dự đoán là "{prediction[0]}"')
    plt.xlabel('Cân nặng (kg)')
    plt.ylabel('Chiều cao (cm)')
    plt.legend()
    plt.grid(True)    
    plt.show() 

# --- CHƯƠNG TRÌNH CHÍNH ---
def run_knn_for_customer():
    file_path = 'du_lieu_khach_hang.xlsx'    
    # --- TẢI DỮ LIỆU ---
    try:
        df = pd.read_excel(file_path)
        X_train = df[['Cân nặng (kg)', 'Chiều cao (cm)']].values
        y_train = df['Phân loại'].values
        
        print("Đã tải dữ liệu huấn luyện từ 'du_lieu_khach_hang.xlsx'.")
        print(f"   Tổng số mẫu: {len(X_train)}")
        print(f"   Các lớp phân loại: {np.unique(y_train)}")
        
    except FileNotFoundError:
        print("\nLỖI: Không tìm thấy file 'du_lieu_khach_hang.xlsx'.")
        print("Vui lòng đảm bảo file dữ liệu đã được tạo và nằm cùng thư mục với chương trình.")
        sys.exit()
    except Exception as e:
        print(f"\nLỖI KHI TẢI DỮ LIỆU: {e}")
        sys.exit()

    # --- NHẬP VÀ PHÂN LOẠI ---
    try:
        print("\n--- NHẬP DỮ LIỆU CẦN PHÂN LOẠI ---")
        
        customer_name = input("Nhập Tên khách hàng: ")
        weight = float(input("Nhập Cân nặng (kg): "))
        height = float(input("Nhập Chiều cao (cm): "))
        
        # Khách hàng chọn trọng số K
        k = int(input("Chọn trọng số K (số láng giềng muốn xét, ví dụ: 3): "))
        
        if k <= 0 or k > len(X_train):
            print(f"K phải là số nguyên dương và nhỏ hơn hoặc bằng số lượng mẫu ({len(X_train)}).")
            # Thoát nếu K không hợp lệ
            return 

        # Chuẩn bị dữ liệu mới (điểm cần phân loại)
        X_test_new = np.array([[weight, height]])

        # 4. Phân loại bằng Thuật toán K-NN
        prediction_results = KNN(X_train, X_test_new, y_train, k)
        predicted_class = prediction_results[0]

        # 5. Hiển thị Kết quả
        print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
        print(f"   Tên khách hàng: {customer_name}")
        print(f"   Cân nặng: {weight} kg, Chiều cao: {height} cm")
        print(f"   Trọng số K đã chọn: {k}")
        print(f"   Phân loại dự đoán: **{predicted_class}**")
        print("-------------------------")

        # Hiển thị Sơ đồ 
        print("Đang hiển thị sơ đồ. Vui lòng đóng cửa sổ để kết thúc chương trình.")
        visualize_knn(X_train, y_train, X_test_new, k, prediction_results)
        
        # 7. Ghi kết quả vào file (Tùy chọn)
        new_row = pd.DataFrame([{
            'Tên khách hàng': customer_name,
            'Cân nặng (kg)': weight,
            'Chiều cao (cm)': height,
            'Phân loại': predicted_class 
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        # Đảm bảo file Excel đã được đóng trước khi ghi
        df.to_excel(file_path, index=False)
        print(f"Đã thêm mẫu {customer_name} vào file Excel với phân loại {predicted_class}.")
        
    except ValueError:
        print("\nLỗi: Dữ liệu nhập phải là số (kg, cm) hoặc số nguyên dương (K).")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi không xác định: {e}")
        
    # --- KẾT THÚC CHƯƠNG TRÌNH ---
    print("\nKết thúc chương trình. Cảm ơn bạn đã sử dụng.")

# Chạy chương trình
run_knn_for_customer()