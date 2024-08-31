import csv  # Nhập thư viện csv để đọc dữ liệu từ file CSV
from math import sqrt, pi, exp  # Nhập các hàm toán học: căn bậc hai, hằng số π, và hàm số mũ
from random import shuffle  # Nhập hàm shuffle để xáo trộn danh sách

def read_csv(filename):
    with open(filename, 'r') as file:  # Mở file CSV để đọc
        reader = csv.reader(file)  # Tạo đối tượng đọc CSV
        next(reader)  # Bỏ qua dòng tiêu đề
        data = [row for row in reader]  # Đọc tất cả các dòng dữ liệu vào danh sách
    return data  # Trả về dữ liệu đọc được

def str_to_float(data):
    return [[float(x) if i < len(row) - 1 else x for i, x in enumerate(row)] for row in data]  # Chuyển đổi các giá trị số từ chuỗi thành số thực

def split_data(data, test_ratio=0.3):
    shuffle(data)  # Xáo trộn dữ liệu ngẫu nhiên
    split_index = int(len(data) * (1 - test_ratio))  # Tính chỉ số phân tách dựa trên tỷ lệ dữ liệu kiểm tra
    train_data = data[:split_index]  # Lấy dữ liệu huấn luyện
    test_data = data[split_index:]  # Lấy dữ liệu kiểm tra
    return train_data, test_data  # Trả về dữ liệu huấn luyện và kiểm tra

def separate_by_class(data):
    separated = {}  # Khởi tạo từ điển để lưu trữ dữ liệu phân loại
    for row in data:
        class_value = row[-1]  # Lấy giá trị lớp từ cột cuối cùng
        if class_value not in separated:
            separated[class_value] = []  # Khởi tạo danh sách cho lớp mới
        separated[class_value].append(row[:-1])  # Thêm dữ liệu vào danh sách tương ứng với lớp
    return separated  # Trả về dữ liệu phân loại

def calculate_stats(numbers):
    n = len(numbers)  # Số lượng giá trị
    mean = sum(numbers) / n  # Tính giá trị trung bình
    variance = sum((x - mean) ** 2 for x in numbers) / (n - 1) if n > 1 else 0  # Tính phương sai
    stdev = sqrt(variance) if variance > 0 else 1e-6  # Tính độ lệch chuẩn, tránh giá trị bằng 0
    return mean, stdev  # Trả về trung bình và độ lệch chuẩn

def summarize_dataset(dataset):
    summaries = {}  # Khởi tạo từ điển để lưu trữ tóm tắt dữ liệu
    for class_value, instances in dataset.items():
        summaries[class_value] = {
            'summaries': [calculate_stats(attribute) for attribute in zip(*instances)],  # Tóm tắt dữ liệu của lớp
            'count': len(instances)  # Số lượng mẫu của lớp
        }
    return summaries  # Trả về tóm tắt dữ liệu của tất cả các lớp

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))  # Tính toán phần số mũ của phân phối Gaussian
    return (1 / (sqrt(2 * pi) * stdev)) * exponent  # Tính toán xác suất theo phân phối Gaussian

def calculate_class_probabilities(summaries, row):
    total_rows = sum(class_info['count'] for class_info in summaries.values())  # Tổng số mẫu
    probabilities = {}  # Khởi tạo từ điển để lưu trữ xác suất lớp
    for class_value, class_info in summaries.items():
        probabilities[class_value] = class_info['count'] / total_rows  # Tính xác suất của lớp
        for i in range(len(class_info['summaries'])):
            mean, stdev = class_info['summaries'][i]  # Lấy trung bình và độ lệch chuẩn của thuộc tính
            x = row[i]  # Lấy giá trị thuộc tính từ hàng
            prob = calculate_probability(x, mean, stdev)  # Tính xác suất của thuộc tính
            probabilities[class_value] *= prob  # Nhân xác suất của thuộc tính vào xác suất lớp
    return probabilities  # Trả về xác suất của tất cả các lớp

def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)  # Tính xác suất của tất cả các lớp cho hàng
    best_label, best_prob = None, -1  # Khởi tạo nhãn và xác suất tốt nhất
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:  # Cập nhật nhãn và xác suất tốt nhất
            best_prob = probability
            best_label = class_value
    return best_label  # Trả về nhãn có xác suất cao nhất

def evaluate_model(test_set, model):
    correct = 0  # Khởi tạo số lượng dự đoán chính xác
    for row in test_set:
        prediction = predict(model, row[:-1])  # Dự đoán lớp của hàng
        if prediction == row[-1]:  # So sánh dự đoán với giá trị thực
            correct += 1  # Tăng số lượng dự đoán chính xác
    return correct / len(test_set)  # Tính toán và trả về độ chính xác của mô hình

def naive_bayes(train):
    separated = separate_by_class(train)  # Phân loại dữ liệu huấn luyện
    return summarize_dataset(separated)  # Tóm tắt dữ liệu của các lớp

if __name__ == "__main__":
    data = read_csv(r'D:\CODING\iris\iris.csv')  # Đọc dữ liệu từ file CSV
    data = str_to_float(data)  # Chuyển đổi các giá trị từ chuỗi thành số thực
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra sử dụng Hold-Out
    train_data, test_data = split_data(data, test_ratio=0.1)
    # Huấn luyện mô hình trên tập huấn luyện
    model = naive_bayes(train_data)
    # Đánh giá mô hình trên tập kiểm tra
    accuracy = evaluate_model(test_data, model)
    # In kết quả độ chính xác
    print(f'Hiệu SUất: {accuracy:.2f}')
