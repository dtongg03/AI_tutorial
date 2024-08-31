import pandas as pd

file_path = 'D:/CODING/thuchanhAI/a.csv'
df = pd.read_csv(file_path)
#print(df.head(50))

# Tính tổng số bệnh nhân
total = len(df)
print(f"Tổng số bệnh nhân: {total}")

# Tính xác suất có bệnh tiểu đường
C1 = len(df[df['diabetes'] == 1]) / total
C2 = len(df[df['diabetes'] == 0]) / total

# Tính xác suất glucose và bloodpressure theo từng nhóm
glucose = int(input("Nhập giá trị glucose: "))
bloodpressure = int(input("Nhập giá trị bloodpressure: "))

Pgd1 = len(df[(df['glucose'] == glucose) & (df['diabetes'] == 1)]) / len(df[df['diabetes'] == 1])
Pgd2 = len(df[(df['glucose'] == glucose) & (df['diabetes'] == 0)]) / len(df[df['diabetes'] == 0])

Pbd1 = len(df[(df['bloodpressure'] == bloodpressure) & (df['diabetes'] == 1)]) / len(df[df['diabetes'] == 1])
Pbd2 = len(df[(df['bloodpressure'] == bloodpressure) & (df['diabetes'] == 0)]) / len(df[df['diabetes'] == 0])

# Tính xác suất P(X | Diabetes)
Px1 = Pgd1 * Pbd1
Px2 = Pgd2 * Pbd2



# So sánh và quyết định
if Px1 * C1 > Px2 * C2:
    print(f"Với glucose = {glucose} và bloodpressure = {bloodpressure}, kết luận là: Diabetes = 1 (có tiểu đường)")
else:
    print(f"Với glucose = {glucose} và bloodpressure = {bloodpressure}, kết luận là: Diabetes = 0 (không có tiểu đường)")
