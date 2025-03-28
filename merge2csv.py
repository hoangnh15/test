import pandas as pd
import os
def merge_csv(csv_file1, csv_file2, csv_file3, csv_file4, merged_csv_file):
    try:
        # Kiểm tra sự tồn tại của các tệp CSV
        for file in [csv_file1, csv_file2, csv_file3, csv_file4]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")

        # Đọc các tệp CSV vào DataFrame
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
        df3 = pd.read_csv(csv_file3)
        df4 = pd.read_csv(csv_file4)
        
        # Hợp nhất DataFrame
        merged_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        
        # Ghi DataFrame đã hợp nhất vào tệp CSV mới
        merged_df.to_csv(merged_csv_file, index=False)
        
        print(f"Successfully merged files into {merged_csv_file}")
    except Exception as e:
        print(f"Error merging CSV files: {e}")
def merge_2csv(csv_file1, csv_file2, merged_csv_file):
    # Đọc hai file CSV vào DataFrame
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    # Trộn DataFrame lại với nhau sử dụng phương thức merge() hoặc concat()
    merged_df = pd.concat([df1, df2], ignore_index=True)  # Hoặc sử dụng df1.merge(df2) tùy vào yêu cầu
    
    # Ghi DataFrame đã trộn vào file CSV mới
    merged_df.to_csv(merged_csv_file, index=False)

# Thử nghiệm trộn hai file CSV
csv_file1 = "F:/Code proj/Malware/Dataset/output-pdfparser-B-CLEAN.csv"
csv_file2 = "F:/Code proj/Malware/Dataset/output-pdfparser-B-EVASE.csv"
csv_file3 = "F:/Code proj/Malware/Dataset/output-pdfparser-M-EVASE.csv"
csv_file4 = "F:/Code proj/Malware/Dataset/output-pdfparser-M-MALICIOUS.csv"
merged_csv_file = "PDFPARSER-DATASET.csv"
merge_csv(csv_file1, csv_file2, csv_file3, csv_file4, merged_csv_file)

