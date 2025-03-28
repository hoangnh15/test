import subprocess
import re
import json
import os
import pandas as pd

def extract_pdf_features(pdf_file):
    try:
        # Chạy lệnh pdf-parser và lấy output
        output = subprocess.run(["pdf-parser", pdf_file], capture_output=True, text=True).stdout

        # Tạo một dictionary để lưu thông tin
        pdf_info = {}

        # Các từ khóa cần tìm
        keywords = [
            'JS', 'JavaScript', 'Size', 'startxref', '%EOF', 'Producer', 'ProcSet',
            'ID', 'S', 'CreationDate', 'obj', 'xref', 'Font', 'XObject', 'ModDate',
            'Info', 'XML', 'Comment', 'Widget', 'Referencing', 'FontDescriptor',
            'Image', 'Rect', 'Length', 'Action'
        ]

        # Sử dụng một từ điển tạm để lưu kết quả của re.findall
        keyword_counts = {keyword: len(re.findall(fr'/{keyword}', output)) for keyword in keywords}

        # Chuyển kết quả vào pdf_info
        pdf_info.update(keyword_counts)
        pdf_info['xref'] -= pdf_info['startxref']
        pdf_info['≪'] = output.count('<<')
        pdf_info['≫'] = output.count('>>')

        return pdf_info
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        return None

def save_to_json(pdf_info, output_file):
    with open(output_file, 'w') as f:
        json.dump(pdf_info, f, indent=4)

def list_files_in_folder(folder_path):
    file_list = []
    list_js = []
    count_file = 1
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    for file_path in file_list:
        print(f"Processing file {count_file}/{len(file_list)}: {file_path}")
        count_file += 1
        try:
            output = extract_pdf_features(file_path)
            if output is None:
                continue

            file_name = os.path.basename(file_path)

            output['header_length'] = len(file_name)
            output['small_content'] = 1 if output['obj'] <= 14 else 0

            count = sum(1 for key in ['JS', 'JavaScript', 'Action'] if output[key] >= 1)
            output['malicecontent'] = 1 if count >= 2 else 0
            output['label'] = 0
            list_js.append(output)
        except Exception as e:
            print("Error:", e)

    return list_js

def json_to_csv(json_data, csv_file):
    # Chuyển đổi dữ liệu JSON thành DataFrame
    df = pd.DataFrame(json_data)
    # Chuyển đổi DataFrame thành file CSV
    df.to_csv(csv_file, index=False)

def main():
    folder_path = "./test_folder/Benign/Clean"
    list_js = list_files_in_folder(folder_path)

    output_file = "pdfparser_js-B-Clean.json"
    with open(output_file, 'w') as f:
        json.dump(list_js, f, indent=4)
    print("Successfully saved list_js to", output_file)

    csv_file = "output-pdfparser-B-CLEAN.csv"
    json_to_csv(list_js, csv_file)
    print("Successfully saved JSON data to", csv_file)

if __name__ == "__main__":
    main()
