import subprocess
import json
import os
import pandas as pd
from pdfparser2json import extract_pdf_features
from pdfid.extraction import pdf_analysis
def save_to_json(pdf_info, output_file):
    with open(output_file, 'w') as f:
        json.dump(pdf_info, f, indent=4)
        
def pdfinfo_to_json(pdf_file):
    # Chạy lệnh pdfinfo và lấy output
    output = subprocess.check_output(['pdfinfo', pdf_file]).decode('utf-8')
    
    # Khởi tạo một dictionary để lưu thông tin
    pdf_info = {}
    
    # Danh sách các trường thông tin cần thiết
    required_fields = ['Metadata Stream', 'Tagged', 'UserProperties', 'Suspects', 'Form',
                       'JavaScript', 'Pages', 'Encrypted', 'Page size', 'Page rot',
                       'File size', 'Optimized', 'PDF version']
    
    # Xử lý output từ pdfinfo
    
    for field in required_fields:
        pdf_info[field] = 0  # Mặc định giá trị là 0
    
    '''pdf_info["label"] = 1 '''

    for line in output.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Kiểm tra xem key có trong danh sách cần thiết không
            if key in required_fields:
                # Kiểm tra xem key có phải là PDF version không
                if key == 'PDF version':
                    # Nếu không có PDF version, không thêm vào dictionary
                    if value == '':
                        continue
                    else:
                        
                        version_str = value.strip().split()[0]
                        if version_str.startswith("1.") and version_str[2:] in ['0', '1', '2', '3', '4', '5', '6', '7']:
                        	value = int(version_str[2:])
                        else:
                        	value = 8
                if key == 'Encrypted':
                	if 'yes' in value:
                        	value = 1
                # Kiểm tra các giá trị yes/no và chuyển đổi thành 1/0
                if value == 'yes':
                    value = 1
                elif value == 'no':
                    value = 0
                if key in ['Pages', 'Page rot']:
                    value = int(value)
                # Kiểm tra nếu key là "Page size"
                if key == 'Page size':
                    # Kiểm tra xem value có chứa 'A4', 'Letter', 'A3' không
                    if 'A4' in value:
                        value = 0
                    elif 'letter' in value:
                        value = 1
                    elif 'A3' in value:
                        value = 2
                    else:
                        value = 3
                if key == 'Form':
                    if 'XFA' in value:
                        value = 0
                    elif 'AcroForm' in value:
                        value = 1
                    elif 'none' in value:
                        value = 2
                    else:
                        value = 3
                if key == 'File size':
                    value = int(value.split()[0]) / 1024
                
                
                
                # Gán giá trị đã xử lý cho key tương ứng trong pdf_info
                pdf_info[key] = value
    
    return pdf_info 



def list_files_in_folder(folder_path):
    file_list = []
    list_js = []
    for root, dirs, files in os.walk(folder_path):
        count_file = 1
        for file in files:
            print(count_file)
            count_file += 1
            file_path = os.path.join(root, file)
            file_list.append(file_path)
            try:
                tmp_js = pdf_analysis(file_path)

                output = pdfinfo_to_json(file_path)

                output['header_length'] = len(file)
                if tmp_js[0] <= 14:
                    output['small_content'] = 1
                else:
                    output['small_content'] = 0
                count = 0
                arr = [10, 11, 12, 13, 17]
                for index in arr:

                    if tmp_js[index] >= 1:

                        count += 1
                if count >= 2:
                    output['malicecontent'] = 1
                else:
                    output['malicecontent'] = 0
                #label
                output['label'] = 1
                list_js.append(output)
            except Exception as e:
                print("Error:", e)
                # Xóa file bị lỗi
                os.remove(file_path)
    return list_js


def json_to_csv(json_file, csv_file):
    # Đọc file JSON vào DataFrame
    df = pd.read_json(json_file)

    # Chuyển đổi DataFrame thành file CSV
    df.to_csv(csv_file, index=False)
# Thử nghiệm chuyển đổi
folder_path = "./test_folder/Malicious/malicious"
list_js = list_files_in_folder(folder_path)

output_file = "pdfinfo-M-Malicious.json"
with open(output_file, 'w') as f:
    json.dump(list_js, f, indent=4)

csv_file = 'output-pdfinfo-M-MALICIOUS.csv'
json_to_csv(output_file, csv_file)
print("Successfully saved list_js to", output_file)

	
