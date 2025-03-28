from pdfid.pdfid import PDFiD, PDFiD2JSON
import json
import os
import pandas as pd
import numpy as np
def pdf_analysis(filename):

    xml = PDFiD(filename)
    doc = PDFiD2JSON(xml, False)

    y = json.loads(doc)
    js = json.loads(json.dumps(y[0]))
    

    l = []
    for w in js["pdfid"]["keywords"]["keyword"]:
        l.append(w["count"])

    return l

def dir_analysis(dirname):
    mat = []
    count_file = 1
    for filename in os.listdir(dirname):
        print(count_file)
        count_file += 1
        f = os.path.join(dirname, filename)
        if os.path.isfile(f):
            l = pdf_analysis(f)
            l.append(len(filename))
            if l[0] <= 14:
                l.append(1)
            else:
                l.append(0)
            count = 0
            arr = [10, 11, 12, 13, 17]
            for i in arr:
                if l[i] >= 1:
                    count += 1
            if count >= 2:
                l.append(1)
            else:
                l.append(0)
            #label
            l.append(1)
            mat.append(l)
            #print("..." + str(i))
            #i += 1
            
    return mat
    
def extract():
    path = "../test_folder/Malicious/malicious"
 
    #print(dir_analysis(path))
    df = pd.DataFrame(data=dir_analysis(path), columns=["obj", "endobj", "stream", "endstream", "xref", "trailer", "startxref", "/Page", "/Encrypt", "/ObjStm", "/JS", "/JavaScript", "/AA", "/OpenAction", "/AcroForm", "/JBIG2Decode", "/RichMedia", "/Launch", "/EmbeddedFile", "/XFA", "/Colors > 2^24", "header_length", "small_content", "malicecontent", "label"])

    file_name = "ouput-pdfid-M-MALICIOUS.csv"
    df.to_csv(file_name, index = False)
    
if __name__ == "__main__":
    extract()
