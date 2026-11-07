import subprocess, sys
pdf_path = r"C:\Users\Asif Ali\Desktop\Dip_project\Satellite_Image_Change_Detection_DIP_Project_Proposal.pdf"
try:
    from PyPDF2 import PdfReader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    from PyPDF2 import PdfReader

reader = PdfReader(pdf_path)
full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

with open(r"C:\Users\Asif Ali\Desktop\Dip_project\extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Extraction complete. Text saved to extracted_text.txt")