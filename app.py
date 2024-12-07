import os
import time
import streamlit as st
from PIL import Image,ImageEnhance 
from streamlit_drawable_canvas import st_canvas
from sympy import sympify, parse_expr, lambdify
from sympy.parsing.latex import parse_latex
from scipy.optimize import fsolve
from rapid_latex_ocr import LaTeXOCR
from dotenv import load_dotenv
import io
import logging
import traceback
import numpy as np
from datetime import datetime
import cv2

# Gọi set_page_config
st.set_page_config(page_title="HNUE Latex OCR", layout="wide")

# Tải biến môi trường
load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Khởi tạo OCR
latex_ocr = LaTeXOCR()

# Hàm lưu lịch sử
def save_history(entry):
    try:
        os.makedirs('history', exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        history_file = f"history/history_{today}.txt"
        with open(history_file, "a", encoding='utf-8') as file:
            file.write(f"{datetime.now()} - {entry}\n")
    except Exception as e:
        logging.error(f"Error saving history: {str(e)}")

# Hàm xuất lịch sử
def export_history():
    try:
        history_files = sorted(
            [f for f in os.listdir('history') if f.startswith('history_')],
            reverse=True
        )
        if history_files:
            selected_file = st.selectbox("Select History File", history_files)
            with open(os.path.join('history', selected_file), 'r', encoding='utf-8') as f:
                history_content = f.read()
            st.download_button(
                label="Download History",
                data=history_content,
                file_name=selected_file,
                mime='text/plain'
            )
    except Exception as e:
        st.error(f"Error exporting history: {str(e)}")
        
        
# Hàm tiền xử lý ảnh
def preprocess_image(image):
    try:
        # Chuyển đổi PIL Image sang định dạng numpy
        image_array = np.array(image)
        
        # Chuyển đổi ảnh sang thang xám
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Cân bằng histogram
        equalized_image = cv2.equalizeHist(gray_image)
        
        # Điều chỉnh gamma
        def adjust_gamma(image, gamma=1.2):
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        
        gamma_corrected = adjust_gamma(equalized_image)
        
        # Chuyển về định dạng PIL Image
        processed_image = Image.fromarray(gamma_corrected)
        return processed_image
    except Exception as e:
        logging.error(f"Lỗi xử lý ảnh: {traceback.format_exc()}")
        return image  # Trả lại ảnh gốc nếu lỗi

# Hàm chuyển đổi ảnh sang LaTeX
def convert_image_to_latex(image_data):
    try:
        result, _ = latex_ocr(image_data)
        return result
    except Exception as e:
        logging.error(f"Lỗi OCR: {traceback.format_exc()}")
        return f"Error: {str(e)}"

# Hàm giải toán nâng cao
def solve_advanced_equation(latex_equation):
    try:
        # Parse LaTeX thành biểu thức SymPy
        equation = parse_latex(latex_equation)
        expr = equation.lhs - equation.rhs
        
        # Hàm lambdify để số hóa biểu thức
        func = lambdify('x', expr, 'numpy')
        
        # Dùng fsolve từ SciPy để giải nghiệm
        solution = fsolve(func, [0])  # Bắt đầu từ giá trị x=0
        return solution
    except Exception as e:
        logging.error(f"Lỗi giải toán: {traceback.format_exc()}")
        return f"Error: {str(e)}"

# Hàm giữ nguyên ảnh không bị cắt xén
def resize_image_maintain_aspect_ratio(image, max_width=1200):
    """Resize ảnh giữ nguyên tỷ lệ với chiều rộng tối đa"""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

# Hàm cắt vùng ảnh
def crop_image(image_array, bbox):
    x, y, width, height = bbox
    cropped_image = image_array[y:y + height, x:x + width]
    return cropped_image

# UI: Chuyển đổi LaTeX
def convert_latex_ui():
    st.header("📄 Chuyển Ảnh Sang LaTeX")
    uploaded_file = st.file_uploader("Tải ảnh toán học", type=["png", "jpg", "jpeg"], help="Hỗ trợ ảnh chứa công thức toán học")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image = resize_image_maintain_aspect_ratio(image)
        # Tiền xử lý ảnh
        with st.spinner("Đang tiền xử lý ảnh..."):
            image = preprocess_image(image)
            
        # Tạo container để kiểm soát chiều rộng hiển thị
        col1, col2 = st.columns([2, 1])
        with col1:
                st.image(image, caption="Ảnh đã tải", use_column_width=True)
        #Chuyển đổi ảnh sang array
        image_array = np.array(image)
            
        st.write("🔍 Chọn vùng cần xử lý:")
        canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                background_image=image,
                update_streamlit=True,
                height=image.height,
                width=image.width,
                drawing_mode="rect",
                key="canvas",
                display_toolbar=True
            )

        if canvas_result.json_data is not None:
                for rect in canvas_result.json_data["objects"]:
                    if rect["type"] == "rect":
                        x = int(rect["left"])
                        y = int(rect["top"])
                        width = int(rect["width"])
                        height = int(rect["height"])
                        cropped = crop_image(image_array, (x, y, width, height))
                        st.image(cropped, caption="Vùng được chọn", use_column_width=True)

                        with st.spinner('Đang chuyển đổi LaTeX...'):
                            buffered = io.BytesIO()
                            cropped_image = Image.fromarray(cropped)
                            cropped_image.save(buffered, format="PNG")
                            latex_result = convert_image_to_latex(buffered.getvalue())

                            if "Error" not in latex_result:
                                st.success("Chuyển đổi LaTeX thành công:")
                                st.code(latex_result, language='latex')
                                save_history(f"LaTeX chuyển đổi: {latex_result}")
                                
def solve_math_ui():
    st.header("🔢 Giải Toán Từ LaTeX")
    latex_equation = st.text_input("Nhập công thức LaTeX cần giải:", help="Ví dụ: x^2 + 3*x - 4 = 0")
    
    if st.button("Giải"):
        with st.spinner("Đang giải bài toán..."):
            if latex_equation:
                solution = solve_advanced_equation(latex_equation)
                if "Error" not in str(solution):
                    st.success("Lời giải:")
                    st.write(solution)
                    save_history(f"Lời giải: {solution}")
                else:
                    st.error("Không thể giải bài toán.")
            else:
                st.warning("Vui lòng nhập công thức.")

# UI: Lịch sử
def history_ui():
    st.header("📜 Lịch Sử Hoạt Động")
    export_history()

#Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    /* Thay đổi toàn bộ nền */
    body {
        background-color: #F0F8FF;
        font-family: Arial, sans-serif;
    }

    /* Header */
    .css-18e3th9 {
        background-color: #4CAF50 !important;
        color: white !important;
        text-align: center;
        font-size: 24px;
        border-radius: 8px;
        padding: 10px;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #FFFFFF !important;
        border-radius: 10px;
        padding: 10px;
    }

    /* Nút bấm */
    button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-size: 16px;
        padding: 8px 12px;
        border: none;
        transition: 0.3s;
    }

    button:hover {
        background-color: #45A049;
        cursor: pointer;
    }

    /* Input box */
    .stTextInput input {
        background-color: #FFFFFF;
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 8px;
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# Main
def main():
    apply_custom_css()
    st.title("📘 HNUE LATEX OCR")
    st.sidebar.header("🔧 Tính năng")
    feature = st.sidebar.radio("Chọn chức năng", ["Chuyển Đổi LaTeX", "Giải Toán", "Lịch Sử"])

    if feature == "Chuyển Đổi LaTeX":
        convert_latex_ui()
    elif feature == "Giải Toán":
        solve_math_ui()
    elif feature == "Lịch Sử":
        history_ui()

if __name__ == "__main__":
    main()
