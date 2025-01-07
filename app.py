import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from distfit import distfit
import io

# Hàm fitting cho việc vẽ các phân phối
def fitting(sample: np.array, chart='pdf', n_top=3, figsize=(8,6), fontsize=8, emp_properties=None, cii_properties=None):
    dfit = distfit(alpha=0.01)
    dfit.fit_transform(sample)
    fig, _ = dfit.plot(chart=chart, n_top=n_top, figsize=figsize, fontsize=fontsize, emp_properties=emp_properties, cii_properties=cii_properties)
    return dfit, fig

# Hàm tạo dữ liệu demo
def load_demo_data():
    # Tạo các mẫu có kích thước khác nhau
    sample1 = np.random.normal(loc=0, scale=1, size=250)
    sample2 = np.random.normal(loc=2, scale=1.5, size=100)
    sample3 = np.random.normal(loc=1, scale=1.5, size=150)
    
    # Chọn số lượng phần tử lớn nhất (size max)
    max_size = max(len(sample1), len(sample2), len(sample3))
    
    # Điền dữ liệu thiếu (padding) bằng NaN để các mẫu có kích thước bằng max_size
    sample1_padded = np.pad(sample1, (0, max_size - len(sample1)), constant_values=np.nan)
    sample2_padded = np.pad(sample2, (0, max_size - len(sample2)), constant_values=np.nan)
    sample3_padded = np.pad(sample3, (0, max_size - len(sample3)), constant_values=np.nan)
    
    # Tạo DataFrame từ các mẫu đã điền dữ liệu thiếu
    data = {
        'Sample1': sample1_padded,
        'Sample2': sample2_padded,
        'Sample3': sample3_padded
    }
    return pd.DataFrame(data)

# Tạo giao diện cho người dùng chọn dữ liệu
st.title('Tìm Bestfitting và Cộng Distribution')

data_option = st.selectbox('Chọn dữ liệu:', ['Tải lên dữ liệu CSV', 'Dữ liệu mẫu ngẫu nhiên (size khác nhau)'])
# Khởi tạo biến df là None
df = None

if data_option == 'Tải lên dữ liệu CSV':
    uploaded_file = st.file_uploader("Tải lên file CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader('Dữ liệu CSV đã tải lên:')
        st.write(df)
    else:
        st.warning("Vui lòng tải lên một file CSV.")
else:
    # Nếu chọn dữ liệu mẫu, sử dụng dữ liệu demo đã tạo
    df = load_demo_data()
    st.subheader('Dữ liệu mẫu:')
    st.write(df)

# Kiểm tra xem df có tồn tại hay không trước khi tiếp tục
if df is not None:
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Chọn các cột để vẽ Histogram và Fitting", columns)

    # Thêm tùy chọn `n_top` để thay đổi số lượng phân phối tốt nhất được hiển thị
    n_top = st.slider("Chọn số lượng phân phối tốt nhất (n_top)", min_value=1, max_value=10, value=3)

    # Tùy chọn cho n khi tạo mẫu mới
    n_samples = st.slider("Chọn số lượng mẫu tạo ra từ phân phối (n)", min_value=100, max_value=5000, value=10000)

    # Kiểm tra xem người dùng đã chọn cột chưa
    if len(selected_columns) > 0:
        st.subheader('Biểu đồ Histogram và Fitting của các cột đã chọn')

        # Tạo cột để hiển thị biểu đồ cạnh nhau
        num_charts = len(selected_columns)
        columns_for_charts = st.columns(num_charts)

        # Khởi tạo danh sách để lưu mẫu tạo ra
        generated_samples = []

        # Vẽ histogram và fitting cho mỗi cột đã chọn
        for i, col in enumerate(selected_columns):
            sample = df[col].dropna().values  # Lấy mẫu không có giá trị NaN
            
            # Tùy chọn màu cho biểu đồ (mặc định là màu '#C41E3A')
            color = st.color_picker(f"Chọn màu cho cột {col}", '#607B8B')  # Màu mặc định là '#C41E3A'
            
            # Tùy chọn bin size cho biểu đồ
            bin_size = st.slider(f"Chọn bin size cho cột {col}", min_value=5, max_value=50, value=30, step=5)  # Mặc định là 30
            
            # Tùy chọn màu viền cho histogram
            edge_color = st.color_picker(f"Chọn màu viền cho cột {col}", '#5A5A5A')  # Màu viền mặc định là đen

            # Vẽ biểu đồ histogram
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            ax_hist.hist(sample, bins=bin_size, alpha=0.6, color=color, edgecolor=edge_color, label=col)
            ax_hist.set_title(f'Histogram của cột: {col}')
            ax_hist.set_xlabel('Giá trị')
            ax_hist.set_ylabel('Tần suất')
            ax_hist.legend()

            # Vẽ biểu đồ fitting với tham số `n_top` từ lựa chọn của người dùng
            dfit, fig_fitting = fitting(sample, chart='pdf', n_top=n_top, figsize=(8,6), fontsize=8)
            
            # Hiển thị các biểu đồ vào mỗi cột
            columns_for_charts[i].pyplot(fig_hist)
            columns_for_charts[i].pyplot(fig_fitting)

            # Tạo mẫu mới từ phân phối đã ước lượng (1000 mẫu)
            generated_sample = dfit.generate(n=n_samples)  # Sử dụng giá trị n đã chọn từ người dùng
            
            # Giới hạn giá trị của mẫu tạo ra trong khoảng min và max của sample gốc
            generated_sample = np.clip(generated_sample, sample.min(), sample.max())

            # Lưu mẫu tạo ra vào danh sách
            generated_samples.append(generated_sample)

            # Vẽ histogram của các mẫu được tạo ra
            fig_generated_hist, ax_generated_hist = plt.subplots(figsize=(6, 4))
            ax_generated_hist.hist(generated_sample, bins=bin_size, alpha=0.6, color='#607B8B', edgecolor=edge_color, label=f"Generated {col}")
            ax_generated_hist.set_title(f'Histogram của mẫu được tạo ra từ {col}')
            ax_generated_hist.set_xlabel('Giá trị')
            ax_generated_hist.set_ylabel('Tần suất')
            ax_generated_hist.legend()

            # Hiển thị biểu đồ histogram mẫu tạo ra
            columns_for_charts[i].pyplot(fig_generated_hist)

        # Tính tổng của hai mẫu tạo ra
        if len(generated_samples) >= 2:
            total_generated_sample = np.concatenate(generated_samples)
            # Tùy chọn chỉnh bin size cho histogram tổng hợp
            bin_size_total = st.slider("Chọn bin size cho histogram tổng hợp", min_value=5, max_value=50, value=30, step=5)

            # Tùy chọn DPI khi lưu hình ảnh
            dpi = st.slider("Chọn độ phân giải DPI khi lưu hình", min_value=50, max_value=300, value=150, step=50)

            # Tính P10, P50 (median), P90 của tổng mẫu tạo ra
            p10 = np.percentile(total_generated_sample, 10)
            p50 = np.percentile(total_generated_sample, 50)
            p90 = np.percentile(total_generated_sample, 90)

            # Vẽ histogram tổng hợp của hai mẫu đã tạo ra với bin size người dùng chọn
            fig_total_hist, ax_total_hist = plt.subplots(figsize=(6, 4))
            ax_total_hist.hist(total_generated_sample, bins=bin_size_total, alpha=0.6, color='#607B8B', edgecolor=edge_color, label='Tổng mẫu tạo ra')
            
            # Thêm các chỉ số P10, P50, P90 vào biểu đồ
            ax_total_hist.axvline(p10, color='g', linestyle='dashed', linewidth=2, label=f'P10 = {p10:.2f}')
            ax_total_hist.axvline(p50, color='b', linestyle='dashed', linewidth=2, label=f'P50 (Median) = {p50:.2f}')
            ax_total_hist.axvline(p90, color='r', linestyle='dashed', linewidth=2, label=f'P90 = {p90:.2f}')

            # Thêm legend cho P10, P50, P90
            ax_total_hist.legend()

            # Hiển thị biểu đồ tổng hợp với các chỉ số P10, P50, P90
            st.subheader('Histogram tổng hợp của hai mẫu tạo ra với P10, P50, P90')
            st.pyplot(fig_total_hist)

            # Lưu hình ảnh biểu đồ tổng hợp vào file PNG với DPI do người dùng chọn
            img_bytes = io.BytesIO()
            fig_total_hist.savefig(img_bytes, format='png', dpi=dpi)  # Sử dụng DPI người dùng chọn
            img_bytes.seek(0)
            st.download_button(label="Tải hình ảnh Histogram tổng hợp", data=img_bytes, file_name="total_histogram.png", mime="image/png")

            # Lưu dữ liệu vào file CSV
            total_sample_df = pd.DataFrame(total_generated_sample, columns=["Generated Sample"])
            csv_bytes = total_sample_df.to_csv(index=False).encode()
            st.download_button(label="Tải dữ liệu tổng hợp", data=csv_bytes, file_name="generated_samples.csv", mime="text/csv")
else:
    st.error("Vui lòng tải lên dữ liệu hoặc chọn dữ liệu mẫu.")
