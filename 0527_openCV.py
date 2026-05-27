import os
import shutil
import cv2
from deepface import DeepFace

# 載入 Haar Cascade 人臉模型
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================================
# 1. 資料夾路徑設定
# ==========================================
source_base = "data"
sub_folders = ["Sad", "Angry", "Happy"]
target_folder = "face_data"
output_folder = "face_data_ok"

# 建立相關資料夾
os.makedirs(target_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 支援的圖片格式
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

print("--- 階段一：將三個資料夾整合至 face_data ---")

# ==========================================
# 2. 整合資料夾（加上前綴防檔名重複）
# ==========================================
for folder in sub_folders:
    folder_path = os.path.join(source_base, folder)

    if not os.path.exists(folder_path):
        print(f"找不到資料夾: {folder_path}，跳過。")
        continue

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(image_extensions):
            continue

        source_file = os.path.join(folder_path, filename)

        if os.path.isfile(source_file):
            # 檔名變成：資料夾名_原檔名.jpg (例如 Sad_001.jpg)
            new_filename = f"{folder}_{filename}"
            target_file = os.path.join(target_folder, new_filename)
            shutil.copy(source_file, target_file)

print(f"資料夾整合完成！所有圖片已集中至: {target_folder}\n")

print("--- 階段二：開始人臉偵測與情緒分析（舊版相容） ---")

# ==========================================
# 3. 讀取 face_data 圖片並進行臉部分析
# ==========================================
for filename in os.listdir(target_folder):
    if not filename.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(target_folder, filename)
    print(f"Processing: {filename}")

    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 OpenCV Haar Cascade 偵測人臉
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # 紀錄這張圖片是否有成功分析出人臉
    has_face_analyzed = False

    # 逐一處理偵測到的每張臉
    for (x, y, w, h) in faces:
        face_img = img[y:y + h, x:x + w]

        # 防止切下來的區塊不合法
        if face_img.size == 0:
            continue

        try:
            # 針對舊版 DeepFace 進行分析
            # 舊版回傳格式直接為 dict -> result['dominant_emotion']
            result = DeepFace.analyze(
                img_path=face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip"
            )

            # 舊版直接取字典欄位
            emotion = result["dominant_emotion"]

            # 標記畫框與文字 (綠色框)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            has_face_analyzed = True

        except Exception as e:
            print(f"Error analyzing face in {filename}: {e}")

    # ==========================================
    # 4. 只有成功辨識到臉部的圖片，才輸出到 face_data_ok
    # ==========================================
    if has_face_analyzed:
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        print(f"-> 成功導出至: {output_path}")
    else:
        print(f"-> 跳過 {filename} (未偵測到明確人臉)")

print("\n【全部處理完成】請至 face_data_ok 資料夾查看辨識成果！")