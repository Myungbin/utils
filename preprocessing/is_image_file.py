# ==================================
# 이미지로 인식되지 않은 이미지 삭제
# ==================================
from PIL import Image
import os


def is_image_file(file_name):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    return file_name.lower().endswith(image_extensions)


def open_and_delete_invalid_images_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and is_image_file(file_name):
            try:
                with Image.open(file_path):
                    print(f"{file_path} 이미지가 열렸습니다.")
            except Exception as e:
                print(f"{file_path} 이미지를 열 수 없습니다. 예외: {e}")
                os.remove(file_path)
                print(f"{file_path} 이미지를 삭제했습니다.")


folder_path = r"C:\MB_Project\data\Malware\Test"
open_and_delete_invalid_images_in_folder(folder_path)
