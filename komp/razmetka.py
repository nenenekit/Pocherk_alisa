import sys
import tempfile
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QRubberBand, QHBoxLayout, QMessageBox
)

# Словарь для транслитерации кириллицы
TRANSLIT_DICT = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '',
    'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
}


def sanitize_name(name: str) -> str:
    """Транслитерация и очистка имени файла"""
    translit = []
    for char in name.lower():
        if char in TRANSLIT_DICT:
            translit.append(TRANSLIT_DICT[char])
        elif char.isalnum():
            translit.append(char)
    return ''.join(translit)[:50]  # Ограничение длины имени


def generate_unique_filename(index, counter):
    """Генерация уникального имени файла с использованием временной метки."""
    timestamp = int(time.time())  # Уникальная временная метка
    return f"img_{timestamp}_{index:04d}_frag_{counter:04d}.png"


class ROIImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = None
        self.selection_callback = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, event):
        if self.origin:
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubberBand.setGeometry(rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.origin:
            self.rubberBand.hide()
            rect = QRect(self.origin, event.pos()).normalized()
            self.origin = None
            if self.selection_callback:
                self.selection_callback(rect)


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotation Tool")
        self.current_index = 0
        self.results = []
        self.fragment_counter = 0  # Счётчик фрагментов

        # Временная папка для страниц PDF
        self.temp_dir = Path(tempfile.mkdtemp())

        # Виджеты
        self.image_label = ROIImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.selection_callback = self.handle_roi

        self.info_label = QLabel("", self)
        self.entry_field = QLineEdit(self)
        self.add_button = QPushButton("Add Fragment", self)
        self.add_button.clicked.connect(self.confirm_fragment)
        self.add_button.setEnabled(False)
        self.next_button = QPushButton("Next Image", self)
        self.next_button.clicked.connect(self.next_image)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.add_button)
        btn_layout.addWidget(self.next_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.entry_field)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # Папки
        self.images_dir = Path('miss')
        self.crop_dir = Path('crops')
        self.crop_dir.mkdir(exist_ok=True)

        # Загрузка изображений
        self.images_list = self.load_images()

        # Восстанавливаем состояние, если есть
        try:
            with open('state.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Устанавливаем индекс в пределах границ списка изображений
                self.current_index = min(max(0, data['index']), len(self.images_list)-1)
                self.results = data['results']
        except FileNotFoundError:
            pass  # Начинаем с первого изображения

        if self.images_list:
            self.display_image()
        else:
            self.info_label.setText("Нет изображений для обработки.")

    def load_images(self):
        """Загрузка и обработка изображений"""
        images = []
        for f in sorted(self.images_dir.iterdir()):
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                safe_name = sanitize_name(f.name)
                tmp_path = self.temp_dir / f"{safe_name}.png"
                Image.open(f).save(tmp_path)
                images.append((tmp_path, f.name))
            elif f.suffix.lower() == '.pdf':
                doc = fitz.open(str(f))
                base_name = sanitize_name(f.name)
                for i, page in enumerate(doc):
                    pix = page.get_pixmap()
                    tmp_path = self.temp_dir / f"{base_name}_{i+1}.png"
                    pix.save(tmp_path)
                    images.append((tmp_path, f"{base_name}_{i+1}.png"))
        return images

    def load_image_cv(self, path: Path):
        """Загрузка изображения в формате OpenCV"""
        pil = Image.open(path)
        arr = np.array(pil.convert('RGB'))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def display_image(self):
        """Отображение текущего изображения"""
        if not self.images_list:
            self.info_label.setText("Нет изображений для обработки.")
            return

        img_path, label = self.images_list[self.current_index]
        try:
            self.orig_cv_img = self.load_image_cv(img_path)
        except Exception as e:
            self.info_label.setText(f"Ошибка загрузки изображения {label}: {e}")
            return

        h, w = self.orig_cv_img.shape[:2]
        self.orig_size = (w, h)

        pixmap = QPixmap(str(img_path)).scaledToWidth(500, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        self.info_label.setText(
            f"Изображение {self.current_index + 1}/{len(self.images_list)}: {label} | Размер: {w}x{h}"
        )
        self.entry_field.clear()
        self.current_roi = None

    def handle_roi(self, rect):
        """Обработка выделенной области"""
        disp_w = self.image_label.pixmap().width()
        disp_h = self.image_label.pixmap().height()
        scale_x = self.orig_size[0] / disp_w
        scale_y = self.orig_size[1] / disp_h

        # Получаем абсолютные координаты выделения
        x = int(rect.x() * scale_x)
        y = int(rect.y() * scale_y)
        w = int(rect.width() * scale_x)
        h = int(rect.height() * scale_y)

        # Создаем уникальное имя файла
        crop_fname = generate_unique_filename(self.current_index, self.fragment_counter)

        # Сохраняем вырезанный фрагмент
        crop = self.orig_cv_img[y:y+h, x:x+w]
        crop_path = self.crop_dir / crop_fname
        cv2.imwrite(str(crop_path), crop)

        self.current_roi = crop_fname
        self.fragment_counter += 1
        self.add_button.setEnabled(True)
        self.info_label.setText(
            f"Выделенная область: {crop_fname}. Введите текст и нажмите 'Add Fragment'"
        )

    def confirm_fragment(self):
        """Сохранение данных фрагмента"""
        text = self.entry_field.text().strip()
        if not self.current_roi or not text:
            self.info_label.setText("Выберите фрагмент и введите текст.")
            return

        self.results.append(f"{self.current_roi}\t{text}")
        self.current_roi = None
        self.entry_field.clear()
        self.add_button.setEnabled(False)
        self.info_label.setText("Фрагмент успешно сохранён.")

    def next_image(self):
        """Переход к следующему изображению"""
        self.current_index += 1
        self.fragment_counter = 0  # Сбрасываем счётчик фрагментов

        if self.current_index < len(self.images_list):
            self.display_image()
        else:
            # Сохраняем текущее состояние
            state_data = {'index': self.current_index, 'results': self.results}
            with open('state.json', 'w', encoding='utf-8') as f:
                json.dump(state_data, f)

            # Сохраняем результаты разметки
            with open('tetradi1.tsv', 'w', encoding='utf-8') as f:
                f.write("filename\ttext\n")
                for line in self.results:
                    f.write(f"{line}\n")

            QMessageBox.information(self, "Готово", "Разметка выполнена успешно!")
            QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageProcessor()
    win.show()
    sys.exit(app.exec_())