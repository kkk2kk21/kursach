from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from io import BytesIO
import hashlib
from typing import List, Tuple

# Функции для работы с изображениями

def adaptive_resize(image: Image.Image, max_size: int = 300) -> Image.Image:
    # Подгоняет размер изображения так, чтобы ни одна сторона не превышала max_size пикселей,
    # сохраняя пропорции для ускорения последующей обработки

    width, height = image.size  # текущие размеры

    # если сторона больше max_size, масштабируем
    if max(width, height) > max_size:
        scale = max_size / float(max(width, height))
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height))
    # иначе возвращаем оригинал
    return image


def get_dominant_colors(
    image: Image.Image,
    k: int = 4,
    num_colors: int = 3
) -> List[Tuple[int,int,int]]:
    # Вычисляет наиболее доминантные цвета в изображении с помощью K-Means
    # k: число кластеров K-Means (чем больше, тем больше вариантов цветовых групп)
    # num_colors: сколько топ-цветов вернуть (от 1 до k)
    # Возвращает список rgb кортежей

    image = image.convert("RGB")
    # Уменьшаем размер для ускорения (максимум 300px по стороне)
    image = adaptive_resize(image, max_size=300)
    # Дополнительно создаём мини-версию (100x100) для быстрого доступа к пикселям
    image.thumbnail((100, 100))

    # Переводим в numpy-массив размера (num_pixels, 3)
    arr = np.array(image).reshape(-1, 3)

    # Применяем K-Means, чтобы сгруппировать пиксели по цвету
    kmeans = KMeans(n_clusters=k, random_state=0).fit(arr)
    # Считаем, сколько пикселей попало в каждый кластер
    counts = np.bincount(kmeans.labels_)
    # Индексы самых популярных кластеров (по убыванию)
    top_idxs = counts.argsort()[::-1][:num_colors]

    # Центры выбранных кластеров и есть доминантные цвета
    colors = []
    for idx in top_idxs:
        center = kmeans.cluster_centers_[idx]
        # Приводим координаты к целым и в кортеж
        colors.append((int(center[0]), int(center[1]), int(center[2])))
    return colors


def get_dominant_colors_from_bytes(
    image_bytes: bytes,
    k: int = 4,
    num_colors: int = 3
) -> List[Tuple[int,int,int]]:
    # Обёртка для работы с байтами
    # Разбирает байты в изображение и вызывает get_dominant_colors

    # Открываем изображение из байтового потока
    image = Image.open(BytesIO(image_bytes))
    return get_dominant_colors(image, k=k, num_colors=num_colors)


def generate_image_hash(image_bytes: bytes) -> str:
    # Генерирует уникальное название файла на основе хэша
    return hashlib.sha256(image_bytes).hexdigest()
