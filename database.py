from sqlalchemy.engine.url import URL
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, Sequence,
    ForeignKey, SmallInteger, select
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from math import sqrt
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from typing import List

# Базовый класс для описания моделей
Base = declarative_base()

class Presentation(Base):
    # Модель таблицы presentation.presentation
    __tablename__ = "presentation"
    __table_args__ = {"schema": "presentation"}

    # PK: идентификатор презентации
    id = Column(
        Integer,
        Sequence("presentation_id_seq", schema="presentation"),
        primary_key=True
    )
    # Название презентации
    name = Column(String)
    # Флаг шаблона
    template = Column(Boolean, default=False)

    # Связь к палитрам шаблона
    templates = relationship(
        "TemplateColor",
        back_populates="presentation",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class TemplateColor(Base):
    # Модель таблицы presentation.template_color для палитры шаблона
    __tablename__ = "template_color"
    __table_args__ = {"schema": "presentation"}

    # PK: уникальный идентификатор палитры
    template_color_id = Column(Integer, primary_key=True)
    # FK на Presentation с каскадом
    presentation_id = Column(
        Integer,
        ForeignKey(
            "presentation.presentation.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )
    # Позиция цвета в палитре
    color_index = Column(SmallInteger, nullable=False)
    # Цвет [R,G,B]
    rgb = Column(ARRAY(SmallInteger), nullable=False)

    # Обратная связь к презентации
    presentation = relationship(
        "Presentation",
        back_populates="templates"
    )

class Slide(Base):
    # Модель таблицы presentation.slide
    __tablename__ = "slide"
    __table_args__ = {"schema": "presentation"}

    # PK: уникальный идентификатор слайда
    id = Column(
        Integer,
        Sequence("slide_id_seq", schema="presentation"),
        primary_key=True
    )
    # FK на презентацию с каскадом
    id_presentation = Column(
        Integer,
        ForeignKey(
            "presentation.presentation.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )
    # Связь: один слайд может иметь много изображений, с каскадом удаления
    images = relationship(
        "SlideImage",
        back_populates="slide",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class SlideImage(Base):
    # Модель таблицы presentation.slide_image
    __tablename__ = "slide_image"
    __table_args__ = {"schema": "presentation"}

    # PK: уникальный идентификатор записи изображения
    id = Column(
        Integer,
        Sequence("slide_image_id_seq", schema="presentation"),
        primary_key=True
    )
    # FK на слайд с каскадом
    id_slide = Column(
        Integer,
        ForeignKey(
            "presentation.slide.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )
    # Путь к файлу или URL изображения
    image_path = Column(String, nullable=False)
    # Тип изображения
    type = Column(Integer, default=1)
    # Флаг: сгенерировано AI
    ai = Column(Boolean, default=False)
    # Флаг: загружено пользователем
    uploaded = Column(Boolean, default=False)

    # Обратная связь к слайду
    slide = relationship(
        "Slide",
        back_populates="images"
    )
    # Связь: доминантные цвета, каскадное удаление
    colors = relationship(
        "SlideImageColor",
        back_populates="slide_image",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class SlideImageColor(Base):
    # Модель таблицы presentation.slide_image_color для хранения доминантных цветов
    __tablename__ = "slide_image_color"
    __table_args__ = {"schema": "presentation"}

    # PK: уникальный идентификатор записи цветового кластера
    slide_image_color_id = Column(Integer, primary_key=True)
    # FK на запись SlideImage с каскадом
    slide_image_id = Column(
        Integer,
        ForeignKey(
            "presentation.slide_image.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )
    # Индекс цвета в палитре изображения
    color_index = Column(SmallInteger, nullable=False)
    # Сам цвет в формате [R,G,B]
    rgb = Column(ARRAY(SmallInteger), nullable=False)

    # Обратная связь к изображению
    slide_image = relationship(
        "SlideImage",
        back_populates="colors"
    )

# Настройка подключения к БД
ENGINE = create_engine(
    URL.create(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="78787878",
        host="localhost",
        port=5432,
        database="postgres"
    ),
    echo=True
)
SessionLocal = sessionmaker(
    bind=ENGINE,
    autoflush=False,
    autocommit=False
)

# Создаём таблицы (если отсутствуют)
Base.metadata.create_all(bind=ENGINE)

# Функция выбора шаблонов по цвету
# target_colors: список RGB-кортежей
# palette_size: кол-во кластеров для target_colors
# count: сколько шаблонов вернуть
# temperature: для случайного отбора

def select_templates_by_color(
    target_colors: List[tuple[int, int, int]],
    palette_size: int = 5,
    count: int = 5,
    temperature: float | None = None
):
    # Преобразуем в numpy-массив для кластеризации
    pts = np.array(target_colors, dtype=float)
    if pts.size == 0:
        return []
    # Число кластеров не больше числа точек
    k = min(palette_size, len(pts))
    # Строим палитру центров кластеров K-Means
    palette = KMeans(n_clusters=k, random_state=0).fit(pts).cluster_centers_

    # Евклидово расстояние между двумя векторами
    def euclid(a, b):
        return sqrt(((a - b) ** 2).sum())

    session = SessionLocal()
    # Берём все шаблоны
    all_tpls = session.query(Presentation).filter_by(template=True).all()
    scores = []

    for tpl in all_tpls:
        # Загружаем палитру шаблона
        cols = session.scalars(
            select(TemplateColor.rgb).where(TemplateColor.presentation_id == tpl.id)
        ).all()
        tpl_arr = np.array(cols, dtype=float)
        if tpl_arr.size == 0:
            continue

        # Метрика: среднеминимальное расстояние
        d_min = [min(euclid(p, q) for q in tpl_arr) for p in palette]
        avg_min = sum(d_min) / len(d_min)

        # Метрика: оптимальное паросочетание (венгерский алгоритм)
        cost = np.zeros((len(palette), len(tpl_arr)))
        for i, p in enumerate(palette):
            for j, q in enumerate(tpl_arr):
                cost[i, j] = euclid(p, q)
        ri, ci = linear_sum_assignment(cost)
        bip = cost[ri, ci].mean()

        # Итоговые очки = среднее двух метрик
        scores.append((tpl, (avg_min + bip) / 2))

    session.close()
    # Сортируем по возрастанию score
    scores.sort(key=lambda x: x[1])

    if temperature and temperature > 0:
        # Случайный отбор с температурой
        dists = np.array([s for _, s in scores])
        weights = np.exp(-dists / temperature)
        probs = weights / weights.sum()
        idxs = np.random.choice(
            len(scores), size=min(count, len(scores)), replace=False, p=probs
        )
        return [scores[i] for i in idxs]

    # Возвращаем топ count
    return scores[:count]
