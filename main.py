from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import os
import requests
import cairosvg
from sqlalchemy import select
from sqlalchemy.orm import Session

# Импорт ORM-моделей и функций из database.py
from database import (
    SessionLocal,
    Slide,
    SlideImage,
    SlideImageColor,
    Presentation,
    TemplateColor,
    select_templates_by_color
)
# Импорт функций для обработки изображений
from image_processing import get_dominant_colors_from_bytes, generate_image_hash
import cairosvg

# Инициализация FastAPI приложения
app = FastAPI()
UPLOAD_DIR = "./static/editor_images"  # папка для хранения загруженных изображений
# Создаём папку, если её нет
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Подключаем статику и шаблоны
app.mount("/static", StaticFiles(directory="static"), name="static")
html = Jinja2Templates(directory="html")

# Разрешённые расширения для изображений
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.svg'}

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    # Рендерим HTML из файла html/main.html, передавая request
    return html.TemplateResponse("main.html", {"request": request})

@app.post("/upload/")
async def upload_images(
    files: List[UploadFile] = File(default=[]),
    urls: Optional[str] = Form(default=None),
    id_slide: int = Form(...),
):
    created = []
    session: Session = SessionLocal()
    try:
        # 1) Обрабатываем загруженные файлы
        for file in files:
            if not file.filename:
                continue
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue  # пропускаем неподдерживаемые
            data = await file.read()
            if not data:
                continue
            fname = generate_image_hash(data) + ext
            path = f"/static/editor_images/{fname}"
            with open(os.path.join(UPLOAD_DIR, fname), "wb") as f:
                f.write(data)
            rec = SlideImage(
                id_slide=id_slide,
                image_path=path,
                type=1,
                ai=False,
                uploaded=True
            )
            session.add(rec)
            session.flush()
            slide = session.get(Slide, id_slide)
            created.append({
                "slide_image_id": rec.id,
                "id_slide": id_slide,
                "id_presentation": slide.id_presentation if slide else None
            })

        # 2) Обрабатываем URL
        if urls:
            for url in [u.strip() for u in urls.split(",") if u.strip()]:
                ext = os.path.splitext(url)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue
                rec = SlideImage(
                    id_slide=id_slide,
                    image_path=url,
                    type=1,
                    ai=False,
                    uploaded=False
                )
                session.add(rec)
                session.flush()
                slide = session.get(Slide, id_slide)
                created.append({
                    "slide_image_id": rec.id,
                    "id_slide": id_slide,
                    "id_presentation": slide.id_presentation if slide else None
                })
        session.commit()
    finally:
        session.close()

    if not created:
        raise HTTPException(400, "Нечего добавить или формат файла не поддерживается")
    return JSONResponse({"created": created})

@app.post("/delete/")
def delete_image(
    slide_image_id: int = Form(...)
):
    session: Session = SessionLocal()
    try:
        rec = session.get(SlideImage, slide_image_id)
        if not rec:
            raise HTTPException(404, "Изображение не найдено")

        file_rel_path = rec.image_path.lstrip("/")  # например, "static/editor_images/xxx.png"
        file_abs_path = os.path.join(os.getcwd(), file_rel_path)
        if os.path.exists(file_abs_path):
            try:
                os.remove(file_abs_path)
            except OSError as e:
                # Если файл занят/недоступен — логируем
                print(f"Ошибка удаления файла {file_abs_path}: {e}")

        session.delete(rec)
        session.commit()
    finally:
        session.close()
    return JSONResponse({"detail": "Удалено"})

@app.get("/pick_templates/")
def pick_templates(
    id_presentation: int = Query(...),
    dominant_colors_count: int = Query(3, ge=1, le=10),
    palette_size: int = Query(5, ge=1, le=20),
    temperature: float = Query(1.0, ge=0),
):
    session: Session = SessionLocal()
    try:
        slide_ids = session.scalars(
            select(Slide.id).where(Slide.id_presentation == id_presentation)
        ).all()
        if not slide_ids:
            raise HTTPException(404, "Презентация не найдена или нет слайдов")

        images = session.scalars(
            select(SlideImage).where(SlideImage.id_slide.in_(slide_ids))
        ).all()
        if not images:
            raise HTTPException(400, "Нет изображений для этой презентации")

        all_colors: List[tuple[int, int, int]] = []

        for img in images:
            # получаем уже сохранённые доминантные цвета
            image_colors = session.scalars(
                select(SlideImageColor.rgb)
                .where(SlideImageColor.slide_image_id == img.id)
                .order_by(SlideImageColor.color_index)
            ).all()

            if not image_colors:
                try:
                    if img.image_path.startswith("http"):
                        resp = requests.get(img.image_path, timeout=5)
                        resp.raise_for_status()
                        data = resp.content
                    else:
                        with open(img.image_path.lstrip("/"), "rb") as f:
                            data = f.read()
                except Exception:
                    continue

                ext = os.path.splitext(img.image_path)[1].lower()
                if ext == '.svg':
                    try:
                        data = cairosvg.svg2png(bytestring=data, background_color=None)
                    except Exception:
                        continue

                try:
                    cols = get_dominant_colors_from_bytes(
                        data,
                        k=dominant_colors_count,
                        num_colors=dominant_colors_count
                    )
                except Exception:
                    continue

                for idx, (r, g, b) in enumerate(cols):
                    session.add(
                        SlideImageColor(
                            slide_image_id=img.id,
                            color_index=idx,
                            rgb=[r, g, b]
                        )
                    )
                session.commit()
                image_colors = session.scalars(
                    select(SlideImageColor.rgb)
                    .where(SlideImageColor.slide_image_id == img.id)
                    .order_by(SlideImageColor.color_index)
                ).all()

            # добавляем к общему списку
            for rgb in image_colors:
                all_colors.append(tuple(rgb))

        if not all_colors:
            raise HTTPException(400, "Нет доминантных цветов для этой презентации")

        templates = select_templates_by_color(
            target_colors=all_colors,
            palette_size=palette_size,
            count=5,
            temperature=temperature
        )

        result = []
        for tpl, _ in templates:
            tpl_cols = session.scalars(
                select(TemplateColor.rgb)
                .where(TemplateColor.presentation_id == tpl.id)
                .order_by(TemplateColor.color_index)
            ).all()
            result.append({
                "id": tpl.id,
                "name": tpl.name,
                "template_colors": [tuple(c) for c in tpl_cols]
            })
    finally:
        session.close()

    return JSONResponse({
        "used_colors": all_colors,
        "recommended_templates": result
    })