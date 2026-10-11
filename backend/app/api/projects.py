from pathlib import Path
import shutil

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.config import settings
from app.core.errors import require_docx_upload, require_image_upload, raise_api_error
from app.models.project import Project, ProjectStatus
from app.models.task import DocumentTask, TaskStatus
from app.schemas.project import (
    ProjectUploadResponse,
    ProjectStatusResponse,
    ProjectDeleteResponse,
    ProjectAnalysisResponse,
    ProjectFileUploadResponse,
    ProjectProcessResponse,
    ProjectSuggestionsResponse,
    ApplySuggestionsBody,
    ApplySuggestionsResponse,
)
from app.services import orchestrator
from app.services import project_metadata as pm
from app.services.download_files import resolve_project_output
from app.services.image_suggestions_engine import build_image_suggestions_from_metadata
from app.services.ml_image_analysis import register_uploaded_image_in_metadata
from app.services.project_docx import resolve_primary_project_docx
from app.services.storage import (
    save_project_source_file,
    copy_project_file_to_task_input,
)
from app.services.task_pipeline import run_document_task_pipeline

router = APIRouter(prefix="/projects", tags=["projects"])


def _ensure_project_owner(project: Project, user_id: str, action: str) -> None:
    if not project.user_id:
        raise HTTPException(status_code=403, detail=f"У проекта не задан владелец. {action} запрещено.")
    if project.user_id != user_id:
        raise HTTPException(status_code=403, detail=f"Нельзя {action} для проекта другого пользователя.")


@router.post("/upload", response_model=ProjectUploadResponse)
def upload_project_file(
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),
    topic: str | None = Form(default=None, description="Тема работы для ML"),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise_api_error("invalid_file_format", "Имя файла отсутствует.")
    require_docx_upload(file.filename)

    project = Project(
        user_id=user_id,
        status=ProjectStatus.UPLOADED,
        source_filename=Path(file.filename).name,
        source_path="",
        metadata_path="",
        payload=None,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    try:
        source_path = save_project_source_file(project.id, file)
    except ValueError:
        project.status = ProjectStatus.ERROR
        db.commit()
        raise_api_error("invalid_file_format", "Некорректный формат файла. Поддерживается только DOCX.")

    meta = pm.empty(
        project.id,
        status=ProjectStatus.UPLOADED.value,
        topic=(topic or "").strip() or None,
        source_filename=project.source_filename,
    )
    pm.set_input_file(meta, source_path)
    metadata_path = pm.save(project, meta)

    project.source_path = source_path
    project.metadata_path = metadata_path
    project.status = ProjectStatus.UPLOADED
    db.commit()

    return ProjectUploadResponse(
        project_id=project.id,
        status=project.status.value,
        source_filename=project.source_filename,
    )


@router.get("/{project_id}", response_model=ProjectStatusResponse)
def get_project_status(project_id: str, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")

    return ProjectStatusResponse(
        project_id=project.id,
        user_id=project.user_id,
        status=project.status.value,
        source_filename=project.source_filename,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.post("/{project_id}/files", response_model=ProjectFileUploadResponse)
def upload_project_additional_file(
    project_id: str,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")
    _ensure_project_owner(project, user_id, "загружать файлы")
    if not file.filename:
        raise_api_error("invalid_file_format", "Имя файла отсутствует.")
    require_image_upload(file.filename)

    try:
        file_path = save_project_source_file(project_id, file)
    except ValueError:
        raise_api_error(
            "invalid_file_format",
            "Некорректный формат файла. Для доп. загрузки поддерживаются только PNG и JPG.",
        )

    path_obj = Path(file_path)
    file_type = "image" if path_obj.parent.name == "images" else "input"

    meta = pm.load(project)
    if file_type == "image":
        pm.append_image(meta, path_obj.name)
        register_uploaded_image_in_metadata(meta, project.id, path_obj)
    else:
        pm.set_input_file(meta, file_path)
    pm.save(project, meta)

    return ProjectFileUploadResponse(
        project_id=project.id,
        status=project.status.value,
        file_path=file_path,
        file_type=file_type,
    )


@router.post("/{project_id}/process", response_model=ProjectProcessResponse)
def process_project_document(
    project_id: str,
    user_id: str = Form(...),
    faculty: str = Form(...),
    department: str = Form(...),
    student_group: str = Form(...),
    lab_title: str = Form(...),
    lab_number: str = Form(...),
    student_name: str = Form(...),
    reviewer_name: str = Form(...),
    discipline: str = Form(...),
    topic: str | None = Form(default=None, description="Тема для ML (если не задана при upload)"),
    db: Session = Depends(get_db),
):
    """
    Связка doc-service: extract → apply_ml_changes → download PDF + DOCX.
    """
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")
    _ensure_project_owner(project, user_id, "запускать обработку")

    docx_path = resolve_primary_project_docx(project)
    if not docx_path:
        project.status = ProjectStatus.ERROR
        db.commit()
        meta = pm.load(project)
        pm.add_error(meta, "В проекте нет DOCX для обработки (нужен .docx в загрузке или в input/).")
        pm.save(project, meta)
        raise HTTPException(status_code=400, detail="В проекте нет DOCX для обработки.")

    payload = {
        "faculty": faculty,
        "department": department,
        "student_group": student_group,
        "lab_title": lab_title,
        "lab_number": lab_number,
        "student_name": student_name,
        "reviewer_name": reviewer_name,
        "discipline": discipline,
    }

    task = DocumentTask(
        user_id=user_id,
        project_id=project_id,
        original_filename=docx_path.name,
        input_path="",
        payload=payload,
        status=TaskStatus.CREATED,
        errors=[],
        warnings=[],
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    project.status = ProjectStatus.PROCESSING
    db.commit()

    input_path = copy_project_file_to_task_input(task.id, docx_path)

    if topic and topic.strip():
        meta = pm.load(project)
        meta["topic"] = topic.strip()
        pm.save(project, meta)

    try:
        result = run_document_task_pipeline(db, task, project, input_path)
    except HTTPException as exc:
        meta = pm.load(project)
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        pm.add_error(meta, detail)
        pm.save(project, meta)
        raise

    db.refresh(task)
    db.refresh(project)

    if result["type"] == "validation_failed":
        report = result["report"]
        db.refresh(project)
        return ProjectProcessResponse(
            project_id=project.id,
            task_id=task.id,
            status=project.status.value,
            report=report,
        )

    db.refresh(project)
    return ProjectProcessResponse(
        project_id=project.id,
        task_id=task.id,
        status=project.status.value,
        report=result["report"],
    )


@router.get("/{project_id}/suggestions", response_model=ProjectSuggestionsResponse)
def get_image_suggestions(
    project_id: str,
    user_id: str = Query(..., description="Идентификатор пользователя"),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")
    _ensure_project_owner(project, user_id, "смотреть подсказки")

    meta = pm.load(project)
    suggestions = build_image_suggestions_from_metadata(project.id, meta)
    meta["image_suggestions"] = suggestions
    pm.save(project, meta)

    return ProjectSuggestionsResponse(
        project_id=project.id,
        status=project.status.value,
        suggestions=suggestions,
    )


@router.post("/{project_id}/suggestions/apply", response_model=ApplySuggestionsResponse)
def apply_image_suggestions(
    project_id: str,
    body: ApplySuggestionsBody,
    user_id: str = Query(..., description="Идентификатор пользователя"),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")
    _ensure_project_owner(project, user_id, "применять подсказки")

    meta = pm.load(project)
    suggestions = meta.get("image_suggestions") or []
    applied: list[str] = []
    id_set = set(body.suggestion_ids)
    for item in suggestions:
        if item.get("id") in id_set:
            item["applied"] = True
            applied.append(item["id"])
    meta["image_suggestions"] = suggestions
    pm.save(project, meta)

    return ApplySuggestionsResponse(project_id=project.id, status=project.status.value, applied_ids=applied)


@router.get("/{project_id}/download")
def download_project_result(
    project_id: str,
    user_id: str = Query(..., description="Идентификатор пользователя"),
    format: str = Query(
        default='pdf',
        alias='format',
        description='Формат: pdf или docx',
    ),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")
    _ensure_project_owner(project, user_id, "скачать результат")

    fmt = format.lower().strip()
    path, media_type, filename = resolve_project_output(db, project_id, fmt)
    return FileResponse(str(path), media_type=media_type, filename=filename)


@router.delete("/{project_id}", response_model=ProjectDeleteResponse)
def delete_project(
    project_id: str,
    user_id: str = Query(..., description="Идентификатор пользователя"),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")
    _ensure_project_owner(project, user_id, "удалить проект")

    related_tasks = db.execute(select(DocumentTask).where(DocumentTask.project_id == project_id)).scalars().all()
    for task in related_tasks:
        for file_path in [task.input_path, task.output_path, task.report_path]:
            if file_path:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
        db.delete(task)

    project_dir = Path(settings.projects_dir) / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)

    db.delete(project)
    db.commit()
    return ProjectDeleteResponse(project_id=project_id, status="deleted")


@router.post("/{project_id}/analyze", response_model=ProjectAnalysisResponse)
def analyze_project(
    project_id: str,
    user_id: str = Query(..., description="Идентификатор пользователя"),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")
    _ensure_project_owner(project, user_id, "запускать анализ")

    try:
        docx_path = resolve_primary_project_docx(project)
        if not docx_path:
            raise HTTPException(status_code=400, detail="В проекте нет DOCX для анализа.")

        meta = pm.load(project)
        topic_value = orchestrator.topic_from_context(None, pm.topic_value(meta, project.source_filename))

        project.status = ProjectStatus.PROCESSING
        db.commit()
        raw = orchestrator.extract_via_doc_service(docx_path, project.id)
        preview = orchestrator.normalize_extract_for_frontend(raw)
        analysis_result = {
            "project_id": project.id,
            "topic": topic_value,
            "preview": preview,
            "note": "Превью текста через doc-service. Оформление — POST /projects/{id}/process.",
        }

        project.status = ProjectStatus.READY
        db.commit()
        meta["analysis"] = analysis_result
        meta["errors"] = []
        pm.save(project, meta)
        return ProjectAnalysisResponse(project_id=project.id, status=project.status.value, analysis=analysis_result)
    except Exception as exc:
        project.status = ProjectStatus.ERROR
        db.commit()
        meta = pm.load(project)
        pm.add_error(meta, str(exc))
        pm.save(project, meta)
        raise HTTPException(status_code=400, detail=f"Ошибка анализа: {str(exc)}")


@router.get("/{project_id}/analysis", response_model=ProjectAnalysisResponse)
def get_project_analysis(
    project_id: str,
    user_id: str = Query(..., description="Идентификатор пользователя"),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден.")
    _ensure_project_owner(project, user_id, "смотреть анализ")

    meta = pm.load(project)
    analysis = meta.get("analysis")
    if not analysis:
        raise HTTPException(status_code=404, detail="Результат анализа пока отсутствует.")

    return ProjectAnalysisResponse(project_id=project.id, status=project.status.value, analysis=analysis)
