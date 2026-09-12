"""
database_sangue.py
===================

Camada de persistência (SQLite) para a interface de triagem de doenças
por exame de sangue. Guarda os dados do paciente/exame e o resultado de
cada detecção (incluindo o detalhe da votação entre modelos, quando
aplicável).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "blood_app.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                prontuario TEXT,
                data_nascimento TEXT,
                idade INTEGER,
                sexo TEXT,
                data_exame TEXT,
                medico_solicitante TEXT,
                indicacao_clinica TEXT,
                sintomas TEXT,
                comorbidades TEXT,
                observacoes TEXT,
                hemoglobina REAL,
                hematocrito REAL,
                hemacias REAL,
                leucocitos REAL,
                plaquetas REAL,
                glicemia REAL,
                pressao_sistolica REAL,
                pressao_diastolica REAL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blood_detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                modo_analise TEXT NOT NULL,
                modelo_utilizado TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                prob_saudavel REAL,
                prob_hipertenso REAL,
                prob_leucemia REAL,
                prob_diabetes REAL,
                votos_individuais TEXT,
                concordancia_modelos INTEGER,
                responsavel_tecnico TEXT,
                observacoes_laudo TEXT,
                data_deteccao TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def calculate_age(birth_date: Optional[date]) -> Optional[int]:
    if not birth_date:
        return None
    today = date.today()
    return today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )


def insert_patient(data: dict) -> int:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO patients (
                nome, prontuario, data_nascimento, idade, sexo, data_exame,
                medico_solicitante, indicacao_clinica, sintomas, comorbidades,
                observacoes, hemoglobina, hematocrito, hemacias, leucocitos,
                plaquetas, glicemia, pressao_sistolica, pressao_diastolica,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("nome"),
                data.get("prontuario"),
                data.get("data_nascimento"),
                data.get("idade"),
                data.get("sexo"),
                data.get("data_exame"),
                data.get("medico_solicitante"),
                data.get("indicacao_clinica"),
                json.dumps(data.get("sintomas") or [], ensure_ascii=False),
                json.dumps(data.get("comorbidades") or [], ensure_ascii=False),
                data.get("observacoes"),
                data.get("hemoglobina"),
                data.get("hematocrito"),
                data.get("hemacias"),
                data.get("leucocitos"),
                data.get("plaquetas"),
                data.get("glicemia"),
                data.get("pressao_sistolica"),
                data.get("pressao_diastolica"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_detection(data: dict) -> int:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO blood_detections (
                patient_id, modo_analise, modelo_utilizado, label, confidence,
                prob_saudavel, prob_hipertenso, prob_leucemia, prob_diabetes,
                votos_individuais, concordancia_modelos, responsavel_tecnico,
                observacoes_laudo, data_deteccao
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["patient_id"],
                data["modo_analise"],
                data["modelo_utilizado"],
                data["label"],
                data["confidence"],
                data.get("prob_saudavel"),
                data.get("prob_hipertenso"),
                data.get("prob_leucemia"),
                data.get("prob_diabetes"),
                json.dumps(data.get("votos_individuais"), ensure_ascii=False) if data.get("votos_individuais") else None,
                data.get("concordancia_modelos"),
                data.get("responsavel_tecnico"),
                data.get("observacoes_laudo"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def fetch_history(nome_filtro: Optional[str] = None, resultado_filtro: str = "Todos") -> list[dict]:
    conn = _get_connection()
    try:
        query = """
            SELECT
                d.detection_id, d.patient_id, d.modo_analise, d.modelo_utilizado,
                d.label, d.confidence, d.prob_saudavel, d.prob_hipertenso,
                d.prob_leucemia, d.prob_diabetes, d.votos_individuais,
                d.concordancia_modelos, d.responsavel_tecnico, d.observacoes_laudo,
                d.data_deteccao,
                p.nome, p.prontuario, p.idade, p.sexo, p.data_exame,
                p.medico_solicitante, p.indicacao_clinica, p.hemoglobina,
                p.hematocrito, p.hemacias, p.leucocitos, p.plaquetas,
                p.glicemia, p.pressao_sistolica, p.pressao_diastolica
            FROM blood_detections d
            JOIN patients p ON p.patient_id = d.patient_id
            WHERE 1=1
        """
        params: list = []
        if nome_filtro:
            query += " AND (p.nome LIKE ? OR p.prontuario LIKE ?)"
            like = f"%{nome_filtro}%"
            params.extend([like, like])
        if resultado_filtro and resultado_filtro != "Todos":
            query += " AND d.label = ?"
            params.append(resultado_filtro)
        query += " ORDER BY d.data_deteccao DESC"

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_stats_summary() -> dict:
    conn = _get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM blood_detections").fetchone()[0]
        stats = {"total": total}
        for label, key in [
            ("Saudável", "saudavel"),
            ("Hipertenso", "hipertenso"),
            ("Leucemia", "leucemia"),
            ("Diabetes", "diabetes"),
        ]:
            count = conn.execute(
                "SELECT COUNT(*) FROM blood_detections WHERE label = ?", (label,)
            ).fetchone()[0]
            stats[key] = count
        return stats
    finally:
        conn.close()
