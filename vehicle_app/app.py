"""Aplicación web para gestión de vehículos.

CRUD completo usando Flask + SQLite. Permite registrar, listar,
editar y eliminar vehículos, además de buscarlos por matrícula,
marca o modelo.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime

from flask import (
    Flask,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "vehicles.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["DATABASE"] = DATABASE


# --------------------------------------------------------------------------- #
# Base de datos
# --------------------------------------------------------------------------- #
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exception: BaseException | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    """Crea la tabla de vehículos si no existe."""
    with closing(sqlite3.connect(app.config["DATABASE"])) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                plate        TEXT    NOT NULL UNIQUE,
                brand        TEXT    NOT NULL,
                model        TEXT    NOT NULL,
                year         INTEGER NOT NULL,
                color        TEXT,
                fuel_type    TEXT,
                mileage      INTEGER DEFAULT 0,
                owner        TEXT,
                status       TEXT    NOT NULL DEFAULT 'available',
                created_at   TEXT    NOT NULL,
                updated_at   TEXT    NOT NULL
            )
            """
        )
        db.commit()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
VALID_STATUSES = ("available", "in_use", "maintenance", "retired")
VALID_FUELS = ("gasoline", "diesel", "electric", "hybrid", "lpg", "other")


def _parse_form(form) -> tuple[dict, list[str]]:
    """Valida y normaliza los datos del formulario.

    Devuelve (datos_normalizados, lista_de_errores).
    """
    errors: list[str] = []
    plate = (form.get("plate") or "").strip().upper()
    brand = (form.get("brand") or "").strip()
    model = (form.get("model") or "").strip()
    year_raw = (form.get("year") or "").strip()
    color = (form.get("color") or "").strip() or None
    fuel_type = (form.get("fuel_type") or "").strip() or None
    mileage_raw = (form.get("mileage") or "0").strip()
    owner = (form.get("owner") or "").strip() or None
    status = (form.get("status") or "available").strip()

    if not plate:
        errors.append("La matrícula es obligatoria.")
    if not brand:
        errors.append("La marca es obligatoria.")
    if not model:
        errors.append("El modelo es obligatorio.")

    current_year = datetime.now().year
    try:
        year = int(year_raw)
        if year < 1900 or year > current_year + 1:
            errors.append(f"El año debe estar entre 1900 y {current_year + 1}.")
    except ValueError:
        year = 0
        errors.append("El año debe ser un número entero.")

    try:
        mileage = int(mileage_raw or 0)
        if mileage < 0:
            errors.append("El kilometraje no puede ser negativo.")
    except ValueError:
        mileage = 0
        errors.append("El kilometraje debe ser un número entero.")

    if fuel_type and fuel_type not in VALID_FUELS:
        errors.append("Tipo de combustible no válido.")

    if status not in VALID_STATUSES:
        errors.append("Estado no válido.")

    data = {
        "plate": plate,
        "brand": brand,
        "model": model,
        "year": year,
        "color": color,
        "fuel_type": fuel_type,
        "mileage": mileage,
        "owner": owner,
        "status": status,
    }
    return data, errors


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    q = (request.args.get("q") or "").strip()
    status_filter = (request.args.get("status") or "").strip()

    sql = "SELECT * FROM vehicles WHERE 1=1"
    params: list = []

    if q:
        sql += " AND (plate LIKE ? OR brand LIKE ? OR model LIKE ? OR owner LIKE ?)"
        like = f"%{q}%"
        params.extend([like, like, like, like])

    if status_filter and status_filter in VALID_STATUSES:
        sql += " AND status = ?"
        params.append(status_filter)

    sql += " ORDER BY created_at DESC"

    vehicles = get_db().execute(sql, params).fetchall()

    counts = get_db().execute(
        """
        SELECT
            COUNT(*)                                          AS total,
            SUM(CASE WHEN status='available'   THEN 1 ELSE 0 END) AS available,
            SUM(CASE WHEN status='in_use'      THEN 1 ELSE 0 END) AS in_use,
            SUM(CASE WHEN status='maintenance' THEN 1 ELSE 0 END) AS maintenance,
            SUM(CASE WHEN status='retired'     THEN 1 ELSE 0 END) AS retired
        FROM vehicles
        """
    ).fetchone()

    return render_template(
        "index.html",
        vehicles=vehicles,
        q=q,
        status_filter=status_filter,
        counts=counts,
        statuses=VALID_STATUSES,
    )


@app.route("/vehicles/new", methods=["GET", "POST"])
def create_vehicle():
    if request.method == "POST":
        data, errors = _parse_form(request.form)
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template(
                "form.html",
                vehicle=data,
                statuses=VALID_STATUSES,
                fuels=VALID_FUELS,
                action_url=url_for("create_vehicle"),
                title="Nuevo vehículo",
            )

        now = _now_iso()
        try:
            get_db().execute(
                """
                INSERT INTO vehicles
                    (plate, brand, model, year, color, fuel_type,
                     mileage, owner, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["plate"], data["brand"], data["model"], data["year"],
                    data["color"], data["fuel_type"], data["mileage"],
                    data["owner"], data["status"], now, now,
                ),
            )
            get_db().commit()
        except sqlite3.IntegrityError:
            flash(f"Ya existe un vehículo con la matrícula {data['plate']}.", "error")
            return render_template(
                "form.html",
                vehicle=data,
                statuses=VALID_STATUSES,
                fuels=VALID_FUELS,
                action_url=url_for("create_vehicle"),
                title="Nuevo vehículo",
            )

        flash("Vehículo registrado correctamente.", "success")
        return redirect(url_for("index"))

    return render_template(
        "form.html",
        vehicle=None,
        statuses=VALID_STATUSES,
        fuels=VALID_FUELS,
        action_url=url_for("create_vehicle"),
        title="Nuevo vehículo",
    )


@app.route("/vehicles/<int:vehicle_id>")
def view_vehicle(vehicle_id: int):
    vehicle = get_db().execute(
        "SELECT * FROM vehicles WHERE id = ?", (vehicle_id,)
    ).fetchone()
    if vehicle is None:
        flash("Vehículo no encontrado.", "error")
        return redirect(url_for("index"))
    return render_template("detail.html", vehicle=vehicle)


@app.route("/vehicles/<int:vehicle_id>/edit", methods=["GET", "POST"])
def edit_vehicle(vehicle_id: int):
    vehicle = get_db().execute(
        "SELECT * FROM vehicles WHERE id = ?", (vehicle_id,)
    ).fetchone()
    if vehicle is None:
        flash("Vehículo no encontrado.", "error")
        return redirect(url_for("index"))

    if request.method == "POST":
        data, errors = _parse_form(request.form)
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template(
                "form.html",
                vehicle={**data, "id": vehicle_id},
                statuses=VALID_STATUSES,
                fuels=VALID_FUELS,
                action_url=url_for("edit_vehicle", vehicle_id=vehicle_id),
                title=f"Editar {vehicle['plate']}",
            )

        try:
            get_db().execute(
                """
                UPDATE vehicles
                   SET plate=?, brand=?, model=?, year=?, color=?,
                       fuel_type=?, mileage=?, owner=?, status=?, updated_at=?
                 WHERE id=?
                """,
                (
                    data["plate"], data["brand"], data["model"], data["year"],
                    data["color"], data["fuel_type"], data["mileage"],
                    data["owner"], data["status"], _now_iso(), vehicle_id,
                ),
            )
            get_db().commit()
        except sqlite3.IntegrityError:
            flash(f"Ya existe un vehículo con la matrícula {data['plate']}.", "error")
            return render_template(
                "form.html",
                vehicle={**data, "id": vehicle_id},
                statuses=VALID_STATUSES,
                fuels=VALID_FUELS,
                action_url=url_for("edit_vehicle", vehicle_id=vehicle_id),
                title=f"Editar {vehicle['plate']}",
            )

        flash("Vehículo actualizado correctamente.", "success")
        return redirect(url_for("view_vehicle", vehicle_id=vehicle_id))

    return render_template(
        "form.html",
        vehicle=vehicle,
        statuses=VALID_STATUSES,
        fuels=VALID_FUELS,
        action_url=url_for("edit_vehicle", vehicle_id=vehicle_id),
        title=f"Editar {vehicle['plate']}",
    )


@app.route("/vehicles/<int:vehicle_id>/delete", methods=["POST"])
def delete_vehicle(vehicle_id: int):
    db = get_db()
    cursor = db.execute("DELETE FROM vehicles WHERE id = ?", (vehicle_id,))
    db.commit()
    if cursor.rowcount == 0:
        flash("Vehículo no encontrado.", "error")
    else:
        flash("Vehículo eliminado.", "success")
    return redirect(url_for("index"))


# --------------------------------------------------------------------------- #
# Filtros de plantilla
# --------------------------------------------------------------------------- #
STATUS_LABELS = {
    "available": "Disponible",
    "in_use": "En uso",
    "maintenance": "Mantenimiento",
    "retired": "Retirado",
}

FUEL_LABELS = {
    "gasoline": "Gasolina",
    "diesel": "Diésel",
    "electric": "Eléctrico",
    "hybrid": "Híbrido",
    "lpg": "GLP",
    "other": "Otro",
}


@app.template_filter("status_label")
def status_label(value: str) -> str:
    return STATUS_LABELS.get(value, value or "-")


@app.template_filter("fuel_label")
def fuel_label(value: str) -> str:
    return FUEL_LABELS.get(value, value or "-")


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
