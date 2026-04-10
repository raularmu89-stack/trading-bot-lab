# Vehicle Manager

Aplicación web sencilla para gestionar una flota de vehículos. Construida con
**Flask** y **SQLite** (sin dependencias externas aparte de Flask).

## Características

- CRUD completo de vehículos (crear, listar, ver, editar y eliminar).
- Búsqueda libre por matrícula, marca, modelo o propietario.
- Filtro por estado (`Disponible`, `En uso`, `Mantenimiento`, `Retirado`).
- Panel con contadores por estado.
- Validación de formularios en servidor y matrícula única.
- Interfaz responsive sin dependencias de frontend.

## Campos de un vehículo

| Campo        | Obligatorio | Notas                                          |
|--------------|-------------|------------------------------------------------|
| Matrícula    | Sí          | Única, se normaliza a mayúsculas               |
| Marca        | Sí          |                                                |
| Modelo       | Sí          |                                                |
| Año          | Sí          | Entre 1900 y el año siguiente al actual        |
| Color        | No          |                                                |
| Combustible  | No          | gasoline / diesel / electric / hybrid / lpg    |
| Kilometraje  | No          | Entero ≥ 0                                     |
| Propietario  | No          |                                                |
| Estado       | Sí          | available / in_use / maintenance / retired     |

## Instalación y uso

```bash
cd vehicle_app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

La aplicación se expondrá en <http://127.0.0.1:5000/>. La base de datos SQLite
(`vehicles.db`) se crea automáticamente la primera vez que se ejecuta.

## Estructura

```
vehicle_app/
├── app.py              # Aplicación Flask + rutas + esquema de BD
├── requirements.txt
├── static/
│   └── style.css
└── templates/
    ├── base.html
    ├── index.html      # Listado + filtros
    ├── form.html       # Alta y edición
    └── detail.html     # Ficha detallada
```
