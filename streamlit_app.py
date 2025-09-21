"""Streamlit migration of the Appsmith "📝Поставки" page."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import psycopg2
import streamlit as st
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

POSTGRES_REQUIRED_KEYS = ["host", "port", "dbname", "user", "password"]


@st.cache_resource(show_spinner=False)
def get_connection() -> psycopg2.extensions.connection:
    """Create (and cache) the PostgreSQL connection."""

    if "postgres" not in st.secrets:
        st.error(
            "Не найдены параметры подключения. Добавьте секцию 'postgres' в "
            ".streamlit/secrets.toml перед запуском приложения."
        )
        st.stop()

    config = st.secrets["postgres"]
    missing = [key for key in POSTGRES_REQUIRED_KEYS if key not in config]
    if missing:
        st.error(
            "В секции [postgres] отсутствуют поля: " + ", ".join(missing)
        )
        st.stop()

    connection = psycopg2.connect(
        host=config.get("host"),
        port=config.get("port"),
        dbname=config.get("dbname"),
        user=config.get("user"),
        password=config.get("password"),
        cursor_factory=RealDictCursor,
        application_name="streamlit_supplies_migration",
    )
    connection.autocommit = True
    return connection


def run_select(query: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    """Execute a SELECT statement and return a list of dictionaries."""

    connection = get_connection()
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        records: List[Dict[str, Any]] = cursor.fetchall()
    return records


def run_modify(query: str, params: Optional[Sequence[Any]] = None) -> None:
    """Execute INSERT/UPDATE/DELETE statements."""

    connection = get_connection()
    with connection.cursor() as cursor:
        cursor.execute(query, params)


def run_modify_values(query: str, values: Iterable[Sequence[Any]]) -> None:
    """Execute statements that rely on psycopg2.execute_values."""

    connection = get_connection()
    with connection.cursor() as cursor:
        execute_values(cursor, query, values)


def clear_all_caches() -> None:
    """Clear all cached datasets after a mutation."""

    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Domain specific helpers
# ---------------------------------------------------------------------------

DESTINATIONS = {
    "SPB": "Санкт-Петербург",
    "MSK": "Москва",
    "NSK": "Новосибирск",
    "KRD": "Краснодар",
    "KZN": "Казань",
}

ORDER_TYPES = ["Срочная", "Новинки", "Большая", "Внеплановая", "Сэмплы"]

PRODUCT_TYPES = [
    "1. Прозрачный",
    "2. Матовый",
    "3. Книжки",
    "4. Закупаемые чехлы",
    "5. Premium Soft-Touch",
    "7. Аксессуары",
    "8. Производство",
    "9. Остальное",
]

DELIVERY_METHODS = [
    "HL Авиа",
    "HL Земля",
    "HL прямой",
    "789 земля",
    "Другое",
    "CS Прямой",
    "CS Карго",
]

RUSSIAN_LOGISTICS = [
    "Деловые Линии",
    "АэропортСервис",
    "Желдор",
    "Частник",
    "Самовывоз",
    "Другое",
]

STATUS_OPTIONS = ["Заказан", "На оплате", "Оплачен", "Отправлен", "Пришло"]


def to_date(value: Any) -> Optional[date]:
    """Convert assorted representations into ``datetime.date`` values."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str) and value.strip():
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y"):
            try:
                return datetime.strptime(value.split(" ")[0], pattern).date()
            except ValueError:
                continue
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalise_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "y"}:
        return True
    if value_str in {"false", "0", "no", "n"}:
        return False
    return None


def parse_label(label: str) -> Tuple[str, int, str]:
    """Split a delivery label into prefix, numeric part and suffix."""

    match = re.search(r"(\d+)([A-Za-z]*)$", label or "")
    if not match:
        return label, 0, ""
    number_part = int(match.group(1))
    suffix = match.group(2) or ""
    prefix = label[: match.start(1)]
    return prefix, number_part, suffix


def compute_next_delivery_labels(supplies: pd.DataFrame) -> Dict[str, str]:
    """Replicate Utils.getNextDeliveryLabels from Appsmith."""

    labels = supplies.get("delivery_label", pd.Series(dtype=str)).dropna()

    def next_label(prefix: str) -> str:
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        numbers = [int(match.group(1)) for label in labels for match in [pattern.match(label)] if match]
        next_number = max(numbers) + 1 if numbers else 1
        return f"{prefix}{next_number}"

    return {"MS-CS": next_label("MS-CS"), "MS-BC": next_label("MS-BC")}


def expand_models_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the delivery_labels column is always a list."""

    if df.empty:
        return df
    if "delivery_labels" in df.columns:
        df = df.copy()
        df["delivery_labels"] = df["delivery_labels"].apply(
            lambda value: list(value) if isinstance(value, (list, tuple)) else ([value] if value else [])
        )
    return df


def apply_problematic_flag(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["is_problematic"] = pd.Series([], dtype=bool)
        df["days_from_order"] = pd.Series([], dtype="float")
        return df
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    today = pd.Timestamp(date.today())
    df["days_from_order"] = (today - df["order_date"]).dt.days
    df["is_problematic"] = (
        df["status"].fillna("") != "Отправлен"
    ) & df["order_date"].notna() & (df["days_from_order"] > 20)
    return df


# ---------------------------------------------------------------------------
# Cached data accessors (Part 1)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def fetch_supplies() -> pd.DataFrame:
    query = """
        WITH orders_summary AS (
            SELECT SUM(quantity_ordered) AS qty_path, delivery_label
            FROM public.cp_orders_list
            WHERE delivery_label IS NOT NULL
            GROUP BY delivery_label
        )
        SELECT
            sl.product_type,
            sl.amount_rmb,
            sl.logistic_cost,
            sl.payment_importer,
            sl.packing_list,
            sl.qty_box,
            sl.weight,
            sl.order_in_ms,
            sl.delivery_label,
            sl.supplier,
            sl.logistics_method,
            sl.status,
            sl.order_date,
            sl.invoice_date,
            sl.dispatch_store_date,
            sl.dispatch_russia_date,
            sl.estimated_date,
            sl.russia_logistics_method,
            sl.logistic_cost_ru,
            sl.id,
            sl.payment_id,
            sl.invoice_cny,
            sl.invoice_usd,
            sl.invoice_rub,
            sl.date_russia,
            sl.date_china,
            sl.currency_rate,
            sl.brand,
            sl.russian_log_number,
            sl.commentary,
            sl.brand_shipped,
            sl.destination,
            sl.order_type,
            CASE
                WHEN EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'hw_cargos'
                      AND column_name = 'ttnFile_name'
                ) THEN (
                    SELECT hw."ttnFile_name" FROM public.hw_cargos hw WHERE id = sl.hw_cargo_id
                )
                ELSE NULL
            END AS ttnFile_name,
            CASE
                WHEN EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'hw_cargos'
                      AND column_name = 'ttnFile_uuid'
                ) THEN (
                    SELECT hw."ttnFile_uuid" FROM public.hw_cargos hw WHERE id = sl.hw_cargo_id
                )
                ELSE NULL
            END AS ttnFile_uuid,
            hw."account_source",
            os.qty_path
        FROM public.cp_supply_list sl
        LEFT JOIN public.hw_cargos hw ON sl.hw_cargo_id = hw.id
        LEFT JOIN orders_summary os ON sl.delivery_label = os.delivery_label
        WHERE sl.delivery_in_russia_date IS NULL
          AND sl.product_type IS NOT NULL
          AND sl.status <> 'Пришло'
        ORDER BY sl.product_type, sl.status DESC, sl.order_date ASC;
    """
    dataframe = pd.DataFrame(run_select(query))
    if dataframe.empty:
        return dataframe

    numeric_columns = [
        "amount_rmb",
        "logistic_cost",
        "qty_box",
        "weight",
        "invoice_cny",
        "invoice_usd",
        "invoice_rub",
        "currency_rate",
        "logistic_cost_ru",
        "qty_path",
    ]
    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    date_columns = [
        "order_date",
        "invoice_date",
        "dispatch_store_date",
        "dispatch_russia_date",
        "estimated_date",
        "date_russia",
        "date_china",
        "delivery_date",
        "delivery_in_russia_date",
    ]
    for column in date_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")

    if "payment_importer" in dataframe.columns:
        dataframe["payment_importer"] = dataframe["payment_importer"].map(normalise_bool)

    dataframe = apply_problematic_flag(dataframe)
    return dataframe


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models_mapping() -> pd.DataFrame:
    query = """
        SELECT DISTINCT g."Модель" AS model, ARRAY_AGG(way.delivery_articul) AS delivery_labels
        FROM cp_path way
        LEFT JOIN public."goods" g ON way.uuid = g.uuid
        GROUP BY g."Модель"
        ORDER BY g."Модель" ASC
    """
    dataframe = pd.DataFrame(run_select(query))
    return expand_models_dataframe(dataframe)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_suppliers_list() -> List[str]:
    query = "SELECT DISTINCT supplier FROM public.cp_suppliers_list"
    return sorted(
        [row["supplier"] for row in run_select(query) if row.get("supplier")]
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_shipment_tags() -> List[str]:
    query = "SELECT DISTINCT delivery_label FROM public.cp_supply_list"
    return sorted(
        [row["delivery_label"] for row in run_select(query) if row.get("delivery_label")]
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_labels_with_payments() -> pd.DataFrame:
    query = "SELECT DISTINCT delivery_label, payment_id, id FROM public.cp_supply_list"
    return pd.DataFrame(run_select(query))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_archive_supplies() -> pd.DataFrame:
    query = """
        SELECT *
        FROM cp_supply_list
        WHERE status = 'Пришло'
        ORDER BY CAST(delivery_date AS DATE) DESC
    """
    dataframe = pd.DataFrame(run_select(query))
    if dataframe.empty:
        return dataframe
    date_columns = [
        "order_date",
        "delivery_date",
        "estimated_date",
        "date_russia",
        "date_china",
        "logistic_date",
        "dispatch_store_date",
    ]
    for column in date_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")
    for column in ["amount_rmb", "invoice_cny", "invoice_usd", "logistic_cost", "logistic_cost_ru", "weight"]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    dataframe = apply_problematic_flag(dataframe)
    return dataframe


@st.cache_data(ttl=300, show_spinner=False)
def fetch_supply_positions(delivery_label: Optional[str]) -> pd.DataFrame:
    if not delivery_label:
        return pd.DataFrame()
    query = """
        SELECT
            COALESCE(g."Модель", bs.model) AS model,
            way.quantity_shipped,
            COALESCE(g."Цвет", bs.color) AS color
        FROM public."cp_orders_list" way
        LEFT JOIN public."goods" g ON way.name = g."Наименование"
        LEFT JOIN public."goods_msk" bs ON way.name = bs.name_ms AND g."Модель" IS NULL
        WHERE way.delivery_label = %s
    """
    return pd.DataFrame(run_select(query, (delivery_label,)))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_archive_positions(delivery_label: Optional[str]) -> pd.DataFrame:
    if not delivery_label:
        return pd.DataFrame()
    query = """
        SELECT g."Модель" AS model, way.quantity_shipped, g."Цвет" AS color
        FROM public."cp_orders_list" way
        LEFT JOIN public."goods" g ON way.name = g."Наименование"
        WHERE way.delivery_label = %s
    """
    return pd.DataFrame(run_select(query, (delivery_label,)))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_supply_plan() -> pd.DataFrame:
    query = """
        WITH supply_data AS (
            SELECT
                order_type,
                DATE_TRUNC('week', order_date) AS order_week,
                delivery_label,
                payment_id,
                approximate_rub,
                invoice_cny,
                invoice_usd,
                invoice_rub,
                logistic_cost
            FROM cp_supply_list
        ),
        supply_summary AS (
            SELECT
                order_type,
                order_week,
                SUM(approximate_rub) AS total_approximate_rub,
                SUM(invoice_cny) AS total_invoice_cny,
                SUM(invoice_usd) AS total_invoice_usd,
                SUM(invoice_rub) AS total_supply_invoice_rub,
                SUM(logistic_cost) AS total_logistic_cost
            FROM supply_data
            GROUP BY order_type, order_week
        ),
        order_summary AS (
            SELECT
                sd.order_type,
                sd.order_week,
                SUM(co.quantity_ordered) AS total_quantity
            FROM cp_orders_list co
            JOIN supply_data sd ON co.delivery_label = sd.delivery_label
            GROUP BY sd.order_type, sd.order_week
        ),
        payment_totals AS (
            SELECT
                cp.payment_id,
                cp.invoice_rub AS payment_invoice_rub,
                cp.invoice_cny AS payment_invoice_cny,
                (
                    SELECT SUM(sd.invoice_cny)
                    FROM supply_data sd
                    WHERE sd.payment_id = cp.payment_id
                      AND sd.payment_id IS NOT NULL
                ) AS total_supply_cny_for_payment
            FROM cp_payments_list cp
            WHERE cp.payment_id IN (
                SELECT DISTINCT payment_id FROM supply_data WHERE payment_id IS NOT NULL
            )
        ),
        payment_distributions AS (
            SELECT
                sd.order_type,
                sd.order_week,
                sd.payment_id,
                pt.payment_invoice_rub,
                pt.payment_invoice_cny,
                SUM(sd.invoice_cny) AS week_invoice_cny,
                pt.total_supply_cny_for_payment,
                CASE
                    WHEN pt.total_supply_cny_for_payment > 0
                        THEN (SUM(sd.invoice_cny) / pt.total_supply_cny_for_payment) * pt.payment_invoice_rub
                    ELSE 0
                END AS proportional_payment_rub
            FROM supply_data sd
            JOIN payment_totals pt ON sd.payment_id = pt.payment_id
            WHERE sd.payment_id IS NOT NULL
            GROUP BY
                sd.order_type,
                sd.order_week,
                sd.payment_id,
                pt.payment_invoice_rub,
                pt.payment_invoice_cny,
                pt.total_supply_cny_for_payment
        ),
        payment_summary AS (
            SELECT
                order_type,
                order_week,
                SUM(payment_invoice_rub) AS total_payment_invoice_rub,
                SUM(proportional_payment_rub) AS total_proportional_payment_rub
            FROM payment_distributions
            GROUP BY order_type, order_week
        )
        SELECT
            ss.order_type,
            ss.order_week,
            COALESCE(os.total_quantity, 0) AS total_quantity,
            ss.total_approximate_rub,
            ss.total_invoice_cny,
            ss.total_invoice_usd,
            ss.total_supply_invoice_rub,
            ss.total_logistic_cost,
            COALESCE(ps.total_payment_invoice_rub, 0) AS total_payment_invoice_rub,
            COALESCE(ps.total_proportional_payment_rub, 0) AS total_proportional_payment_rub
        FROM supply_summary ss
        LEFT JOIN order_summary os ON ss.order_type = os.order_type AND ss.order_week = os.order_week
        LEFT JOIN payment_summary ps ON ss.order_type = ps.order_type AND ss.order_week = ps.order_week
        ORDER BY ss.order_week DESC, ss.order_type;
    """
    dataframe = pd.DataFrame(run_select(query))
    if dataframe.empty:
        return dataframe
    dataframe["order_week"] = pd.to_datetime(dataframe["order_week"], errors="coerce")
    for column in dataframe.columns:
        if dataframe[column].dtype == "object":
            dataframe[column] = pd.to_numeric(dataframe[column], errors="ignore")
    return dataframe


@st.cache_data(ttl=300, show_spinner=False)
def fetch_supplies_without_logistics() -> pd.DataFrame:
    query = """
        SELECT id, weight, delivery_label, supplier, brand_shipped,
               shipped_quantity, delivery_date
        FROM cp_supply_list
        WHERE logistic_cost IS NULL
    """
    dataframe = pd.DataFrame(run_select(query))
    if dataframe.empty:
        return dataframe
    dataframe["weight"] = pd.to_numeric(dataframe["weight"], errors="coerce")
    dataframe["delivery_date"] = pd.to_datetime(dataframe["delivery_date"], errors="coerce")
    return dataframe


@st.cache_data(ttl=300, show_spinner=False)
def fetch_quality_checks() -> pd.DataFrame:
    query = """
        SELECT qc.*, l.supplier, l.product_type
        FROM cp_quality_checks qc
        LEFT JOIN cp_supply_list l ON l.delivery_label = qc.batch_label
        ORDER BY qc.id DESC
    """
    dataframe = pd.DataFrame(run_select(query))
    if dataframe.empty:
        return dataframe
    for column in ["check_date", "created_at"]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")
    return dataframe


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def update_supply_record(supply_id: int, payload: Dict[str, Any]) -> None:
    query = """
        UPDATE cp_supply_list
        SET
            order_date = %(order_date)s,
            product_type = %(product_type)s,
            delivery_label = %(delivery_label)s,
            logistic_cost = %(logistic_cost)s,
            payment_importer = %(payment_importer)s,
            order_in_ms = %(order_in_ms)s,
            invoice_date = %(invoice_date)s,
            logistic_date = %(logistic_date)s,
            estimated_date = %(estimated_date)s,
            logistics_method = %(logistics_method)s,
            status = %(status)s,
            amount_rmb = %(amount_rmb)s,
            invoice_cny = %(invoice_cny)s,
            invoice_usd = %(invoice_usd)s,
            qty_box = %(qty_box)s,
            weight = %(weight)s,
            russia_logistics_method = %(russia_logistics_method)s,
            russian_log_number = %(russian_log_number)s,
            supplier = %(supplier)s,
            destination = %(destination)s,
            order_type = %(order_type)s,
            delivery_date = CASE WHEN %(status)s = 'Пришло' THEN CURRENT_DATE ELSE delivery_date END,
            delivery_in_russia_date = CASE WHEN %(status)s = 'Пришло' THEN CURRENT_DATE ELSE delivery_in_russia_date END,
            dispatch_store_date = CASE WHEN %(status)s = 'Отправлен' THEN %(dispatch_store_date)s ELSE dispatch_store_date END
        WHERE id = %(supply_id)s
    """
    parameters = payload.copy()
    parameters["supply_id"] = supply_id
    run_modify(query, parameters)
    clear_all_caches()


def insert_new_supplies(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    query = """
        INSERT INTO cp_supply_list (
            order_date,
            delivery_label,
            order_in_ms,
            estimated_date,
            logistics_method,
            status,
            amount_rmb,
            product_type,
            supplier,
            destination,
            order_type,
            commentary,
            payment_id,
            brand,
            invoice_date,
            invoice_cny,
            invoice_usd,
            invoice_rub,
            date_russia,
            date_china
        ) VALUES %s
    """
    tuples: List[Tuple[Any, ...]] = []
    for record in records:
        tuples.append(
            (
                record.get("order_date"),
                record.get("delivery_label"),
                record.get("order_in_ms"),
                record.get("estimated_date"),
                record.get("logistics_method"),
                record.get("status"),
                record.get("amount_rmb"),
                record.get("product_type"),
                record.get("supplier"),
                record.get("destination"),
                record.get("order_type"),
                record.get("commentary"),
                record.get("payment_id"),
                record.get("brand"),
                record.get("invoice_date"),
                record.get("invoice_cny"),
                record.get("invoice_usd"),
                record.get("invoice_rub"),
                record.get("date_russia"),
                record.get("date_china"),
            )
        )
    run_modify_values(query, tuples)
    clear_all_caches()


def delete_supply_record(supply_id: int) -> None:
    run_modify("DELETE FROM cp_supply_list WHERE id = %s", (supply_id,))
    clear_all_caches()


def update_archive_supply(supply_id: int, payload: Dict[str, Any]) -> None:
    query = """
        UPDATE cp_supply_list
        SET
            russia_logistics_method = %(russia_logistics_method)s,
            russian_log_number = %(russian_log_number)s,
            logistic_cost_ru = %(logistic_cost_ru)s,
            delivery_date = %(delivery_date)s,
            logistic_cost = %(logistic_cost)s,
            invoice_cny = %(invoice_cny)s,
            amount_rmb = %(amount_rmb)s,
            logistic_date = %(logistic_date)s
        WHERE id = %(supply_id)s
    """
    parameters = payload.copy()
    parameters["supply_id"] = supply_id
    run_modify(query, parameters)
    clear_all_caches()


def mark_supply_returned(supply_id: int) -> None:
    query = """
        UPDATE cp_supply_list
        SET status = 'Отправлен',
            delivery_date = NULL,
            delivery_in_russia_date = NULL
        WHERE id = %s
    """
    run_modify(query, (supply_id,))
    clear_all_caches()


def insert_ttn_records(rows: List[Dict[str, Any]]) -> None:
    cleaned: List[Tuple[Optional[float], str]] = []
    for row in rows:
        number = (row or {}).get("ttn_number")
        if not number:
            continue
        cleaned.append((to_float(row.get("sum")), number))
    if not cleaned:
        return
    run_modify_values(
        "INSERT INTO cp_supplies_ttn_list (sum, ttn_number) VALUES %s",
        cleaned,
    )


def distribute_logistics_cost(assignments: List[Tuple[int, float]]) -> None:
    if not assignments:
        return
    query = sql.SQL(
        "UPDATE cp_supply_list AS c SET logistic_cost = data.cost FROM (VALUES %s) AS data(id, cost) WHERE c.id = data.id"
    )
    run_modify_values(query.as_string(get_connection()), assignments)
    clear_all_caches()


# ---------------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------------


def render_supplies_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("По выбранным условиям поставки не найдены.")
        return

    display_df = df.copy()
    for column in ["order_date", "estimated_date", "dispatch_store_date"]:
        if column in display_df:
            display_df[column] = display_df[column].dt.date
    display_df["Проблемный"] = display_df["is_problematic"].map({True: "⚠️", False: ""})
    columns_to_show = [
        "delivery_label",
        "supplier",
        "product_type",
        "status",
        "destination",
        "logistics_method",
        "order_date",
        "estimated_date",
        "dispatch_store_date",
        "amount_rmb",
        "invoice_cny",
        "logistic_cost",
        "Проблемный",
        "commentary",
    ]
    existing_columns = [column for column in columns_to_show if column in display_df.columns]
    st.dataframe(
        display_df.set_index("id")[existing_columns],
        use_container_width=True,
        hide_index=False,
    )


def summary_badges(df: pd.DataFrame) -> None:
    if df.empty:
        return
    total_amount = df["amount_rmb"].sum(skipna=True)
    total_logistics = df["logistic_cost"].sum(skipna=True)
    total_invoice_cny = df["invoice_cny"].sum(skipna=True)
    problem_count = int(df["is_problematic"].sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Поставок", f"{len(df):,}".replace(",", " "))
    col2.metric("Сумма заказа, CNY", f"{total_amount:,.0f}".replace(",", " "))
    col3.metric("Стоимость логистики, ₽", f"{total_logistics:,.0f}".replace(",", " "))
    col4.metric("Проблемные", problem_count)


def destination_label(code: Optional[str]) -> str:
    if not code:
        return "—"
    return f"{code} — {DESTINATIONS.get(code, code)}"


def optional_date_input(
    widget_host: Any,
    label: str,
    value: Any,
    key_prefix: str,
) -> Optional[date]:
    """Render a date input with the ability to keep the value empty."""

    current_value = to_date(value)
    clear = widget_host.checkbox(
        "Оставить пустым",
        value=current_value is None,
        key=f"{key_prefix}_clear",
    )
    selected = widget_host.date_input(
        label,
        value=current_value or date.today(),
        key=f"{key_prefix}_value",
        disabled=clear,
    )
    return None if clear else selected


def interactive_data_editor(data: Any, **kwargs: Any) -> pd.DataFrame:
    """Use the available editable table widget regardless of Streamlit version."""

    editor = getattr(st, "data_editor", None)
    if editor is None:
        editor = getattr(st, "experimental_data_editor", None)
    if editor is not None:
        return editor(data, **kwargs)

    st.warning(
        "Текущая версия Streamlit не поддерживает редактирование таблицы."
        " Заполните данные вручную в интерфейсе или обновите Streamlit."
    )
    dataframe = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    st.dataframe(dataframe, use_container_width=kwargs.get("use_container_width", True))
    return dataframe.copy()


def collect_payment_metadata(payment_id: str, reference_label: Optional[str]) -> Dict[str, Any]:
    brand_rows = run_select(
        "SELECT brand FROM cp_payments_list WHERE payment_id = %s LIMIT 1",
        (payment_id,),
    )
    supply_rows: List[Dict[str, Any]] = []
    if reference_label:
        supply_rows = run_select(
            "SELECT invoice_date, date_russia, date_china FROM cp_supply_list WHERE delivery_label = %s LIMIT 1",
            (reference_label,),
        )
    return {
        "payment_id": payment_id,
        "brand": brand_rows[0]["brand"] if brand_rows else None,
        "invoice_date": to_date((supply_rows[0] if supply_rows else {}).get("invoice_date")),
        "invoice_cny": 0.0,
        "invoice_usd": 0.0,
        "invoice_rub": 0.0,
        "date_russia": to_date((supply_rows[0] if supply_rows else {}).get("date_russia")),
        "date_china": to_date((supply_rows[0] if supply_rows else {}).get("date_china")),
    }


def build_new_supply_records(
    base_label: str,
    parts: int,
    ms_number: Optional[str],
    logistics_method: Optional[str],
    amount_rmb: Optional[float],
    product_type: Optional[str],
    supplier: Optional[str],
    destination: Optional[str],
    order_type: Optional[str],
    commentary: Optional[str],
    payment_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    prefix, start_number, suffix = parse_label(base_label)
    order_date = date.today()
    estimated = order_date + timedelta(days=39)
    records: List[Dict[str, Any]] = []
    for offset in range(parts):
        generated_label = f"{prefix}{start_number + offset}{suffix}"
        record = {
            "order_date": order_date,
            "delivery_label": generated_label,
            "order_in_ms": ms_number or None,
            "estimated_date": estimated,
            "logistics_method": logistics_method or None,
            "status": "Заказан",
            "amount_rmb": amount_rmb,
            "product_type": product_type or None,
            "supplier": supplier or None,
            "destination": destination or None,
            "order_type": order_type or None,
            "commentary": commentary or None,
            "payment_id": None,
            "brand": None,
            "invoice_date": None,
            "invoice_cny": None,
            "invoice_usd": None,
            "invoice_rub": None,
            "date_russia": None,
            "date_china": None,
        }
        if payment_meta:
            record.update(payment_meta)
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Streamlit tab renderers
# ---------------------------------------------------------------------------


def render_editor_tab(
    supplies_df: pd.DataFrame,
    models_df: pd.DataFrame,
    suppliers_list: List[str],
    shipment_tags: List[str],
    labels_with_payments: pd.DataFrame,
) -> None:
    st.subheader("Текущие поставки")

    with st.expander("Фильтры", expanded=True):
        col1, col2, col3 = st.columns(3)
        supplier_filter = col1.multiselect("Поставщик", suppliers_list)
        destination_filter = col2.multiselect(
            "Направление",
            list(DESTINATIONS.keys()),
            format_func=destination_label,
        )
        logistics_filter = col3.multiselect(
            "Тип доставки",
            sorted([value for value in supplies_df.get("logistics_method", pd.Series(dtype=str)).dropna().unique()]),
        )
        col4, col5, col6 = st.columns(3)
        status_filter = col4.multiselect("Статус", STATUS_OPTIONS + ["Проблемный"])
        label_filter = col5.multiselect("Артикул", sorted(shipment_tags))
        brand_filter = col6.multiselect(
            "Бренд",
            sorted([value for value in supplies_df.get("brand", pd.Series(dtype=str)).dropna().unique()]),
        )
        model_filter = st.multiselect(
            "Товар (модель)",
            sorted([value for value in models_df.get("model", pd.Series(dtype=str)).dropna().unique()]),
        )

    filtered = supplies_df.copy()
    if supplier_filter:
        filtered = filtered[filtered["supplier"].isin(supplier_filter)]
    if destination_filter:
        filtered = filtered[filtered["destination"].isin(destination_filter)]
    if logistics_filter:
        filtered = filtered[filtered["logistics_method"].isin(logistics_filter)]
    if label_filter:
        filtered = filtered[filtered["delivery_label"].isin(label_filter)]
    if brand_filter:
        filtered = filtered[filtered["brand"].isin(brand_filter)]
    if status_filter:
        statuses = [value for value in status_filter if value != "Проблемный"]
        mask = pd.Series(False, index=filtered.index)
        if statuses:
            mask = mask | filtered["status"].isin(statuses)
        if "Проблемный" in status_filter:
            mask = mask | filtered["is_problematic"].fillna(False)
        filtered = filtered[mask]
    if model_filter:
        model_map = {
            row["model"]: row["delivery_labels"]
            for _, row in models_df.dropna(subset=["model"]).iterrows()
        }
        allowed_labels: set[str] = set()
        for model in model_filter:
            allowed_labels.update(model_map.get(model, []))
        filtered = filtered[filtered["delivery_label"].isin(allowed_labels)]

    summary_badges(filtered)
    render_supplies_table(filtered)

    if filtered.empty:
        return

    selection_df = filtered[["id", "delivery_label", "supplier", "status"]].copy()
    selection_df["label"] = selection_df.apply(
        lambda row: f"{row['delivery_label']} — {row['supplier']} ({row['status']})",
        axis=1,
    )
    selection_dict = dict(zip(selection_df["id"], selection_df["label"]))
    selected_id = st.selectbox(
        "Выберите поставку для редактирования",
        options=list(selection_dict.keys()),
        format_func=lambda value: selection_dict.get(value, str(value)),
    )
    selected_supply = supplies_df.loc[supplies_df["id"] == selected_id]
    if selected_supply.empty:
        st.warning("Не удалось загрузить выбранную поставку.")
        return
    row = selected_supply.iloc[0]

    st.markdown("### Обновление данных поставки")
    with st.form("update_supply_form", clear_on_submit=False):
        general_col1, general_col2, general_col3 = st.columns(3)
        delivery_label = general_col1.text_input(
            "Артикул",
            value=row.get("delivery_label", ""),
        )
        supplier_options = suppliers_list if suppliers_list else [row.get("supplier", "") or ""]
        product_type = general_col2.selectbox(
            "Категория",
            PRODUCT_TYPES,
            index=PRODUCT_TYPES.index(row["product_type"]) if row.get("product_type") in PRODUCT_TYPES else 0,
        )
        supplier_value = general_col3.selectbox(
            "Поставщик",
            supplier_options,
            index=supplier_options.index(row["supplier"]) if row.get("supplier") in supplier_options else 0,
        )

        col4, col5, col6 = st.columns(3)
        destination_value = col4.selectbox(
            "Направление",
            list(DESTINATIONS.keys()),
            format_func=destination_label,
            index=list(DESTINATIONS.keys()).index(row["destination"]) if row.get("destination") in DESTINATIONS else 0,
        )
        order_type_value = col5.selectbox(
            "Тип закупки",
            ORDER_TYPES,
            index=ORDER_TYPES.index(row["order_type"]) if row.get("order_type") in ORDER_TYPES else 0,
        )
        logistics_method = col6.selectbox(
            "Тип доставки",
            DELIVERY_METHODS,
            index=DELIVERY_METHODS.index(row["logistics_method"]) if row.get("logistics_method") in DELIVERY_METHODS else 0,
        )

        col7, col8, col9 = st.columns(3)
        order_in_ms = col7.text_input("Номер заказа в MoySklad", value=row.get("order_in_ms", ""))
        order_date_value = col8.date_input(
            "Дата заказа",
            value=to_date(row.get("order_date")) or date.today(),
        )
        estimated_date_value = col9.date_input(
            "Плановая дата прихода",
            value=to_date(row.get("estimated_date")) or date.today() + timedelta(days=39),
        )

        col10, col11, col12 = st.columns(3)
        status_value = col10.selectbox(
            "Статус",
            STATUS_OPTIONS,
            index=STATUS_OPTIONS.index(row["status"]) if row.get("status") in STATUS_OPTIONS else 0,
        )
        amount_rmb = col11.number_input(
            "Стоимость товара, CNY",
            min_value=0.0,
            value=float(row.get("amount_rmb") or 0.0),
            step=100.0,
        )
        logistic_cost = col12.number_input(
            "Логистика из Китая, ₽",
            min_value=0.0,
            value=float(row.get("logistic_cost") or 0.0),
            step=100.0,
        )

        col13, col14, col15 = st.columns(3)
        invoice_cny = col13.number_input(
            "Инвойс, CNY",
            min_value=0.0,
            value=float(row.get("invoice_cny") or 0.0),
            step=100.0,
        )
        invoice_usd = col14.number_input(
            "Инвойс, USD",
            min_value=0.0,
            value=float(row.get("invoice_usd") or 0.0),
            step=100.0,
        )
        qty_box_raw = col15.text_input("Кол-во коробок", value=str(row.get("qty_box") or ""))

        col16, col17, col18 = st.columns(3)
        weight_raw = col16.text_input("Вес, кг", value=str(row.get("weight") or ""))
        invoice_date_value = optional_date_input(
            col17,
            "Дата инвойса",
            row.get("invoice_date"),
            key_prefix=f"invoice_date_{selected_id}",
        )
        logistic_date_value = optional_date_input(
            col18,
            "Дата логистики",
            row.get("logistic_date"),
            key_prefix=f"logistic_date_{selected_id}",
        )

        col19, col20, col21 = st.columns(3)
        payment_importer = col19.checkbox(
            "Логистику оплачивает импортёр",
            value=bool(row.get("payment_importer")),
        )
        russia_logistics_method = col20.selectbox(
            "ТК по РФ",
            RUSSIAN_LOGISTICS,
            index=RUSSIAN_LOGISTICS.index(row["russia_logistics_method"]) if row.get("russia_logistics_method") in RUSSIAN_LOGISTICS else 0,
        )
        russian_ttn = col21.text_input("ТТН РФ", value=row.get("russian_log_number", ""))

        shipment_date_value = None
        if status_value == "Отправлен":
            shipment_date_value = optional_date_input(
                st,
                "Дата отправки со склада",
                row.get("dispatch_store_date"),
                key_prefix=f"dispatch_store_date_{selected_id}",
            )

        commentary_value = st.text_area("Комментарий", value=row.get("commentary", ""))

        submitted = st.form_submit_button("Сохранить изменения")
        if submitted:
            payload = {
                "order_date": order_date_value,
                "product_type": product_type,
                "delivery_label": delivery_label,
                "logistic_cost": logistic_cost,
                "payment_importer": payment_importer,
                "order_in_ms": order_in_ms or None,
                "invoice_date": invoice_date_value,
                "logistic_date": logistic_date_value,
                "estimated_date": estimated_date_value,
                "logistics_method": logistics_method,
                "status": status_value,
                "amount_rmb": amount_rmb,
                "invoice_cny": invoice_cny,
                "invoice_usd": invoice_usd,
                "qty_box": to_int(qty_box_raw),
                "weight": to_float(weight_raw),
                "russia_logistics_method": russia_logistics_method,
                "russian_log_number": russian_ttn or None,
                "supplier": supplier_value,
                "destination": destination_value,
                "order_type": order_type_value,
                "dispatch_store_date": shipment_date_value,
                "commentary": commentary_value or None,
            }
            try:
                update_supply_record(selected_id, payload)
                st.success("Изменения сохранены")
            except Exception as exc:
                st.error(f"Не удалось сохранить изменения: {exc}")

    st.markdown("#### Позиции в заказе")
    positions_df = fetch_supply_positions(row.get("delivery_label"))
    if positions_df.empty:
        st.info("Для выбранной поставки нет данных о позициях.")
    else:
        st.dataframe(positions_df, use_container_width=True)

    with st.expander("Добавить новую поставку", expanded=False):
        render_new_order_form(
            supplies_df,
            suppliers_list,
            models_df,
            shipment_tags,
            labels_with_payments,
        )


def render_new_order_form(
    supplies_df: pd.DataFrame,
    suppliers_list: List[str],
    models_df: pd.DataFrame,
    shipment_tags: List[str],
    labels_with_payments: pd.DataFrame,
) -> None:
    next_labels = compute_next_delivery_labels(supplies_df)
    st.caption(
        f"Следующие артикулы: MS-CS → {next_labels['MS-CS']}, MS-BC → {next_labels['MS-BC']}"
    )

    with st.form("new_supply_form"):
        col1, col2, col3 = st.columns(3)
        destination_value = col1.selectbox(
            "Направление",
            list(DESTINATIONS.keys()),
            format_func=destination_label,
        )
        supplier_options = suppliers_list if suppliers_list else ["—"]
        supplier_value = col2.selectbox("Поставщик", supplier_options)
        product_type_value = col3.selectbox("Категория", PRODUCT_TYPES)

        col4, col5, col6 = st.columns(3)
        logistics_method = col4.selectbox("Тип доставки", DELIVERY_METHODS)
        order_type_value = col5.selectbox("Тип закупки", ORDER_TYPES)
        ms_number = col6.text_input("Номер заказа в MoySklad", value="")

        col7, col8, col9 = st.columns(3)
        amount_rmb = col7.number_input("Сумма оплаты, CNY", min_value=0.0, step=100.0)
        split_orders = col8.checkbox("Разделить на части", value=False)
        parts_count = col9.number_input(
            "Количество частей",
            min_value=1,
            max_value=50,
            value=1,
            step=1,
        )
        if not split_orders:
            parts_count = 1

        link_payment = st.checkbox("Связать с оплатой", value=False)
        payment_meta: Optional[Dict[str, Any]] = None
        if link_payment:
            available_links = labels_with_payments.dropna(subset=["payment_id", "delivery_label"])
            available_links = available_links[available_links["payment_id"].astype(str) != ""]
            if available_links.empty:
                st.warning("Нет доступных оплаченных поставок для привязки.")
            else:
                link_options = {
                    f"{row['delivery_label']} (оплата {row['payment_id']})": (row["payment_id"], row["delivery_label"])
                    for _, row in available_links.iterrows()
                }
                chosen = st.selectbox("Выберите связанную поставку", list(link_options.keys()))
                payment_meta = collect_payment_metadata(*link_options[chosen])

        commentary = st.text_area("Комментарий", value="")
        suggested_label = (
            next_labels["MS-BC"]
            if destination_value in {"MSK", "NSK", "KZN", "KRD"}
            else next_labels["MS-CS"]
        )
        delivery_label = st.text_input("Артикул", value=suggested_label)

        submitted = st.form_submit_button("Добавить")
        if submitted:
            try:
                records = build_new_supply_records(
                    delivery_label,
                    int(parts_count),
                    ms_number,
                    logistics_method,
                    amount_rmb,
                    product_type_value,
                    supplier_value,
                    destination_value,
                    order_type_value,
                    commentary,
                    payment_meta,
                )
            except Exception as exc:
                st.error(f"Не удалось создать поставку: {exc}")
                return

            existing_labels = set(label for label in shipment_tags if label)
            duplicates = [record["delivery_label"] for record in records if record["delivery_label"] in existing_labels]
            if duplicates:
                st.error(
                    "Невозможно создать поставку: следующие артикулы уже существуют — "
                    + ", ".join(sorted(set(duplicates)))
                )
                return

            try:
                insert_new_supplies(records)
                st.success("Поставка добавлена")
            except Exception as exc:
                st.error(f"Не удалось создать поставку: {exc}")


def render_archive_tab(archive_df: pd.DataFrame) -> None:
    st.subheader("Архив поставок")

    with st.expander("Фильтры", expanded=True):
        col1, col2, col3 = st.columns(3)
        supplier_filter = col1.multiselect(
            "Поставщик",
            sorted([value for value in archive_df.get("supplier", pd.Series(dtype=str)).dropna().unique()]),
        )
        logistics_filter = col2.multiselect(
            "Тип доставки",
            sorted([value for value in archive_df.get("logistics_method", pd.Series(dtype=str)).dropna().unique()]),
        )
        destination_filter = col3.multiselect(
            "Направление",
            list(DESTINATIONS.keys()),
            format_func=destination_label,
        )
        col4, col5 = st.columns(2)
        brand_filter = col4.multiselect(
            "Бренд",
            sorted([value for value in archive_df.get("brand_shipped", pd.Series(dtype=str)).dropna().unique()]),
        )
        if not archive_df.empty:
            min_date = to_date(archive_df["delivery_date"].min()) or date.today()
            max_date = to_date(archive_df["delivery_date"].max()) or date.today()
        else:
            min_date = max_date = date.today()
        date_range = col5.date_input(
            "Интервал дат доставки",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    filtered = archive_df.copy()
    if supplier_filter:
        filtered = filtered[filtered["supplier"].isin(supplier_filter)]
    if logistics_filter:
        filtered = filtered[filtered["logistics_method"].isin(logistics_filter)]
    if destination_filter:
        filtered = filtered[filtered["destination"].isin(destination_filter)]
    if brand_filter:
        filtered = filtered[filtered["brand_shipped"].isin(brand_filter)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["delivery_date"] >= pd.Timestamp(start_date))
            & (filtered["delivery_date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
        ]

    render_supplies_table(filtered)

    if filtered.empty:
        return

    selected_id = st.selectbox(
        "Выберите поставку",
        options=filtered["id"].tolist(),
        format_func=lambda value: filtered.loc[filtered["id"] == value, "delivery_label"].iloc[0],
    )
    selected_row = archive_df.loc[archive_df["id"] == selected_id]
    if selected_row.empty:
        st.warning("Не удалось получить данные поставки из архива.")
        return
    row = selected_row.iloc[0]

    with st.form("archive_form"):
        col1, col2, col3 = st.columns(3)
        logistics_china = col1.number_input(
            "Доставка из Китая, ₽",
            min_value=0.0,
            value=float(row.get("logistic_cost") or 0.0),
            step=100.0,
        )
        logistics_russia = col2.number_input(
            "Доставка по РФ, ₽",
            min_value=0.0,
            value=float(row.get("logistic_cost_ru") or 0.0),
            step=100.0,
        )
        invoice_cny = col3.number_input(
            "Инвойс, CNY",
            min_value=0.0,
            value=float(row.get("invoice_cny") or 0.0),
            step=100.0,
        )

        col4, col5, col6 = st.columns(3)
        amount_rmb = col4.number_input(
            "Принято, CNY",
            min_value=0.0,
            value=float(row.get("amount_rmb") or 0.0),
            step=100.0,
        )
        logistic_date = optional_date_input(
            col5,
            "Дата логистики",
            row.get("logistic_date"),
            key_prefix=f"archive_logistic_{selected_id}",
        )
        delivery_date = optional_date_input(
            col6,
            "Дата доставки",
            row.get("delivery_date"),
            key_prefix=f"archive_delivery_{selected_id}",
        )

        col7, col8 = st.columns(2)
        russia_logistics = col7.selectbox(
            "ТК по РФ",
            RUSSIAN_LOGISTICS,
            index=RUSSIAN_LOGISTICS.index(row["russia_logistics_method"]) if row.get("russia_logistics_method") in RUSSIAN_LOGISTICS else 0,
        )
        russian_ttn = col8.text_input("Номер ТТН", value=row.get("russian_log_number", ""))

        submitted = st.form_submit_button("Сохранить")
        if submitted:
            payload = {
                "russia_logistics_method": russia_logistics,
                "russian_log_number": russian_ttn or None,
                "logistic_cost_ru": logistics_russia,
                "delivery_date": delivery_date,
                "logistic_cost": logistics_china,
                "invoice_cny": invoice_cny,
                "amount_rmb": amount_rmb,
                "logistic_date": logistic_date,
            }
            try:
                update_archive_supply(selected_id, payload)
                st.success("Информация обновлена")
            except Exception as exc:
                st.error(f"Не удалось сохранить: {exc}")

    if st.button("Вернуть из архива", type="secondary"):
        try:
            mark_supply_returned(selected_id)
            st.success("Поставка возвращена в работу")
        except Exception as exc:
            st.error(f"Не удалось вернуть из архива: {exc}")

    st.markdown("#### Позиции")
    archive_positions = fetch_archive_positions(row.get("delivery_label"))
    if archive_positions.empty:
        st.info("Нет позиций для выбранной поставки.")
    else:
        st.dataframe(archive_positions, use_container_width=True)

    st.markdown("#### Учёт ТТН")
    default_table = pd.DataFrame([{"ttn_number": "", "sum": 0.0}])
    with st.form("ttn_form"):
        ttn_table = interactive_data_editor(
            default_table,
            num_rows="dynamic",
            use_container_width=True,
        )
        if st.form_submit_button("Добавить в реестр"):
            try:
                insert_ttn_records(ttn_table.to_dict("records"))
                st.success("ТТН сохранены")
            except Exception as exc:
                st.error(f"Не удалось добавить ТТН: {exc}")

    st.markdown("#### Распределение логистики между поставками")
    distribution_df = fetch_supplies_without_logistics()
    if distribution_df.empty:
        st.info("Нет поставок без распределённой логистики.")
    else:
        st.dataframe(distribution_df, use_container_width=True)
        options = {
            f"{row.delivery_label} — {row.supplier}": row.id
            for row in distribution_df.itertuples()
        }
        selected_keys = st.multiselect("Выберите поставки", list(options.keys()))
        selected_ids = [options[key] for key in selected_keys]
        total_cost = st.number_input("Сумма логистики, ₽", min_value=0.0, step=100.0)
        if st.button("Распределить логистику"):
            chosen = distribution_df[distribution_df["id"].isin(selected_ids)]
            weight_series = chosen["weight"].apply(lambda value: 0.0 if pd.isna(value) else float(value))
            total_weight = weight_series.sum()
            if chosen.empty or total_weight <= 0:
                st.error("Невозможно выполнить распределение: требуется ненулевая масса.")
            else:
                assignments: List[Tuple[int, float]] = []
                allocated = 0.0
                for index, record in enumerate(chosen.itertuples()):
                    weight_value = 0.0 if pd.isna(record.weight) else float(record.weight)
                    share = weight_value / total_weight if total_weight else 0.0
                    cost_value = round(total_cost * share, 2)
                    assignments.append((int(record.id), cost_value))
                    allocated += cost_value

                remainder = round(total_cost - allocated, 2)
                if assignments and remainder:
                    last_id, last_cost = assignments[-1]
                    assignments[-1] = (last_id, round(last_cost + remainder, 2))

                try:
                    distribute_logistics_cost(assignments)
                    st.success("Логистика распределена")
                except Exception as exc:
                    st.error(f"Не удалось обновить логистику: {exc}")


def render_plan_tab(plan_df: pd.DataFrame) -> None:
    st.subheader("План закупок")
    if plan_df.empty:
        st.info("Нет данных для отображения.")
        return
    plan_df = plan_df.copy()
    plan_df["order_week"] = plan_df["order_week"].dt.date
    st.dataframe(plan_df, use_container_width=True)


def render_help_tab() -> None:
    st.subheader("Справка")
    st.markdown(
        """
        ### Основные действия

        * **Редактор** — управление активными поставками: фильтруйте список,
          обновляйте данные, создавайте новые заказы и просматривайте состав
          поставки.
        * **Архив** — фиксация фактической логистики, загрузка номеров ТТН и
          возврат поставок в работу.
        * **План закупок** — агрегированные показатели по неделям и типам
          закупок.
        * **Проверки** — контрольные отчёты: поставки без распределённой
          логистики и результаты инспекций качества.

        ### Настройки подключения

        Данные считываются напрямую из PostgreSQL. Параметры подключения
        задаются в файле `.streamlit/secrets.toml` (см. README.md).

        ### Кэширование

        Для ускорения большая часть запросов кэшируется на несколько минут.
        Кнопка **«Обновить данные»** в боковой панели сбрасывает кэш и
        повторно загружает информацию из базы.
        """
    )


def render_checks_tab() -> None:
    st.subheader("Контрольные отчёты")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Без распределённой логистики")
        distribution_df = fetch_supplies_without_logistics()
        if distribution_df.empty:
            st.info("Все поставки имеют указанную стоимость логистики.")
        else:
            st.dataframe(distribution_df, use_container_width=True)

    with col2:
        st.markdown("#### Проверки качества")
        qc_df = fetch_quality_checks()
        if qc_df.empty:
            st.info("Нет данных о проверках качества.")
        else:
            st.dataframe(qc_df, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Управление поставками", layout="wide")
    st.title("📝 Поставки — Streamlit")
    st.caption("Миграция страницы Appsmith на Python + Streamlit")

    with st.sidebar:
        st.header("Действия")
        if st.button("Обновить данные"):
            clear_all_caches()
            st.experimental_rerun()
        if "postgres" in st.secrets:
            pg = st.secrets["postgres"]
            st.markdown(
                f"**База данных:** {pg.get('dbname', '?')} @ {pg.get('host', '?')}:{pg.get('port', '?')}"
            )

    supplies_df = fetch_supplies()
    models_df = fetch_models_mapping()
    suppliers_list = fetch_suppliers_list()
    shipment_tags = fetch_shipment_tags()
    labels_with_payments = fetch_labels_with_payments()

    tabs = st.tabs([
        "Редактор",
        "Архив поставок",
        "План закупок",
        "Справка",
        "Проверки",
    ])

    with tabs[0]:
        render_editor_tab(
            supplies_df,
            models_df,
            suppliers_list,
            shipment_tags,
            labels_with_payments,
        )

    with tabs[1]:
        archive_df = fetch_archive_supplies()
        render_archive_tab(archive_df)

    with tabs[2]:
        plan_df = fetch_supply_plan()
        render_plan_tab(plan_df)

    with tabs[3]:
        render_help_tab()

    with tabs[4]:
        render_checks_tab()


if __name__ == "__main__":
    main()

