import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime

# --- 1. Настройка страницы и подключение к БД ---

st.set_page_config(layout="wide", page_title="Пульт закупок")

st.title("🤍 Пульт закупок")

# Инициализация и кэширование подключения к базе данных
@st.cache_resource
def init_connection():
    """Подключается к базе данных PostgreSQL."""
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Ошибка подключения к базе данных: {e}")
        return None

conn = init_connection()

# --- 2. Функции для работы с базой данных ---

@st.cache_data(ttl=600) # Кэшируем данные на 10 минут
def run_query(query):
    """Выполняет SELECT-запросы и возвращает результат в виде DataFrame."""
    if conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return pd.DataFrame(rows, columns=colnames)
    return pd.DataFrame()

def modify_db(query, params):
    """Выполняет запросы INSERT, UPDATE, DELETE."""
    if conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()

# --- 3. Загрузка данных ---

# Основной запрос для получения текущих поставок (аналог getSupplies)
supplies_query = """
SELECT 
    id, product_type, delivery_label, supplier, status, 
    order_date, estimated_date, logistics_method, brand,
    destination, order_in_ms, amount_rmb
FROM public.cp_supply_list 
WHERE delivery_in_russia_date IS NULL AND product_type IS NOT NULL AND status <> 'Пришло'
ORDER BY product_type, status DESC, order_date ASC;
"""

# Запрос для получения архивных данных (аналог get_archive_list)
archive_query = """
SELECT 
    id, delivery_date, russia_logistics_method, russian_log_number, 
    delivery_label, supplier, logistic_cost, logistic_cost_ru, 
    weight_from_ms, price_ms
FROM cp_supply_list 
WHERE status = 'Пришло'
ORDER BY CAST(delivery_date AS DATE) DESC;
"""

supplies_df = run_query(supplies_query)
archive_df = run_query(archive_query)
suppliers_list = run_query("SELECT DISTINCT supplier FROM public.cp_suppliers_list")['supplier'].tolist()
logistics_list = supplies_df['logistics_method'].unique().tolist()
brands_list = supplies_df['brand'].dropna().unique().tolist()
destinations_list = supplies_df['destination'].dropna().unique().tolist()

# --- 4. Интерфейс приложения ---

tab_editor, tab_archive = st.tabs(["📝 Редактор", "🗄️ Архив поставок"])

# --- ВКЛАДКА "РЕДАКТОР" ---
with tab_editor:
    st.header("Текущие поставки")

    # --- Фильтры ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_destination = st.multiselect("Направление", options=destinations_list, placeholder="Все направления")
    with col2:
        selected_suppliers = st.multiselect("Поставщик", options=suppliers_list, placeholder="Все поставщики")
    with col3:
        selected_logistics = st.multiselect("Логистика", options=logistics_list, placeholder="Все типы")
    with col4:
        selected_brands = st.multiselect("Бренд (ИП)", options=brands_list, placeholder="Все бренды")

    # Применение фильтров
    filtered_df = supplies_df.copy()
    if selected_destination:
        filtered_df = filtered_df[filtered_df['destination'].isin(selected_destination)]
    if selected_suppliers:
        filtered_df = filtered_df[filtered_df['supplier'].isin(selected_suppliers)]
    if selected_logistics:
        filtered_df = filtered_df[filtered_df['logistics_method'].isin(selected_logistics)]
    if selected_brands:
        filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
    
    # --- Отображение таблицы ---
    st.dataframe(filtered_df, use_container_width=True)


# --- Боковая панель для добавления и редактирования ---
with st.sidebar:
    st.header("Управление поставками")
    
    # --- Форма добавления новой поставки ---
    with st.expander("➕ Добавить новую поставку", expanded=False):
        with st.form("new_supply_form", clear_on_submit=True):
            st.subheader("Данные новой поставки")
            
            new_delivery_label = st.text_input("Артикул поставки*", help="Обязательное поле")
            new_product_type = st.selectbox("Категория*", options=["1. Прозрачный", "2. Матовый", "3. Книжки", "4. Закупаемые чехлы", "5. Premium Soft-Touch", "7. Аксессуары", "8. Производство", "9. Остальное"])
            new_supplier = st.selectbox("Поставщик*", options=suppliers_list)
            new_destination = st.selectbox("Направление*", options=["SPB", "MSK", "NSK", "KRD", "KZN"])
            new_amount_rmb = st.number_input("Сумма (RMB)", min_value=0.0, format="%.2f")
            
            submitted = st.form_submit_button("Добавить поставку")
            if submitted:
                if not new_delivery_label or not new_product_type or not new_supplier:
                    st.warning("Пожалуйста, заполните все обязательные поля (*).")
                else:
                    # Аналог addSupplies
                    insert_query = """
                    INSERT INTO cp_supply_list (order_date, delivery_label, estimated_date, status, product_type, supplier, destination, amount_rmb) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """
                    # Расчет предполагаемой даты (сегодня + 39 дней)
                    estimated_date = datetime.now() + pd.Timedelta(days=39)
                    params = (datetime.now().date(), new_delivery_label, estimated_date.date(), 'Заказан', new_product_type, new_supplier, new_destination, new_amount_rmb)
                    
                    modify_db(insert_query, params)
                    st.success(f"Поставка '{new_delivery_label}' успешно добавлена!")
                    st.cache_data.clear() # Очистка кэша для обновления данных

    # --- Редактирование существующей поставки ---
    with st.expander("✏️ Редактировать поставку", expanded=True):
        if supplies_df.empty:
            st.info("Нет поставок для редактирования.")
        else:
            # Выбор поставки для редактирования
            supply_to_edit_label = st.selectbox("Выберите артикул поставки для редактирования", options=supplies_df['delivery_label'])
            
            if supply_to_edit_label:
                selected_supply = supplies_df[supplies_df['delivery_label'] == supply_to_edit_label].iloc[0]
                
                with st.form("edit_supply_form"):
                    st.subheader(f"Редактирование: {selected_supply['delivery_label']}")
                    
                    supply_id = int(selected_supply['id'])
                    
                    # Поля для редактирования
                    status = st.selectbox("Статус", options=["Заказан", "На оплате", "Оплачен", "Отправлен", "Пришло"], index=["Заказан", "На оплате", "Оплачен", "Отправлен", "Пришло"].index(selected_supply['status']))
                    order_in_ms = st.text_input("Номер заказа в МС", value=selected_supply.get('order_in_ms', ''))
                    amount_rmb = st.number_input("Сумма (RMB)", min_value=0.0, value=float(selected_supply.get('amount_rmb', 0) or 0), format="%.2f")

                    col_save, col_delete = st.columns(2)
                    with col_save:
                        save_clicked = st.form_submit_button("Сохранить изменения", use_container_width=True)
                    with col_delete:
                        delete_clicked = st.form_submit_button("❌ Удалить поставку", use_container_width=True)
                    
                    if save_clicked:
                        # Аналог updateSupplies
                        update_query = """
                        UPDATE cp_supply_list SET status = %s, order_in_ms = %s, amount_rmb = %s WHERE id = %s;
                        """
                        params = (status, order_in_ms, amount_rmb, supply_id)
                        modify_db(update_query, params)
                        st.success(f"Данные для поставки '{supply_to_edit_label}' обновлены.")
                        st.cache_data.clear()
                    
                    if delete_clicked:
                        # Аналог deleteSupplies
                        delete_query = "DELETE FROM cp_supply_list WHERE id = %s;"
                        modify_db(delete_query, (supply_id,))
                        st.warning(f"Поставка '{supply_to_edit_label}' удалена.")
                        st.cache_data.clear()


# --- ВКЛАДКА "АРХИВ" ---
with tab_archive:
    st.header("Архив поставок (статус 'Пришло')")
    
    # --- Фильтры для архива ---
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Дата доставки с", value=None)
    with col2:
        end_date = st.date_input("Дата доставки по", value=None)

    # Применение фильтров
    filtered_archive_df = archive_df.copy()
    if start_date:
        filtered_archive_df = filtered_archive_df[filtered_archive_df['delivery_date'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_archive_df = filtered_archive_df[filtered_archive_df['delivery_date'] <= pd.to_datetime(end_date)]
        
    st.dataframe(filtered_archive_df, use_container_width=True)