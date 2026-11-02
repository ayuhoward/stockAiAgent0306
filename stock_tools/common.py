import psycopg2
import streamlit as st


def get_db_connection():
    return psycopg2.connect(st.secrets["SUPABASE_DB_URL"])


def normalize_symbol(symbol: str) -> str:
    clean_symbol = symbol.strip().upper()
    if clean_symbol.isdigit():
        clean_symbol += ".TW"
    elif not clean_symbol.endswith(".TW") and not clean_symbol.endswith(".TWO") and not clean_symbol.isalpha():
        clean_symbol += ".TW"
    return clean_symbol
