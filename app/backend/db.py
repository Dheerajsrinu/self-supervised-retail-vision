import uuid
import psycopg
from datetime import datetime, timedelta
from app.config import DB_CONFIG
from app.backend.security import hash_password, verify_password

def get_connection():
    return psycopg.connect(**DB_CONFIG)

def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Chat threads table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id UUID PRIMARY KEY,
                    title TEXT,
                    user_id TEXT,
                    is_waiting_for_review BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                        
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    thread_id UUID NOT NULL,
                    role TEXT CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
                ON chat_messages(thread_id);
            """)

            # Users table with role
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id UUID PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    address TEXT NOT NULL,
                    pincode INT NOT NULL,
                    age INT DEFAULT 0,
                    role TEXT DEFAULT 'customer' CHECK (role IN ('customer', 'store_manager')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Migration: Add role column if not exists
            cur.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='users' AND column_name='name') THEN
                        ALTER TABLE users ADD COLUMN name TEXT DEFAULT '';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='users' AND column_name='age') THEN
                        ALTER TABLE users ADD COLUMN age INT DEFAULT 0;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='users' AND column_name='role') THEN
                        ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'customer';
                    END IF;
                END $$;
            """)

            # Manager codes table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS manager_codes (
                    id SERIAL PRIMARY KEY,
                    code TEXT UNIQUE NOT NULL,
                    description TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Insert default manager codes if table is empty
            cur.execute("""
                INSERT INTO manager_codes (code, description) 
                SELECT 'STORE2025', 'Default store manager code'
                WHERE NOT EXISTS (SELECT 1 FROM manager_codes WHERE code = 'STORE2025');
            """)
            cur.execute("""
                INSERT INTO manager_codes (code, description) 
                SELECT 'MANAGER123', 'Secondary manager code'
                WHERE NOT EXISTS (SELECT 1 FROM manager_codes WHERE code = 'MANAGER123');
            """)

            # Orders table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    products JSONB NOT NULL,
                    status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_orders_thread
                        FOREIGN KEY (user_id)
                        REFERENCES chat_threads(thread_id)
                        ON DELETE CASCADE
                );
            """)
            
            # Migration: Add status column to orders if not exists
            cur.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='orders' AND column_name='status') THEN
                        ALTER TABLE orders ADD COLUMN status TEXT DEFAULT 'completed';
                    END IF;
                END $$;
            """)

            # Telemetry events table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    user_id UUID,
                    user_email TEXT,
                    user_role TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_events_type
                ON telemetry_events(event_type);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_events_created
                ON telemetry_events(created_at);
            """)

            # Model performance table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    operation TEXT,
                    duration_ms FLOAT NOT NULL,
                    input_size TEXT,
                    user_id UUID,
                    thread_id UUID,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_performance_name
                ON model_performance(model_name);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_performance_created
                ON model_performance(created_at);
            """)

            # Model verification logs table - tracks inference vs LLM output consistency
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_verification_logs (
                    id SERIAL PRIMARY KEY,
                    thread_id UUID,
                    inference_output JSONB NOT NULL,
                    llm_output JSONB NOT NULL,
                    match_status BOOLEAN NOT NULL,
                    mismatched_items JSONB,
                    total_inference_items INT,
                    total_llm_items INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_verification_logs_thread
                ON model_verification_logs(thread_id);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_verification_logs_match
                ON model_verification_logs(match_status);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_verification_logs_created
                ON model_verification_logs(created_at);
            """)

            conn.commit()

# ==================== THREAD FUNCTIONS ====================

def create_thread(user_id, title="Custom Chat"):
    thread_id = str(uuid.uuid4())
    user_id = str(user_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_threads (thread_id, title, user_id) VALUES (%s, %s, %s)",
                (thread_id, title, user_id)
            )
            conn.commit()
    return thread_id

def get_all_threads():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT thread_id, title
                FROM chat_threads
                ORDER BY created_at DESC
            """)
            return cur.fetchall()

def get_threads_by_user(user_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT thread_id, title, user_id FROM chat_threads WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            return cur.fetchall()

# ==================== MESSAGE FUNCTIONS ====================

def save_message(thread_id: str, role: str, content: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (thread_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (thread_id, role, content)
            )
            conn.commit()

def get_messages_by_thread(thread_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE thread_id = %s
                ORDER BY created_at
                """,
                (thread_id,)
            )
            return cur.fetchall()

# ==================== REVIEW STATE FUNCTIONS ====================

def clear_waiting_for_review(thread_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                UPDATE chat_threads
                SET is_waiting_for_review = FALSE,
                    updated_at = NOW()
                    WHERE thread_id = %s
            """
            cur.execute(query, (thread_id,))

def mark_waiting_for_review(thread_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                UPDATE chat_threads
                SET is_waiting_for_review = TRUE,
                    updated_at = NOW()
                    WHERE thread_id = %s
            """
            cur.execute(query, (thread_id,))

def is_waiting_for_review(thread_id: str) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT is_waiting_for_review
                FROM chat_threads
                WHERE thread_id = %s
                """,
                (thread_id,)
            )

            row = cur.fetchone()

            if row is None:
                return False
            
            return bool(row[0])

# ==================== USER FUNCTIONS ====================

def validate_manager_code(code: str) -> bool:
    """Check if the manager code is valid and active"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM manager_codes 
                WHERE code = %s AND is_active = TRUE
                """,
                (code,)
            )
            return cur.fetchone() is not None

def create_user(email: str, password: str, name: str, age: int, address: str, pincode: int, role: str = 'customer'):
    user_id = str(uuid.uuid4())
    pwd_plain = password  # no hashing

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, email, password_hash, name, age, address, pincode, role)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, email, pwd_plain, name, age, address, pincode, role)
            )
            conn.commit()

    return user_id


def get_user_by_email(email: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, email, password_hash, name, role
                FROM users
                WHERE email = %s
                """,
                (email,)
            )
            return cur.fetchone()


def authenticate_user(email: str, password: str):
    row = get_user_by_email(email)

    if not row:
        return None

    user_id, email, stored_password, name, role = row

    # plain-text comparison
    if password != stored_password:
        return None

    return {
        "user_id": str(user_id),
        "email": email,
        "name": name or email.split('@')[0],
        "role": role or 'customer'
    }

def get_all_users():
    """Get all users (for store manager view)"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, email, name, role, created_at
                FROM users
                ORDER BY created_at DESC
                """
            )
            return cur.fetchall()

# ==================== ORDER FUNCTIONS ====================

def create_order(user_id: str, products: dict):
    order_id = str(uuid.uuid4())

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO orders (id, user_id, products)
                VALUES (%s, %s, %s)
                """,
                (order_id, user_id, psycopg.types.json.Json(products))
            )
            conn.commit()

    return order_id

def get_orders_by_user(user_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT o.id, o.products, o.created_at
                FROM orders o 
                join chat_threads ct on ct.thread_id = o.user_id
                WHERE ct.user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,)
            )
            return cur.fetchall()

def get_all_orders(from_date=None, to_date=None, user_email=None):
    """Get all orders with optional filters (for store manager)"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT o.id, o.products, o.created_at, o.status, u.email, u.name
                FROM orders o 
                JOIN chat_threads ct ON ct.thread_id = o.user_id
                JOIN users u ON u.user_id::text = ct.user_id
                WHERE 1=1
            """
            params = []
            
            if from_date:
                query += " AND o.created_at >= %s"
                params.append(from_date)
            
            if to_date:
                query += " AND o.created_at <= %s"
                params.append(to_date + timedelta(days=1))
            
            if user_email:
                query += " AND u.email ILIKE %s"
                params.append(f"%{user_email}%")
            
            query += " ORDER BY o.created_at DESC"
            
            cur.execute(query, params)
            return cur.fetchall()

# ==================== TELEMETRY FUNCTIONS ====================

def log_telemetry_event(event_type: str, user_id: str = None, user_email: str = None, 
                        user_role: str = None, metadata: dict = None):
    """Log a telemetry event"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telemetry_events (event_type, user_id, user_email, user_role, metadata)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (event_type, user_id, user_email, user_role, 
                 psycopg.types.json.Json(metadata) if metadata else None)
            )
            conn.commit()

def log_model_performance(model_name: str, duration_ms: float, operation: str = None,
                          input_size: str = None, user_id: str = None, 
                          thread_id: str = None, metadata: dict = None):
    """Log model performance metrics"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_performance (model_name, operation, duration_ms, input_size, 
                                               user_id, thread_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (model_name, operation, duration_ms, input_size, user_id, thread_id,
                 psycopg.types.json.Json(metadata) if metadata else None)
            )
            conn.commit()

def get_telemetry_stats():
    """Get telemetry statistics for dashboard"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            stats = {}
            
            # Total users
            cur.execute("SELECT COUNT(*) FROM users")
            stats['total_users'] = cur.fetchone()[0]
            
            # Users by role
            cur.execute("SELECT role, COUNT(*) FROM users GROUP BY role")
            stats['users_by_role'] = dict(cur.fetchall())
            
            # Logins today
            cur.execute("""
                SELECT COUNT(*) FROM telemetry_events 
                WHERE event_type = 'login' AND created_at >= CURRENT_DATE
            """)
            stats['logins_today'] = cur.fetchone()[0]
            
            # Total logins
            cur.execute("SELECT COUNT(*) FROM telemetry_events WHERE event_type = 'login'")
            stats['total_logins'] = cur.fetchone()[0]
            
            # Total orders
            cur.execute("SELECT COUNT(*) FROM orders")
            stats['total_orders'] = cur.fetchone()[0]
            
            # Orders today
            cur.execute("SELECT COUNT(*) FROM orders WHERE created_at >= CURRENT_DATE")
            stats['orders_today'] = cur.fetchone()[0]
            
            # Login trends (last 7 days)
            cur.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM telemetry_events 
                WHERE event_type = 'login' AND created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            stats['login_trends'] = cur.fetchall()
            
            # Order trends (last 7 days)
            cur.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM orders 
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            stats['order_trends'] = cur.fetchall()
            
            return stats

def get_model_performance_stats():
    """Get model performance statistics"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            stats = {}
            
            # Average duration by model
            cur.execute("""
                SELECT model_name, 
                       AVG(duration_ms) as avg_ms,
                       MIN(duration_ms) as min_ms,
                       MAX(duration_ms) as max_ms,
                       COUNT(*) as count
                FROM model_performance
                GROUP BY model_name
            """)
            stats['by_model'] = cur.fetchall()
            
            # Recent performance (last 24 hours)
            cur.execute("""
                SELECT model_name, AVG(duration_ms) as avg_ms
                FROM model_performance
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY model_name
            """)
            stats['recent_avg'] = dict(cur.fetchall())
            
            # Performance trends (last 7 days)
            cur.execute("""
                SELECT DATE(created_at) as date, model_name, AVG(duration_ms) as avg_ms
                FROM model_performance
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(created_at), model_name
                ORDER BY date, model_name
            """)
            stats['trends'] = cur.fetchall()
            
            return stats

def get_transaction_stats():
    """Get transaction timing statistics"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            stats = {}
            
            # Average transaction time
            cur.execute("""
                SELECT AVG((metadata->>'duration_ms')::float) as avg_ms
                FROM telemetry_events 
                WHERE event_type = 'transaction_complete' 
                AND metadata->>'duration_ms' IS NOT NULL
            """)
            result = cur.fetchone()
            stats['avg_transaction_ms'] = result[0] if result[0] else 0
            
            # Transaction count
            cur.execute("""
                SELECT COUNT(*) FROM telemetry_events 
                WHERE event_type = 'transaction_complete'
            """)
            stats['total_transactions'] = cur.fetchone()[0]
            
            return stats
def get_user_age_by_thread(thread_id: str) -> int:
    """Get user age from thread_id by joining chat_threads and users tables"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.age
                FROM users u
                JOIN chat_threads ct ON ct.user_id = u.user_id::text
                WHERE ct.thread_id = %s
                """,
                (thread_id,)
            )
            row = cur.fetchone()
            if row is None:
                return 0
            return row[0] or 0

def get_user_role_by_thread(thread_id: str) -> str:
    """Get user role from thread_id by joining chat_threads and users tables"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.role
                FROM users u
                JOIN chat_threads ct ON ct.user_id = u.user_id::text
                WHERE ct.thread_id = %s
                """,
                (thread_id,)
            )
            row = cur.fetchone()
            if row is None:
                return "customer"
            return row[0] or "customer"


# ==================== MODEL VERIFICATION FUNCTIONS ====================

def log_model_verification(
    thread_id: str,
    inference_output: dict,
    llm_output: dict,
    match_status: bool,
    mismatched_items: dict = None
):
    """
    Log model verification data comparing inference output vs LLM output.
    
    Args:
        thread_id: The chat thread ID
        inference_output: Raw output from inference model (e.g., {"Candy": 3, "Alcohol": 2})
        llm_output: Parsed output from LLM response (e.g., {"Candy": 3, "Alcohol": 2})
        match_status: True if outputs match, False otherwise
        mismatched_items: Dict of items that differed (e.g., {"Alcohol": {"inference": 2, "llm": 1}})
    """
    import json
    with get_connection() as conn:
        with conn.cursor() as cur:
            total_inference = sum(inference_output.values()) if inference_output else 0
            total_llm = sum(llm_output.values()) if llm_output else 0
            
            cur.execute(
                """
                INSERT INTO model_verification_logs 
                (thread_id, inference_output, llm_output, match_status, mismatched_items, 
                 total_inference_items, total_llm_items)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    thread_id,
                    json.dumps(inference_output),
                    json.dumps(llm_output),
                    match_status,
                    json.dumps(mismatched_items) if mismatched_items else None,
                    total_inference,
                    total_llm
                )
            )
            conn.commit()


def compare_model_outputs(inference_output: dict, llm_output: dict) -> tuple:
    """
    Compare inference output with LLM output and identify mismatches.
    
    Returns:
        (match_status: bool, mismatched_items: dict)
    """
    mismatched_items = {}
    
    # Normalize keys to lowercase for comparison
    inference_normalized = {k.lower().strip(): v for k, v in (inference_output or {}).items()}
    llm_normalized = {k.lower().strip(): v for k, v in (llm_output or {}).items()}
    
    # Get all unique keys
    all_keys = set(inference_normalized.keys()) | set(llm_normalized.keys())
    
    for key in all_keys:
        inf_val = inference_normalized.get(key, 0)
        llm_val = llm_normalized.get(key, 0)
        
        if inf_val != llm_val:
            # Find original key name (prefer inference key)
            original_key = None
            for k in (inference_output or {}):
                if k.lower().strip() == key:
                    original_key = k
                    break
            if not original_key:
                for k in (llm_output or {}):
                    if k.lower().strip() == key:
                        original_key = k
                        break
            
            mismatched_items[original_key or key] = {
                "inference": inf_val,
                "llm": llm_val
            }
    
    match_status = len(mismatched_items) == 0
    return match_status, mismatched_items


def get_model_verification_stats():
    """Get model verification statistics for dashboard"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            stats = {}
            
            # Total verifications
            cur.execute("SELECT COUNT(*) FROM model_verification_logs")
            stats['total_verifications'] = cur.fetchone()[0]
            
            # Match count
            cur.execute("SELECT COUNT(*) FROM model_verification_logs WHERE match_status = TRUE")
            stats['matches'] = cur.fetchone()[0]
            
            # Mismatch count
            cur.execute("SELECT COUNT(*) FROM model_verification_logs WHERE match_status = FALSE")
            stats['mismatches'] = cur.fetchone()[0]
            
            # Agreement rate
            if stats['total_verifications'] > 0:
                stats['agreement_rate'] = round(
                    (stats['matches'] / stats['total_verifications']) * 100, 2
                )
            else:
                stats['agreement_rate'] = 100.0
            
            # Recent verifications (last 24 hours)
            cur.execute("""
                SELECT COUNT(*), 
                       SUM(CASE WHEN match_status THEN 1 ELSE 0 END) as matches
                FROM model_verification_logs
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            row = cur.fetchone()
            stats['recent_total'] = row[0] or 0
            stats['recent_matches'] = row[1] or 0
            if stats['recent_total'] > 0:
                stats['recent_agreement_rate'] = round(
                    (stats['recent_matches'] / stats['recent_total']) * 100, 2
                )
            else:
                stats['recent_agreement_rate'] = 100.0
            
            # Trends (last 7 days)
            cur.execute("""
                SELECT DATE(created_at) as date,
                       COUNT(*) as total,
                       SUM(CASE WHEN match_status THEN 1 ELSE 0 END) as matches
                FROM model_verification_logs
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            stats['trends'] = cur.fetchall()
            
            return stats


def get_model_verification_logs(match_filter: str = None, limit: int = 50):
    """
    Get recent model verification logs for audit.
    
    Args:
        match_filter: 'match', 'mismatch', or None for all
        limit: Max number of records to return
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT thread_id, inference_output, llm_output, match_status, 
                       mismatched_items, total_inference_items, total_llm_items, created_at
                FROM model_verification_logs
            """
            
            params = []
            if match_filter == 'match':
                query += " WHERE match_status = TRUE"
            elif match_filter == 'mismatch':
                query += " WHERE match_status = FALSE"
            
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            
            rows = cur.fetchall()
            logs = []
            for row in rows:
                logs.append({
                    'thread_id': str(row[0]) if row[0] else None,
                    'inference_output': row[1],
                    'llm_output': row[2],
                    'match_status': row[3],
                    'mismatched_items': row[4],
                    'total_inference_items': row[5],
                    'total_llm_items': row[6],
                    'created_at': row[7]
                })
            return logs
