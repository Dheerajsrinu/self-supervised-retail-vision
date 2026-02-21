import streamlit as st
from views.auth_view import render_auth_view
from app.backend.db import init_db, get_orders_by_user
from app.ui.styles import (
    apply_custom_styles, 
    render_header_compact, 
    render_sidebar_logo, 
    render_user_profile_bottom, 
    render_stat_card, 
    render_empty_state,
    COLORS
)

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="My Orders | Retail Assistant",
    page_icon="🛒",
    layout="wide"
)

# -------------------------------------------------
# Apply custom styles
# -------------------------------------------------
apply_custom_styles()

# -------------------------------------------------
# Auth check
# -------------------------------------------------
if "user_id" not in st.session_state:
    render_auth_view()
    st.stop()

user_id = st.session_state.user_id

# -------------------------------------------------
# Init DB
# -------------------------------------------------
init_db()

# -------------------------------------------------
# Sidebar (matching chatbot layout)
# -------------------------------------------------
with st.sidebar:
    # Logo at top
    render_sidebar_logo()
    
    st.markdown("")  # Spacing
    
    # ----- Navigation -----
    st.markdown("##### Navigation")
    
    # Chat button
    if st.button("Chat Assistant", use_container_width=True, key="nav_chat"):
        st.switch_page("chatbot.py")
    
    # Manager Dashboard (only for store managers)
    if st.session_state.get("user_role") == "store_manager":
        if st.button("Manager Dashboard", use_container_width=True, key="nav_manager"):
            st.switch_page("pages/manager_dashboard.py")
    
    # ----- Spacer to push bottom section down -----
    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    
    # ----- Fixed Bottom Section -----
    st.markdown("---")  # Divider
    
    # Logout button (fixed position)
    if st.button("Sign Out", use_container_width=True, key="logout"):
        st.session_state.clear()
        st.rerun()
    
    # User profile at bottom (fixed position)
    user_email = st.session_state.get("user_email", "User")
    render_user_profile_bottom(user_email)

# -------------------------------------------------
# Main Content
# -------------------------------------------------
render_header_compact("My Orders", "View your order history")

orders = get_orders_by_user(user_id)

if not orders:
    render_empty_state("You haven't placed any orders yet", "")
    
    st.markdown("")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Shopping", use_container_width=True, type="primary"):
            st.switch_page("chatbot.py")
    st.stop()

# -------------------------------------------------
# Metrics Overview
# -------------------------------------------------
st.markdown("##### Overview")

total_orders = len(orders)
total_items = sum(sum(products.values()) for _, products, _ in orders)
unique_products = len(set(product for _, products, _ in orders for product in products.keys()))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", total_orders)
    
with col2:
    st.metric("Items Purchased", total_items)
    
with col3:
    st.metric("Unique Products", unique_products)
    
with col4:
    avg_items = round(total_items / total_orders, 1) if total_orders > 0 else 0
    st.metric("Avg Items/Order", avg_items)

st.markdown("")
st.divider()

# -------------------------------------------------
# Orders List
# -------------------------------------------------
st.markdown("##### Order History")

# Filter options
col1, col2 = st.columns([3, 1])
with col2:
    sort_order = st.selectbox(
        "Sort by",
        ["Newest First", "Oldest First"],
        label_visibility="collapsed"
    )

if sort_order == "Oldest First":
    orders = list(reversed(orders))

# Order cards
for order_id, products, created_at in orders:
    total_items = sum(products.values())
    
    with st.container():
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 0.75rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.75rem; color: #718096;">Order ID</div>
                    <div style="font-family: monospace; font-size: 0.85rem; color: #1a365d;">{str(order_id)[:8]}...</div>
                </div>
                <div style="text-align: center;">
                    <div style="
                        background: #edf2f7;
                        color: #1a365d;
                        padding: 0.25rem 0.75rem;
                        border-radius: 20px;
                        font-size: 0.8rem;
                        font-weight: 500;
                    ">{total_items} items</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #718096;">Placed on</div>
                    <div style="font-weight: 500; color: #2d3748; font-size: 0.85rem;">{created_at.strftime('%b %d, %Y')}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Product details expander
        with st.expander(f"View Details", expanded=False):
            # Product grid
            cols = st.columns(4)
            for idx, (product_name, quantity) in enumerate(products.items()):
                with cols[idx % 4]:
                    st.markdown(f"""
                    <div style="
                        background: #f7fafc;
                        border-radius: 8px;
                        padding: 0.75rem;
                        text-align: center;
                        margin-bottom: 0.5rem;
                    ">
                        <div style="font-weight: 500; font-size: 0.85rem; color: #1a365d; margin-bottom: 0.25rem;">{product_name}</div>
                        <div style="
                            background: #1a365d;
                            color: white;
                            border-radius: 15px;
                            padding: 0.2rem 0.6rem;
                            font-size: 0.75rem;
                            display: inline-block;
                        ">x {quantity}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="
                border-top: 1px solid #e2e8f0;
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: #718096; font-size: 0.9rem;">Total Items</span>
                <span style="font-weight: 600; font-size: 1rem; color: #1a365d;">{total_items}</span>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("")
st.divider()

st.markdown("""
<div style="text-align: center; color: #a0aec0; font-size: 0.8rem; padding: 0.5rem 0;">
    Need help? Contact support@iiith.ac.in
</div>
""", unsafe_allow_html=True)
