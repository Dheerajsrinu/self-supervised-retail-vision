import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from views.auth_view import render_auth_view
from app.backend.db import (
    init_db, 
    get_all_users, 
    get_all_orders,
    get_telemetry_stats,
    get_model_performance_stats,
    get_transaction_stats,
    get_model_verification_stats,
    get_model_verification_logs
)
from app.ui.styles import apply_custom_styles, COLORS

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Manager Dashboard | Retail Assistant",
    page_icon="🛒",
    layout="wide"
)

# -------------------------------------------------
# Apply custom styles
# -------------------------------------------------
apply_custom_styles()

# Hide default navigation
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Auth check
# -------------------------------------------------
if "user_id" not in st.session_state:
    render_auth_view()
    st.stop()

# Check if user is store manager
if st.session_state.get("user_role") != "store_manager":
    st.error("Access Denied. This page is only for Store Managers.")
    if st.button("Go to Chat"):
        st.switch_page("chatbot.py")
    st.stop()

# -------------------------------------------------
# Init DB
# -------------------------------------------------
init_db()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 1.5rem 1rem 1rem;
        margin: -1rem -1rem 0 -1rem;
        background: rgba(0,0,0,0.1);
    ">
        <div style="
            width: 60px;
            height: 60px;
            margin: 0 auto 0.5rem;
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid rgba(255,255,255,0.3);
        ">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 3H5L5.4 5M7 13H17L21 5H5.4M7 13L5.4 5M7 13L4.707 15.293C4.077 15.923 4.523 17 5.414 17H17M17 17C15.895 17 15 17.895 15 19C15 20.105 15.895 21 17 21C18.105 21 19 20.105 19 19C19 17.895 18.105 17 17 17ZM9 19C9 20.105 8.105 21 7 21C5.895 21 5 20.105 5 19C5 17.895 5.895 17 7 17C8.105 17 9 17.895 9 19Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div style="font-size: 1rem; font-weight: 600;">Manager Portal</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### Navigation")
    
    if st.button("Chat Assistant", use_container_width=True, key="nav_chat"):
        st.switch_page("chatbot.py")
    
    if st.button("My Orders", use_container_width=True, key="nav_orders"):
        st.switch_page("pages/orders_dashboard.py")
    
    # ----- Spacer to push bottom section down -----
    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    
    # ----- Fixed Bottom Section -----
    st.markdown("---")  # Divider
    
    if st.button("Sign Out", use_container_width=True, key="logout"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown(f"""
    <div style="
        background: rgba(0,0,0,0.15);
        border-radius: 10px;
        padding: 0.75rem;
    ">
        <div style="font-size: 0.7rem; opacity: 0.7;">Signed in as</div>
        <div style="font-size: 0.85rem; font-weight: 500;">{st.session_state.get('user_email', 'Manager')}</div>
        <div style="font-size: 0.7rem; color: #38a169;">Store Manager</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# Main Content
# -------------------------------------------------
st.markdown("## Manager Dashboard")
st.caption("Overview of users, orders, and system performance")

# -------------------------------------------------
# Dashboard Tabs
# -------------------------------------------------
tab_overview, tab_users, tab_orders, tab_telemetry = st.tabs([
    "Overview", "Users", "All Orders", "Telemetry"
])

# -------------------------------------------------
# OVERVIEW TAB
# -------------------------------------------------
with tab_overview:
    try:
        stats = get_telemetry_stats()
        model_stats = get_model_performance_stats()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", stats.get('total_users', 0))
        with col2:
            st.metric("Logins Today", stats.get('logins_today', 0))
        with col3:
            st.metric("Total Orders", stats.get('total_orders', 0))
        with col4:
            st.metric("Orders Today", stats.get('orders_today', 0))
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Login Activity (Last 7 Days)")
            login_trends = stats.get('login_trends', [])
            if login_trends:
                df = pd.DataFrame(login_trends, columns=['Date', 'Logins'])
                df['Date'] = pd.to_datetime(df['Date'])
                st.line_chart(df.set_index('Date'))
            else:
                st.info("No login data available yet")
        
        with col2:
            st.markdown("#### Order Activity (Last 7 Days)")
            order_trends = stats.get('order_trends', [])
            if order_trends:
                df = pd.DataFrame(order_trends, columns=['Date', 'Orders'])
                df['Date'] = pd.to_datetime(df['Date'])
                st.line_chart(df.set_index('Date'))
            else:
                st.info("No order data available yet")
        
        # Users by role
        st.divider()
        st.markdown("#### Users by Role")
        users_by_role = stats.get('users_by_role', {})
        if users_by_role:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Customers", users_by_role.get('customer', 0))
            with col2:
                st.metric("Store Managers", users_by_role.get('store_manager', 0))
            with col3:
                st.metric("Total", sum(users_by_role.values()))
    except Exception as e:
        st.error(f"Error loading overview: {e}")

# -------------------------------------------------
# USERS TAB
# -------------------------------------------------
with tab_users:
    st.markdown("#### All Registered Users")
    
    try:
        users = get_all_users()
        
        if users:
            # Convert to DataFrame
            df = pd.DataFrame(users, columns=['User ID', 'Email', 'Name', 'Role', 'Created At'])
            df['User ID'] = df['User ID'].astype(str)  # Convert UUID to string
            df['Created At'] = pd.to_datetime(df['Created At']).dt.strftime('%Y-%m-%d %H:%M')
            df['Role'] = df['Role'].apply(lambda x: 'Store Manager' if x == 'store_manager' else 'Customer')
            
            # Filters
            col1, col2 = st.columns([2, 1])
            with col1:
                search = st.text_input("Search by email or name", key="user_search")
            with col2:
                role_filter = st.selectbox("Filter by Role", ["All", "Customer", "Store Manager"])
            
            # Apply filters
            if search:
                df = df[df['Email'].str.contains(search, case=False) | 
                       df['Name'].str.contains(search, case=False, na=False)]
            if role_filter != "All":
                df = df[df['Role'] == role_filter]
            
            st.markdown(f"**{len(df)} users found**")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No users registered yet")
    except Exception as e:
        st.error(f"Error loading users: {e}")

# -------------------------------------------------
# ORDERS TAB
# -------------------------------------------------
with tab_orders:
    st.markdown("#### All Customer Orders")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        from_date = st.date_input(
            "From Date",
            value=datetime.now().date() - timedelta(days=30),
            key="order_from_date"
        )
    
    with col2:
        to_date = st.date_input(
            "To Date",
            value=datetime.now().date(),
            key="order_to_date"
        )
    
    with col3:
        user_filter = st.text_input(
            "Filter by User Email",
            placeholder="Search email...",
            key="order_user_filter"
        )
    
    try:
        orders = get_all_orders(
            from_date=datetime.combine(from_date, datetime.min.time()),
            to_date=datetime.combine(to_date, datetime.max.time()),
            user_email=user_filter if user_filter else None
        )
        
        if orders:
            st.markdown(f"**{len(orders)} orders found**")
            
            for order_id, products, created_at, status, user_email, user_name in orders:
                total_items = sum(products.values()) if isinstance(products, dict) else 0
                
                with st.expander(
                    f"Order #{str(order_id)[:8]} | {user_email} | {created_at.strftime('%Y-%m-%d %H:%M')} | {total_items} items"
                ):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**Customer Details**")
                        st.write(f"Name: {user_name or 'N/A'}")
                        st.write(f"Email: {user_email}")
                        st.write(f"Status: {status or 'completed'}")
                    
                    with col2:
                        st.markdown("**Products**")
                        if isinstance(products, dict):
                            for product, qty in products.items():
                                st.write(f"- {product}: {qty}")
                        else:
                            st.write(products)
        else:
            st.info("No orders found for the selected filters")
    except Exception as e:
        st.error(f"Error loading orders: {e}")

# -------------------------------------------------
# TELEMETRY TAB
# -------------------------------------------------
with tab_telemetry:
    st.markdown("#### System Telemetry & Performance")
    
    try:
        model_stats = get_model_performance_stats()
        transaction_stats = get_transaction_stats()
        
        # Model Performance
        st.markdown("##### Model Performance")
        
        by_model = model_stats.get('by_model', [])
        if by_model:
            # Convert ms to seconds for display
            df = pd.DataFrame(by_model, columns=['Model', 'Avg (ms)', 'Min (ms)', 'Max (ms)', 'Count'])
            df['Avg (s)'] = (df['Avg (ms)'] / 1000).round(3)
            df['Min (s)'] = (df['Min (ms)'] / 1000).round(3)
            df['Max (s)'] = (df['Max (ms)'] / 1000).round(3)
            
            # Display table with seconds
            display_df = df[['Model', 'Avg (s)', 'Min (s)', 'Max (s)', 'Count']]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Bar chart in seconds
            st.bar_chart(df.set_index('Model')['Avg (s)'])
        else:
            st.info("No model performance data yet. Data will appear as users interact with the system.")
        
        st.divider()
        
        # Transaction Stats
        st.markdown("##### Transaction Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_time_ms = transaction_stats.get('avg_transaction_ms', 0)
            avg_time_s = avg_time_ms / 1000 if avg_time_ms else 0
            st.metric("Avg Transaction Time", f"{avg_time_s:.2f} s" if avg_time_s else "N/A")
        
        with col2:
            total_tx = transaction_stats.get('total_transactions', 0)
            st.metric("Total Transactions", total_tx)
        
        st.divider()
        
        # LLM/Chatbot Performance
        st.markdown("##### Chatbot Response Statistics")
        col1, col2, col3 = st.columns(3)
        
        # Get LLM specific stats
        llm_stats = model_stats.get('by_model', [])
        llm_data = [row for row in llm_stats if 'llm' in row[0].lower() or 'gpt' in row[0].lower() or 'chatbot' in row[0].lower()]
        
        with col1:
            if llm_data:
                avg_llm_ms = llm_data[0][1] if llm_data else 0
                avg_llm_s = avg_llm_ms / 1000 if avg_llm_ms else 0
                st.metric("Avg LLM Response", f"{avg_llm_s:.2f} s")
            else:
                st.metric("Avg LLM Response", "N/A")
        
        with col2:
            if llm_data:
                llm_count = llm_data[0][4] if llm_data else 0
                st.metric("Total LLM Calls", llm_count)
            else:
                st.metric("Total LLM Calls", 0)
        
        with col3:
            # Average chatbot response = transaction time (includes all processing)
            if avg_time_s:
                st.metric("Avg Chatbot Response", f"{avg_time_s:.2f} s")
            else:
                st.metric("Avg Chatbot Response", "N/A")
        
        st.divider()
        
        # Recent performance
        st.markdown("##### Recent Model Performance (Last 24 Hours)")
        recent_avg = model_stats.get('recent_avg', {})
        if recent_avg:
            cols = st.columns(min(len(recent_avg), 4))
            for i, (model, avg_ms) in enumerate(recent_avg.items()):
                with cols[i % 4]:
                    avg_s = avg_ms / 1000
                    st.metric(model, f"{avg_s:.2f} s")
        else:
            st.info("No recent performance data available")
        
        st.divider()
        
        # Model Verification - Inference vs LLM Agreement
        st.markdown("##### Model Verification (Inference vs LLM)")
        st.caption("Compares product detection from ML models with what LLM reports to users")
        
        verification_stats = get_model_verification_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            agreement_rate = verification_stats.get('agreement_rate', 100.0)
            st.metric("Agreement Rate", f"{agreement_rate}%")
        
        with col2:
            total_verifications = verification_stats.get('total_verifications', 0)
            st.metric("Total Verifications", total_verifications)
        
        with col3:
            matches = verification_stats.get('matches', 0)
            st.metric("Matches", matches)
        
        with col4:
            mismatches = verification_stats.get('mismatches', 0)
            delta_color = "off" if mismatches == 0 else "inverse"
            st.metric("Mismatches", mismatches)
        
        # Recent agreement rate (24 hours)
        st.markdown("###### Recent Verification (Last 24 Hours)")
        recent_col1, recent_col2 = st.columns(2)
        
        with recent_col1:
            recent_rate = verification_stats.get('recent_agreement_rate', 100.0)
            st.metric("Recent Agreement Rate", f"{recent_rate}%")
        
        with recent_col2:
            recent_total = verification_stats.get('recent_total', 0)
            st.metric("Recent Verifications", recent_total)
        
        # Verification logs viewer
        st.markdown("###### Verification Logs")
        
        filter_option = st.selectbox(
            "Filter by status:",
            ["All", "Matches Only", "Mismatches Only"],
            key="verification_filter"
        )
        
        filter_map = {
            "All": None,
            "Matches Only": "match",
            "Mismatches Only": "mismatch"
        }
        
        logs = get_model_verification_logs(match_filter=filter_map[filter_option], limit=20)
        
        if logs:
            for log in logs:
                match_icon = "Match" if log['match_status'] else "MISMATCH"
                match_color = "green" if log['match_status'] else "red"
                
                with st.expander(f"{match_icon} - {log['created_at'].strftime('%Y-%m-%d %H:%M:%S') if log['created_at'] else 'N/A'}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Inference Output:**")
                        st.json(log['inference_output'])
                    
                    with col2:
                        st.markdown("**LLM Output:**")
                        st.json(log['llm_output'])
                    
                    if not log['match_status'] and log['mismatched_items']:
                        st.markdown("**Mismatched Items:**")
                        st.json(log['mismatched_items'])
        else:
            st.info("No verification logs available yet. Logs will appear after image-based product recognition.")
            
    except Exception as e:
        st.error(f"Error loading telemetry: {e}")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption("Manager Dashboard - Retail Assistant")
