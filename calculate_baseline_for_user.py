#!/usr/bin/env python3
"""
Script to calculate baseline stats for a specific user.
Retrieves sessions from the first two uploads and calculates baseline stats using the current logic.

Usage: python calculate_baseline_for_user.py <user_id>
"""

import sys
from collections import defaultdict
from datetime import datetime
from database import SessionLocal, User, Session as SessionModel
from db_service import DatabaseService

def calculate_baseline_stats(sessions, apps_to_monitor=None):
    """
    Calculate baseline stats from sessions.
    Exactly matches the logic in SmartPauseApp.py
    """
    if apps_to_monitor is None:
        apps_to_monitor = []
    
    if not sessions:
        return {
            "median_target_app_usage_seconds": None,
            "median_session_usage_seconds": None,
            "query_interval_seconds": None
        }
    
    # Group sessions by group_id AND date
    groups = defaultdict(list)
    for session in sessions:
        groups[(session.group_id, session.date)].append(session)
    
    # Calculate target app duration per group AND total duration per group
    group_target_durations = []
    group_total_durations = []
    all_session_durations = []
    
    print(f"\n📊 Grouping sessions:")
    print(f"   Total groups found: {len(groups)}")
    
    for (group_id, date), group_sessions in groups.items():
        # Sum up target app durations in this group
        group_target_duration = sum(
            s.duration_seconds for s in group_sessions 
            if s.app_name in apps_to_monitor
        )
        if group_target_duration > 0:
            group_target_durations.append(group_target_duration)
        
        # Sum up ALL session durations in this group
        group_total_duration = sum(s.duration_seconds for s in group_sessions)
        group_total_durations.append(group_total_duration)
        
        # Collect all individual session durations
        for s in group_sessions:
            all_session_durations.append(s.duration_seconds)
        
        print(f"   Group (id={group_id}, date={date}): {len(group_sessions)} sessions")
        print(f"      Total duration: {group_total_duration}s, Target app duration: {group_target_duration}s")
    
    # Calculate medians
    if not group_target_durations:
        median_target_usage = None
    else:
        group_target_durations.sort()
        median_target_usage = group_target_durations[len(group_target_durations) // 2]
    
    if not group_total_durations:
        median_session_usage = None
    else:
        group_total_durations.sort()
        median_session_usage = group_total_durations[len(group_total_durations) // 2]
    
    # 75th percentile for query interval
    if not all_session_durations:
        percentile_75 = None
    else:
        all_session_durations.sort()
        percentile_75_index = int(len(all_session_durations) * 0.75)
        percentile_75 = all_session_durations[percentile_75_index] if percentile_75_index < len(all_session_durations) else all_session_durations[-1]
    
    return {
        "median_target_app_usage_seconds": median_target_usage,
        "median_session_usage_seconds": median_session_usage,
        "query_interval_seconds": percentile_75
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_baseline_for_user.py <user_id>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    
    db = SessionLocal()
    
    try:
        # Check if user exists
        user = DatabaseService.get_user(db, user_id)
        if not user:
            print(f"❌ User '{user_id}' not found in database")
            return
        
        print(f"\n✅ Found user: {user_id}")
        print(f"   Created at: {user.created_at}")
        print(f"   Apps to monitor: {user.apps_to_monitor}")
        print(f"   Current day: {user.current_day}")
        
        # Check if user has completed baseline period (day 2+)
        print(f"\n📤 User is currently on day: {user.current_day}")
        
        if user.current_day < 2:
            print(f"⚠️  User is only on day {user.current_day}. Baseline stats are calculated after day 2.")
            print(f"   Wait for the user to reach day 2 before baseline stats can be calculated.")
            return
        
        # Get baseline sessions (first 2 uploads)
        baseline_sessions = DatabaseService.get_baseline_sessions(db, user_id)
        print(f"📊 Baseline sessions (1st and 2nd upload): {len(baseline_sessions)} sessions")
        
        if not baseline_sessions:
            print("❌ No sessions found for baseline calculation")
            return
        
        # Display session details
        print(f"\n📋 Session Details:")
        unique_dates = set()
        apps_found = set()
        for session in baseline_sessions:
            unique_dates.add(session.date)
            apps_found.add(session.app_name)
            print(f"   {session.date} | {session.app_name:20} | {session.duration_seconds:6.0f}s | Group: {session.group_id}")
        
        print(f"\n📅 Unique dates in baseline period: {sorted(unique_dates)}")
        print(f"📱 Apps found in baseline period: {sorted(apps_found)}")
        
        # Calculate baseline stats
        apps_to_monitor = user.apps_to_monitor if user.apps_to_monitor else []
        print(f"\n🎯 Using apps_to_monitor: {apps_to_monitor}")
        
        stats = calculate_baseline_stats(baseline_sessions, apps_to_monitor)
        
        # Display results
        print(f"\n✅ BASELINE STATS CALCULATED:")
        print(f"   median_target_app_usage_seconds: {stats['median_target_app_usage_seconds']}")
        print(f"   median_session_usage_seconds: {stats['median_session_usage_seconds']}")
        print(f"   query_interval_seconds: {stats['query_interval_seconds']}")
        
        # Check if baseline stats already exist in database
        existing_stats = DatabaseService.get_baseline_stats(db, user_id)
        if existing_stats:
            print(f"\n📊 Baseline stats already in database:")
            print(f"   median_target_app_usage_seconds: {existing_stats.median_target_app_usage_seconds}")
            print(f"   median_session_usage_seconds: {existing_stats.median_session_usage_seconds}")
            print(f"   query_interval_seconds: {existing_stats.query_interval_seconds}")
        else:
            print(f"\n⚠️  No baseline stats stored in database yet")
            
            # Option to save
            response = input("\n💾 Save these baseline stats to database? (y/n): ").strip().lower()
            if response == 'y':
                DatabaseService.save_baseline_stats(db, user_id, stats)
                print("✅ Baseline stats saved to database!")
    
    finally:
        db.close()


if __name__ == "__main__":
    main()
