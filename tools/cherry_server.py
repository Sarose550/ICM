#!/usr/bin/env python3
"""
Cherry Servers management helper for the ICM Zen 4 machine.

Usage:
  python3 tools/cherry_server.py status          # Show server status
  python3 tools/cherry_server.py poweron          # Power on the server
  python3 tools/cherry_server.py poweroff         # Power off the server
  python3 tools/cherry_server.py reboot           # Reboot the server

Requires: pip install cherry-python
Set CHERRY_API_TOKEN environment variable or edit TOKEN below.
"""

import os
import sys
from cherry import Master

TOKEN = os.environ.get("CHERRY_API_TOKEN", "")

if not TOKEN:
    print("Set CHERRY_API_TOKEN environment variable first.")
    print("Get your token from: Cherry Servers Portal → API Keys")
    sys.exit(1)

client = Master(auth_token=TOKEN)

def find_server():
    """Find the Zen 4 server (84.32.186.178) across all projects."""
    teams = client.get_teams()
    for team in teams:
        team_id = team.get("id")
        projects = client.get_projects(team_id)
        for project in projects:
            project_id = project.get("id")
            servers = client.get_servers(project_id)
            for server in servers:
                # Match by IP or hostname
                ip = server.get("ip_addresses", [{}])[0].get("address", "") if server.get("ip_addresses") else ""
                name = server.get("name", "")
                hostname = server.get("hostname", "")
                if "84.32.186.178" in str(server) or "darling-shrew" in str(server).lower():
                    return server
                # Also check primary IP
                for ip_info in server.get("ip_addresses", []):
                    if ip_info.get("address") == "84.32.186.178":
                        return server
    return None

def show_status():
    server = find_server()
    if not server:
        print("Server not found. Listing all servers...")
        teams = client.get_teams()
        for team in teams:
            projects = client.get_projects(team["id"])
            for project in projects:
                servers = client.get_servers(project["id"])
                for s in servers:
                    print(f"  ID={s.get('id')} name={s.get('name')} hostname={s.get('hostname')} status={s.get('status')} power={s.get('power_state')}")
        return

    print(f"Server: {server.get('name', 'unknown')} (ID: {server.get('id')})")
    print(f"  Hostname: {server.get('hostname', 'unknown')}")
    print(f"  Status:   {server.get('status', 'unknown')}")
    print(f"  Power:    {server.get('power_state', 'unknown')}")
    for ip in server.get("ip_addresses", []):
        print(f"  IP:       {ip.get('address', '?')} ({ip.get('type', '?')})")

def power_action(action):
    server = find_server()
    if not server:
        print("Server not found!")
        return

    server_id = server["id"]
    print(f"Performing {action} on server {server_id} ({server.get('name', '')})...")

    if action == "poweron":
        result = client.poweron_server(server_id)
    elif action == "poweroff":
        result = client.poweroff_server(server_id)
    elif action == "reboot":
        result = client.reboot_server(server_id)
    else:
        print(f"Unknown action: {action}")
        return

    print(f"Done. Response: {result}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "status":
        show_status()
    elif cmd in ("poweron", "poweroff", "reboot"):
        power_action(cmd)
    else:
        print(f"Unknown command: {cmd}")
        print("Use: status, poweron, poweroff, reboot")
